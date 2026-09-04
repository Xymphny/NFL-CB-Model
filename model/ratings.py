"""
Baseline computation, opponent adjustment, and team rating aggregation.

Spec references: Sections 3.3 (buckets), 3.4 (raw VOA), 3.5 (opponent
adjustment). Recency weighting is the accuracy improvement discussed
separately from the original spec.
"""

import numpy as np
import pandas as pd

from model.play_value import score_play, expected_turnover_value


def add_situation_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Section 3.3 — coarse bucketing (decided in Section 7: start coarse).

    Buckets: down x distance-bucket x field-zone.
    """
    df = df.copy()

    def distance_bucket(ydstogo):
        if pd.isna(ydstogo):
            return "unknown"
        if ydstogo <= 3:
            return "short"
        elif ydstogo <= 7:
            return "medium"
        return "long"

    def field_zone(yardline_100):
        if pd.isna(yardline_100):
            return "unknown"
        if yardline_100 > 80:
            return "own_1_20"
        elif yardline_100 > 50:
            return "own_21_50"
        elif yardline_100 > 20:
            return "opp_49_20"
        return "red_zone"

    df["distance_bucket"] = df["ydstogo"].apply(distance_bucket)
    df["field_zone"] = df["yardline_100"].apply(field_zone)
    df["down"] = df["down"].fillna(0).astype(int)
    df["bucket"] = (
        df["down"].astype(str) + "_" + df["distance_bucket"] + "_" + df["field_zone"]
    )
    return df


def score_all_plays(df: pd.DataFrame, use_turnover_luck_adjustment: bool = True, league: str = "NFL") -> pd.DataFrame:
    """
    Apply Section 3.2 scoring to every play, optionally with the
    turnover-luck accuracy adjustment applied to fumbles.
    """
    df = df.copy()
    if use_turnover_luck_adjustment:
        df["play_value"] = df.apply(lambda row: expected_turnover_value(row, league=league), axis=1)
    else:
        df["play_value"] = df.apply(lambda row: score_play(row, league=league), axis=1)
    return df


def compute_baselines(df: pd.DataFrame) -> pd.Series:
    """
    Section 3.3/3.4 — league-average play_value per bucket, for this season.
    """
    return df.groupby("bucket")["play_value"].mean()


def compute_raw_voa(df: pd.DataFrame, baselines: pd.Series) -> pd.DataFrame:
    """
    Section 3.4:
        VOA_play = (play_value - baseline_value_for_bucket) / |baseline_value|

    REAL BUG FOUND AND FIXED: the previous guard only protected against
    an EXACTLY zero baseline, not a NEAR-zero one -- found via a real
    validation failure (CAR's offense_voa computed as 1.23, wildly
    outside the normal +/-0.4 range) traced to a single play with
    baseline_value=-0.030 producing voa=30.05 from dividing by a
    near-zero denominator. Checked the real baseline distribution
    before picking a fix: typical |baseline| values run 0.67-1.27
    (25th-75th percentile), with only 4 of 46 buckets under 0.15 --
    a floor of 0.2 fixes the genuine small-sample outlier buckets
    while leaving the other 42+ well-populated buckets completely
    unaffected.
    """
    df = df.copy()
    df["baseline_value"] = df["bucket"].map(baselines)
    MIN_ABS_BASELINE = 0.2
    safe_baseline = df["baseline_value"].abs().clip(lower=MIN_ABS_BASELINE)
    df["voa"] = (df["play_value"] - df["baseline_value"]) / safe_baseline
    df["voa"] = df["voa"].fillna(0.0)
    return df


def opponent_adjust(df: pd.DataFrame, iterations: int = 3, regression: float = 0.5) -> pd.DataFrame:
    """
    Section 3.5 — iterative opponent adjustment.

    Each pass:
      1. Compute each defense's average VOA allowed (by bucket).
      2. Adjust each play's VOA by the opponent's allowed-VOA in that bucket,
         regressed partway toward 0 (league average) to avoid overcorrecting
         on small samples.
      3. Repeat, since adjusting offenses shifts each defense's own average.
    """
    df = df.copy()
    df["voa_adj"] = df["voa"]

    for _ in range(iterations):
        allowed = (
            df.groupby(["defteam", "bucket"])["voa_adj"]
            .mean()
            .rename("defense_allowed_voa")
        )
        df = df.merge(allowed, on=["defteam", "bucket"], how="left")
        df["defense_allowed_voa"] = df["defense_allowed_voa"].fillna(0.0)

        # Regress the adjustment partway toward 0 rather than subtracting
        # the full opponent effect — avoids overcorrecting on thin samples.
        df["voa_adj"] = df["voa_adj"] - regression * df["defense_allowed_voa"]
        df = df.drop(columns=["defense_allowed_voa"])

    return df


def add_recency_weights(df: pd.DataFrame, half_life_weeks: float = 100.0) -> pd.DataFrame:
    """
    Recency weighting (accuracy improvement, not in original spec).

    Exponential decay by week-distance from the most recent week in the
    data, so a next-game prediction leans more on recent form than a flat
    season average would.

    DEFAULT CHANGED FROM 6.0 TO 100.0 (effectively minimal weighting)
    based on real, held-out walk-forward calibration
    (model/calibrate_recency_half_life.py): tested 8 candidate values,
    selecting via training-set MAE alone actively picked the WORST
    real-world choice, because training and test performance were
    directly opposed -- training MAE monotonically favored shorter
    half-lives (more aggressive weighting), while held-out test MAE
    monotonically favored longer ones. The old default (6 weeks) gave
    the worst straight-up accuracy of everything tested (56.25%); a
    long half-life (100, effectively flat season-long averaging) gave
    the best (59.13%) -- consistent and monotonic across all 8 tested
    values on the held-out 2023 season, not a single lucky data point.
    Likely explanation: NFL teams don't show strong week-to-week "hot
    streak" signal independent of true season-long quality -- 
    discounting earlier-season data trades away real sample size for
    responsiveness to what's often just noise.
    """
    df = df.copy()
    max_week = df["week"].max()
    weeks_ago = max_week - df["week"]
    decay_rate = np.log(2) / half_life_weeks
    df["recency_weight"] = np.exp(-decay_rate * weeks_ago)
    return df


def filter_garbage_time(df: pd.DataFrame, wp_low: float = 0.05, wp_high: float = 0.95) -> pd.DataFrame:
    """
    Section 3.6 — garbage-time filter.

    Uses nflverse's own precomputed win-probability field rather than
    building a win-probability model from scratch — nflfastR already
    ships one. Excludes plays where the outcome is essentially decided
    (win probability outside the 5-95% band).
    """
    df = df.copy()
    if "wp" not in df.columns:
        print("[ratings] warning: no wp column found, skipping garbage-time filter")
        return df
    before = len(df)
    df = df[(df["wp"] >= wp_low) & (df["wp"] <= wp_high)].copy()
    print(f"[ratings] garbage-time filter: {before} -> {len(df)} plays ({before - len(df)} excluded)")
    return df


def add_home_field_and_rest(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Section 11.2 — home field / rest, applied as an additive term.

    Joins schedule-level fields (home/away, rest days, neutral site) onto
    the play-level data by game_id, then computes each play's home-field
    and rest adjustment. Neutral-site games get no home-field credit for
    either side, per Section 11.2's explicit design requirement.
    """
    df = df.copy()
    sched = schedules[[
        "game_id", "home_team", "away_team", "home_rest", "away_rest", "is_neutral_site"
    ]].copy()
    df = df.merge(sched, on="game_id", how="left", suffixes=("", "_sched"))

    df["is_home_offense"] = (df["posteam"] == df["home_team"])

    # Home-field adjustment: a small positive nudge to the offense's value
    # when they're the home team, zero at a neutral site. Magnitude here is
    # a placeholder pending real calibration (see calibrate_points_model.py)
    # rather than a number pulled from nowhere.
    df["home_field_adj"] = 0.0
    df.loc[df["is_home_offense"] & ~df["is_neutral_site"].fillna(False), "home_field_adj"] = 1.0
    df.loc[~df["is_home_offense"] & ~df["is_neutral_site"].fillna(False), "home_field_adj"] = -1.0

    # Rest differential: offense's rest days minus defense's rest days.
    df["offense_rest"] = df["home_rest"].where(df["is_home_offense"], df["away_rest"])
    df["defense_rest"] = df["away_rest"].where(df["is_home_offense"], df["home_rest"])
    df["rest_diff"] = df["offense_rest"] - df["defense_rest"]

    return df


def team_ratings(df: pd.DataFrame, use_recency_weights: bool = True) -> pd.DataFrame:
    """
    Section 3.8 — aggregate to team offense/defense ratings.

    Offense rating: team's own opponent-adjusted VOA on offense.
    Defense rating: VOA the team allowed on defense (opponent-adjusted),
    sign-flipped per DVOA convention (negative = good defense).
    """
    if use_recency_weights and "recency_weight" not in df.columns:
        df = add_recency_weights(df)

    weights = df["recency_weight"] if use_recency_weights else pd.Series(1.0, index=df.index)

    df = df.copy()
    df["_w"] = weights
    df["_weighted_voa"] = df["voa_adj"] * df["_w"]

    off = (
        df.groupby("posteam")
        .apply(lambda g: g["_weighted_voa"].sum() / g["_w"].sum())
        .rename("offense_voa")
    )
    defn = (
        df.groupby("defteam")
        .apply(lambda g: g["_weighted_voa"].sum() / g["_w"].sum())
        .rename("defense_voa_allowed")
    )

    ratings = pd.concat([off, defn], axis=1)
    ratings["defense_voa"] = ratings["defense_voa_allowed"]  # allowed-VOA IS the defense's own rating (unflipped)
    ratings["total_rating"] = ratings["offense_voa"] - ratings["defense_voa"]
    ratings = ratings.sort_values("total_rating", ascending=False)
    return ratings[["offense_voa", "defense_voa", "total_rating"]]


def compute_recent_form_rating(df: pd.DataFrame, current_week: int, weeks_back: int) -> pd.DataFrame:
    """
    Recent-form rating -- uses ONLY a team's last `weeks_back` weeks,
    recomputing baselines/opponent-adjustment on just that window
    rather than the full season. A real, previously-mentioned dashboard
    gap: the "last 4/8 games" footnote existed since early in the
    project but was never built.

    df: the same post-garbage-time-filter dataframe used for the main
    season rating (already has situation buckets and play_value).

    Simplification, stated plainly rather than hidden: this uses the
    last `weeks_back` calendar WEEKS, not each team's last N games
    played -- a team coming off a bye would have one fewer real game
    in its window than a team that played every week. This matches
    how most fan-facing "last N games" trackers work in practice, but
    is not a precise per-team game count.

    Returns None if there isn't enough real data in the window (e.g.
    early season, weeks_back=8 with only 3 real weeks played) rather
    than silently returning misleading ratings from too few plays.
    """
    window_start = current_week - weeks_back
    windowed = df[(df["week"] >= window_start) & (df["week"] < current_week)].copy()

    # REAL BUG FOUND BY TESTING: the original guard used
    # min(weeks_back, 2), which caps the requirement at 2 real weeks
    # NO MATTER what weeks_back was actually requested -- a "last 8
    # weeks" request with only 2 real weeks of data available
    # incorrectly proceeded instead of returning None. Fixed to
    # require at least half the REQUESTED window to actually be
    # present, so a genuinely early-season "last 8 weeks" request
    # correctly declines rather than silently computing from a much
    # smaller, misleadingly-labeled sample.
    real_weeks_present = windowed["week"].nunique()
    if real_weeks_present < max(2, weeks_back // 2):
        return None

    baselines = compute_baselines(windowed)
    windowed = compute_raw_voa(windowed, baselines)
    windowed = opponent_adjust(windowed, iterations=3, regression=0.5)
    return team_ratings(windowed, use_recency_weights=False)
