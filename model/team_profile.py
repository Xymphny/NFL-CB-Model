"""
Team profile stats — the metrics for the per-team dashboard page:
EPA per play, success rate, DVOA (already computed in ratings.py),
red zone efficiency / points per trip, and turnover margin.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from model.play_value import get_success_thresholds

DRIVE_RESULT_POINTS = {
    "Touchdown": 7,
    "Field goal": 3,
}


def compute_epa_per_play(df: pd.DataFrame) -> pd.DataFrame:
    """Mean EPA per play, offense and defense (allowed), per team."""
    if "epa" not in df.columns:
        raise ValueError("epa column not found — check ingest/nfl_pbp.py's NEEDED_COLUMNS")

    off = df.groupby("posteam")["epa"].mean().rename("epa_per_play_offense")
    defn = df.groupby("defteam")["epa"].mean().rename("epa_per_play_allowed")
    return pd.concat([off, defn], axis=1)


def compute_success_rate(df: pd.DataFrame, league: str = "NFL") -> pd.DataFrame:
    """
    Binary success rate (Section 3.1's thresholds), as a standalone
    percentage — distinct from the continuous play_value the core
    rating uses, but using the exact same threshold definitions so the
    two stay consistent with each other.
    """
    thresholds = get_success_thresholds(league)

    def is_success(row):
        down, ydstogo, yards_gained = row.get("down"), row.get("ydstogo"), row.get("yards_gained")
        if pd.isna(down) or pd.isna(ydstogo) or ydstogo <= 0:
            return None
        threshold = thresholds.get(int(down), 1.00)
        return (yards_gained / ydstogo) >= threshold

    scrimmage = df[df["play_type"].isin(["pass", "run"])].copy()
    scrimmage["success"] = scrimmage.apply(is_success, axis=1)
    scrimmage = scrimmage.dropna(subset=["success"])

    off = scrimmage.groupby("posteam")["success"].mean().rename("success_rate_offense")
    defn = scrimmage.groupby("defteam")["success"].mean().rename("success_rate_allowed")
    return pd.concat([off, defn], axis=1)


def compute_red_zone_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Red zone trips and points per trip, offense and defense.

    A "trip" is a (game, drive, team) combination where the team had
    the ball at yardline_100 <= 20 at some point during that drive.
    Points per trip uses fixed_drive_result (Touchdown=7, Field goal=3,
    else 0) — the drive's actual outcome, not just the play in the
    red zone itself, since a red zone snap doesn't guarantee the drive
    ended there.
    """
    red_zone_plays = df[df["yardline_100"] <= 20].dropna(subset=["fixed_drive"])

    trips = red_zone_plays.drop_duplicates(subset=["game_id", "posteam", "fixed_drive"])
    trips = trips.copy()
    trips["points"] = trips["fixed_drive_result"].map(DRIVE_RESULT_POINTS).fillna(0)

    off = trips.groupby("posteam").agg(
        red_zone_trips=("fixed_drive", "count"),
        red_zone_points_per_trip=("points", "mean"),
        red_zone_td_pct=("points", lambda p: (p == 7).mean()),
    )
    defn = trips.groupby("defteam" if "defteam" in trips.columns else "posteam").agg(
        red_zone_trips_allowed=("fixed_drive", "count"),
        red_zone_points_per_trip_allowed=("points", "mean"),
    )
    return off.join(defn, how="outer") if "defteam" in trips.columns else off


def compute_turnover_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Giveaways (this team's own turnovers on offense) vs. takeaways
    (turnovers forced while on defense), and the margin between them.
    """
    giveaways = (
        df.groupby("posteam")
        .apply(lambda g: (g["interception"].sum() + g["fumble_lost"].sum()))
        .rename("giveaways")
    )
    takeaways = (
        df.groupby("defteam")
        .apply(lambda g: (g["interception"].sum() + g["fumble_lost"].sum()))
        .rename("takeaways")
    )
    result = pd.concat([giveaways, takeaways], axis=1)
    result["turnover_margin"] = result["takeaways"] - result["giveaways"]
    return result


def build_team_profile(df: pd.DataFrame, dvoa_ratings: pd.DataFrame, league: str = "NFL") -> pd.DataFrame:
    """
    Combines all five dashboard metrics into one table, indexed by team.
    dvoa_ratings is the output of model.ratings.team_ratings() — DVOA
    is already computed there, not re-derived here.
    """
    epa = compute_epa_per_play(df)
    success = compute_success_rate(df, league=league)
    red_zone = compute_red_zone_efficiency(df)
    turnovers = compute_turnover_margin(df)

    profile = dvoa_ratings.join([epa, success, red_zone, turnovers], how="outer")
    return profile


if __name__ == "__main__":
    from ingest.nfl_pbp import load_season
    from model.ratings import (
        add_situation_buckets, score_all_plays, compute_baselines,
        compute_raw_voa, opponent_adjust, team_ratings,
    )

    print("Loading and rating real 2023 data...")
    df = load_season(2023)
    df_scored = add_situation_buckets(df.copy())
    df_scored = score_all_plays(df_scored)
    baselines = compute_baselines(df_scored)
    df_scored = compute_raw_voa(df_scored, baselines)
    df_scored = opponent_adjust(df_scored)
    dvoa = team_ratings(df_scored, use_recency_weights=False)

    print("Building full team profile...")
    profile = build_team_profile(df, dvoa)

    print(f"\n{len(profile)} teams profiled. Sample (sorted by DVOA):")
    cols = ["total_rating", "epa_per_play_offense", "success_rate_offense",
            "red_zone_points_per_trip", "turnover_margin"]
    print(profile.sort_values("total_rating", ascending=False)[cols].head(10).to_string(float_format=lambda x: f"{x:.3f}"))
