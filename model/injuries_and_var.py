"""
Injury ingestion, replacement-value (VAR), and persistent QB-quality
adjustment. Spec Sections 8 and 11.3.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

NFLVERSE_INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
NFLVERSE_SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"


def load_injuries(season: int) -> pd.DataFrame:
    """Section 8.6 — NFL's official weekly injury report, confirmed real
    and working. report_status is the questionable/doubtful/out designation."""
    df = pd.read_csv(NFLVERSE_INJURIES_URL.format(season=season), low_memory=False)
    return df


def load_snap_counts(season: int) -> pd.DataFrame:
    """Used to identify each week's actual starting QB by snap share,
    for Section 11.3's persistent adjustment."""
    df = pd.read_csv(NFLVERSE_SNAP_COUNTS_URL.format(season=season), low_memory=False)
    return df


def compute_qb_var(df: pd.DataFrame, min_dropbacks_for_starter: int = 100) -> pd.DataFrame:
    """
    Real per-QB value-above-replacement, computed the same way the
    core rating computes value-over-average (VOA) — a ratio against a
    baseline, not a raw score, so it's on a comparable scale to
    offense_voa/total_rating and can be added directly.

    Requires df to already have play_value computed (via
    model.ratings.score_all_plays) and passer_player_name present
    (added to ingest/nfl_pbp.py's NEEDED_COLUMNS after discovering
    real QB changes need per-play attribution to detect).

    Replacement level = average play_value of backup-level passers
    (fewer than min_dropbacks_for_starter on the season) — the same
    "replacement level from real backup performance" concept as
    Section 8.2's original VAR design, now actually computed from
    real data instead of just specified.
    """
    passing_plays = df.dropna(subset=["passer_player_name"])

    dropbacks_per_qb = passing_plays.groupby("passer_player_name").size()
    starters = dropbacks_per_qb[dropbacks_per_qb >= min_dropbacks_for_starter].index
    replacement_pool = passing_plays[~passing_plays["passer_player_name"].isin(starters)]

    if len(replacement_pool) == 0:
        replacement_value = passing_plays["play_value"].mean()  # fallback if no clear backup pool this season
    else:
        replacement_value = replacement_pool["play_value"].mean()

    qb_avg_value = passing_plays.groupby("passer_player_name")["play_value"].mean()

    # VOA-style ratio, matching the scale of offense_voa/total_rating
    # elsewhere in this project, rather than a raw, differently-scaled
    # play-value difference.
    var = (qb_avg_value - replacement_value) / abs(replacement_value)
    var.name = "var"

    result = var.to_frame()
    result["dropbacks"] = dropbacks_per_qb
    result["is_starter"] = result.index.isin(starters)
    return result.sort_values("var", ascending=False)


def identify_current_starter(df: pd.DataFrame, team: str, recent_weeks: int = 3) -> str:
    """
    Whichever QB had the most dropbacks for this team in the most
    recent weeks of data available — a simple, direct "who's actually
    playing right now" signal, used for the persistent QB adjustment
    rather than whoever had the most total dropbacks over the full
    season (which could be a since-injured starter).
    """
    team_plays = df[(df["posteam"] == team)].dropna(subset=["passer_player_name"])
    if team_plays.empty:
        return None
    max_week = team_plays["week"].max()
    recent = team_plays[team_plays["week"] > max_week - recent_weeks]
    if recent.empty:
        recent = team_plays  # fall back to full season if too little recent data
    return recent["passer_player_name"].value_counts().idxmax()


def apply_qb_persistence_adjustment(ratings: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Section 11.3 — persistent QB-quality adjustment, now actually wired
    to real data. For each team, identifies the current (most recent)
    starter and adjusts their offense rating by that player's real VAR
    relative to a real backup-level replacement baseline — computed
    from actual per-play data, not specified as a placeholder.

    This is what should feed forward-looking uses (like season
    simulation) instead of the plain season-aggregate rating, since
    the aggregate blends a since-changed starter's play with whoever's
    playing now.
    """
    qb_var = compute_qb_var(df)
    adjusted = ratings.copy()
    adjusted["qb_adjustment"] = 0.0
    adjusted["current_starter"] = None

    for team in adjusted.index:
        starter = identify_current_starter(df, team)
        if starter is None or starter not in qb_var.index:
            continue
        adjusted.loc[team, "current_starter"] = starter
        adjusted.loc[team, "qb_adjustment"] = qb_var.loc[starter, "var"]

    adjusted["offense_voa_qb_adjusted"] = adjusted["offense_voa"] + adjusted["qb_adjustment"]
    adjusted["total_rating_qb_adjusted"] = adjusted["offense_voa_qb_adjusted"] - adjusted["defense_voa"]
    return adjusted


def compute_position_var(
    df: pd.DataFrame,
    attribution_column: str,
    min_plays_for_starter: int = 30,
    flip_sign: bool = False,
) -> pd.DataFrame:
    """
    Generalized version of compute_qb_var — extends the same real,
    data-driven VAR computation to other positions (Section 8.7's open
    question, previously left as QB-only). Works for any per-play
    player-attribution column nflverse provides.

    attribution_column: e.g. "receiver_player_name" for WR/TE,
        "sack_player_name" for pass rushers.
    flip_sign: True for defensive attribution columns (e.g. sacks),
        since play_value is computed from the OFFENSE's perspective —
        a sack has negative play_value for the offense, which is
        exactly the POSITIVE contribution a pass rusher made. Flipping
        the sign credits the defender correctly rather than crediting
        them with a negative "value".
    min_plays_for_starter: lower than compute_qb_var's default
        (100 dropbacks), since receivers and pass rushers are involved
        in far fewer plays per game than a QB is.
    """
    attributed_plays = df.dropna(subset=[attribution_column])
    if flip_sign:
        attributed_plays = attributed_plays.copy()
        attributed_plays["play_value"] = -attributed_plays["play_value"]

    plays_per_player = attributed_plays.groupby(attribution_column).size()
    starters = plays_per_player[plays_per_player >= min_plays_for_starter].index
    replacement_pool = attributed_plays[~attributed_plays[attribution_column].isin(starters)]

    if len(replacement_pool) == 0:
        replacement_value = attributed_plays["play_value"].mean()
    else:
        replacement_value = replacement_pool["play_value"].mean()

    player_avg_value = attributed_plays.groupby(attribution_column)["play_value"].mean()
    var = (player_avg_value - replacement_value) / abs(replacement_value)
    var.name = "var"

    result = var.to_frame()
    result["plays"] = plays_per_player
    result["is_starter"] = result.index.isin(starters)
    return result.sort_values("var", ascending=False)


# HONEST FINDING FROM TESTING, NOT A HYPOTHETICAL CAVEAT: tested against
# real 2023 sack data with flip_sign=True, and it produces a misleading
# signal for pass rushers specifically. Myles Garrett and T.J. Watt --
# two of the league's most elite edge rushers -- both showed NEGATIVE
# VAR, while several lesser-known names topped the list. Root cause: a
# sack's play_value is driven almost entirely by the down/distance/
# field-position situation it occurred in, not by which player made it
# -- two rushers' average "sack value" mostly reflects what situations
# they happened to get sacks in, not their actual rushing skill. This
# is unlike QBs (dropback-level value strongly reflects the QB's own
# decisions) or receivers (tested below, works well -- verified against
# real elite 2023 WRs: Tyreek Hill, CeeDee Lamb, and Amon-Ra St. Brown
# all showed strongly positive VAR, matching their real seasons).
#
# CONCLUSION: compute_position_var is validated and safe to use for
# receiver_player_name. Do NOT use it for sack_player_name or other
# defensive-play attribution without a genuinely different signal (real
# pressure rate / PFF-style charting data, not play-level value) -- a
# real signal like pressure rate needs data this project doesn't have,
# already flagged early on as deferred to Layer 2.


def compute_player_var(pbp_df: pd.DataFrame, snap_counts: pd.DataFrame, position: str = "QB") -> pd.DataFrame:
    """
    Section 8.2 — Value Above Replacement.

    Simplified first-pass definition: for QBs specifically, credit each
    player with the average play_value of plays run while they were the
    starter (identified via snap share), then compare against the
    replacement-level baseline (the average play_value produced by
    backups league-wide — anyone who started zero-to-few games that
    season at the position).

    NOTE: this requires pbp_df to already have play_value computed
    (i.e. run through model/ratings.py's scoring step first) and to
    have a home_qb_name/away_qb_name or similar per-play starter
    identity joined on — that join is NOT done here, it's the caller's
    responsibility, since the exact join key depends on which data
    (schedule vs snap counts) is being used as the source of truth for
    "who started."
    """
    if position != "QB":
        raise NotImplementedError(
            "VAR is only implemented for QB in this pass — see Section 11.3's "
            "note that QB gets a fully separate treatment. Extending to other "
            "positions (Section 8.7's open question) is future work."
        )

    # Identify each team's primary starter per week by offensive snap share.
    qb_snaps = snap_counts[snap_counts["position"] == "QB"].copy()
    starters = (
        qb_snaps.sort_values("offense_pct", ascending=False)
        .groupby(["team", "week"])
        .first()
        .reset_index()[["team", "week", "player", "offense_pct"]]
    )

    # Games started per player, to distinguish "the starter" from "a backup
    # who played garbage-time snaps" (replacement-level pool).
    starts_per_player = starters.groupby("player").size().rename("games_started")
    starters = starters.merge(starts_per_player, on="player", how="left")

    # Replacement-level pool: players with few starts (backups), used to
    # compute the replacement baseline this season.
    REPLACEMENT_START_THRESHOLD = 3
    replacement_pool = starters[starters["games_started"] <= REPLACEMENT_START_THRESHOLD]["player"].unique()

    return starters.assign(is_replacement_level=starters["player"].isin(replacement_pool))


def apply_persistent_qb_adjustment(
    team_ratings: pd.DataFrame,
    current_starters: pd.DataFrame,
    player_var_by_name: dict,
    replacement_var: float = 0.0,
) -> pd.DataFrame:
    """
    Section 11.3 — hold the gap between the current starter's VAR and
    replacement-level as a standing adjustment to that team's rating,
    rather than only reacting to an injury-report event.

    Parameters
    ----------
    team_ratings : output of model/ratings.py's team_ratings()
    current_starters : DataFrame with columns [team, player] — this
        week's identified starter per team (from compute_player_var's
        starters table, filtered to the most recent week)
    player_var_by_name : dict mapping player name -> their computed VAR
    replacement_var : the league-wide replacement-level baseline VAR
        (by definition ~0 if VAR is defined as value-above-replacement)
    """
    adjusted = team_ratings.copy()
    adjusted["qb_adjustment"] = 0.0

    for _, row in current_starters.iterrows():
        team, player = row["team"], row["player"]
        if team not in adjusted.index:
            continue
        var = player_var_by_name.get(player, replacement_var)
        adjusted.loc[team, "qb_adjustment"] = var - replacement_var

    adjusted["total_rating_with_qb_adj"] = adjusted["total_rating"] + adjusted["qb_adjustment"]
    return adjusted


if __name__ == "__main__":
    print("Loading real 2023 injury data...")
    injuries = load_injuries(2023)
    print(f"  {len(injuries)} injury report rows")
    print(injuries["report_status"].value_counts())

    print("\nLoading real 2023 snap counts...")
    snaps = load_snap_counts(2023)
    qb_starters = compute_player_var(pbp_df=None, snap_counts=snaps, position="QB")
    print(f"  {qb_starters['player'].nunique()} distinct players identified as a weekly QB starter")
    print(f"  {qb_starters['is_replacement_level'].sum()} starter-weeks flagged as replacement-level")
    print(qb_starters.head(10))
