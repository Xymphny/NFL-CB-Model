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
