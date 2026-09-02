"""
Special teams sub-model -- Section 3.7, unbuilt since the very first
version of the spec. Scores field goals, punts, and kickoffs the same
"value over average" way the core model scores scrimmage plays, then
aggregates into a team-level special teams rating.

NOT opponent-adjusted (unlike offense/defense) -- a deliberate v1
simplification. Field goal/punt outcomes are much more kicker/
returner-dependent than opponent-dependent compared to scrimmage plays,
so skipping opponent adjustment here is a reasonable start-simple
choice, not an oversight.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

FG_MAKE_PROB_BY_DISTANCE = [
    (29, 0.98), (39, 0.94), (49, 0.84), (59, 0.65), (200, 0.40),
]


def expected_fg_points(distance):
    for max_dist, prob in FG_MAKE_PROB_BY_DISTANCE:
        if distance <= max_dist:
            return 3 * prob
    return 3 * 0.40


def score_field_goal(row):
    distance = row.get("kick_distance", 40)
    if pd.isna(distance):
        distance = 40
    expected = expected_fg_points(distance)

    if row.get("field_goal_result") == "made":
        return 3 - expected
    else:
        return 0 - expected


def score_punt(row, league_avg_net):
    distance = row.get("kick_distance", 0)
    return_yards = row.get("return_yards", 0)
    if pd.isna(distance):
        distance = 0
    if pd.isna(return_yards):
        return_yards = 0
    net = distance - return_yards
    return net - league_avg_net


def score_kickoff(row, league_avg_return):
    return_yards = row.get("return_yards", 0)
    if pd.isna(return_yards):
        return_yards = 0
    return -(return_yards - league_avg_return)


def compute_special_teams_ratings(df):
    fg = df[df["play_type"] == "field_goal"].copy()
    punts = df[df["play_type"] == "punt"].copy()
    kickoffs = df[df["play_type"] == "kickoff"].copy()

    fg["st_value"] = fg.apply(score_field_goal, axis=1)

    league_avg_net_punt = (punts["kick_distance"].fillna(0) - punts["return_yards"].fillna(0)).mean()
    punts["st_value"] = punts.apply(lambda r: score_punt(r, league_avg_net_punt), axis=1)

    league_avg_kickoff_return = kickoffs["return_yards"].fillna(0).mean()
    kickoffs["st_value"] = kickoffs.apply(lambda r: score_kickoff(r, league_avg_kickoff_return), axis=1)

    all_st_plays = pd.concat([
        fg[["posteam", "st_value"]],
        punts[["posteam", "st_value"]],
        kickoffs[["posteam", "st_value"]],
    ])

    st_totals = all_st_plays.groupby("posteam")["st_value"].sum().rename("special_teams_voa_raw")
    st_play_counts = all_st_plays.groupby("posteam").size().rename("special_teams_plays")

    result = pd.concat([st_totals, st_play_counts], axis=1)
    result["special_teams_voa"] = result["special_teams_voa_raw"] / result["special_teams_plays"]

    # SCALE MISMATCH, FOUND AND FIXED BEFORE WIRING IN: the raw per-play
    # value above is in points/yards units, spanning roughly -2.8 to
    # +2.2 in real 2023 data -- nowhere near offense_voa/defense_voa's
    # ratio-based scale (roughly -0.4 to +0.4). Combined directly into
    # total_rating, special teams would have completely dominated the
    # rating rather than contributing proportionally. SPECIAL_TEAMS_SCALE
    # is a documented, approximate rescaling (not derived from a formal
    # calibration) chosen so a strong/weak special teams unit moves
    # total_rating by roughly +/-0.10-0.15 -- a real, non-trivial, but
    # not dominant contribution, consistent with how special teams is
    # generally weighted in real sports analytics.
    SPECIAL_TEAMS_SCALE = 15.0
    result["special_teams_voa"] = result["special_teams_voa"] / SPECIAL_TEAMS_SCALE

    return result[["special_teams_voa", "special_teams_plays"]]


if __name__ == "__main__":
    from ingest.nfl_pbp import load_special_teams_plays

    print("Testing special teams scoring against real 2023 data...")
    df = load_special_teams_plays(2023)
    print(f"  {len(df)} special teams plays loaded")

    ratings = compute_special_teams_ratings(df)
    print(f"\n{len(ratings)} teams rated")
    print("\nBest special teams units:")
    print(ratings.sort_values("special_teams_voa", ascending=False).head(5))
    print("\nWorst special teams units:")
    print(ratings.sort_values("special_teams_voa", ascending=False).tail(5))
