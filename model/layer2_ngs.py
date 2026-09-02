"""
Layer 2 -- real player-tracking data, finally built. Deferred since the
very first message of this whole project (the original spec's Section
4 called for a "PFF-style proxy" since true PFF grades aren't public),
but nflverse's Next Gen Stats release turns out to have real tracking-
chip data covering almost exactly what was envisioned: avg_separation
and avg_yac_above_expectation for receivers, completion_percentage_
above_expectation (CPOE) for QBs, rush_yards_over_expected_per_att for
rushers. This is real tracking data, not another play-by-play
derivative -- a genuine step up in data quality.

Coverage gap, same as originally flagged: no O-line pressure-allowed
or pass-rush pressure-generated data exists in this free source either
-- that still needs real charting data this project doesn't have.
QB/WR/TE/RB are covered; OL/DL/CB are not.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

NGS_URL = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{season}_{stat_type}.csv.gz"


def load_ngs_data(season, stat_type):
    url = NGS_URL.format(season=season, stat_type=stat_type)
    df = pd.read_csv(url, compression="gzip", low_memory=False)
    return df[df["season_type"] == "REG"].copy()


def compute_team_ngs_features(season, through_week=None, preloaded_data=None, min_teams=28):
    """
    min_teams: sanity check on the result before trusting it -- a real
    bug was found during testing where a season's NGS release can be
    severely incomplete (2024's file, for example, has only 4 rows
    total covering one game) despite loading without error. Raises if
    fewer than min_teams show up, so a caller's fallback logic doesn't
    silently use badly incomplete data as if it were a full season.
    """
    if preloaded_data is not None:
        passing, receiving, rushing = preloaded_data["passing"], preloaded_data["receiving"], preloaded_data["rushing"]
    else:
        passing = load_ngs_data(season, "passing")
        receiving = load_ngs_data(season, "receiving")
        rushing = load_ngs_data(season, "rushing")

    if through_week is not None:
        passing = passing[passing["week"] < through_week]
        receiving = receiving[receiving["week"] < through_week]
        rushing = rushing[rushing["week"] < through_week]

    def weighted_avg(df, value_col, weight_col, group_col="team_abbr"):
        df = df.dropna(subset=[value_col, weight_col])
        df = df[df[weight_col] > 0]
        return df.groupby(group_col).apply(
            lambda g: np.average(g[value_col], weights=g[weight_col])
        )

    team_cpoe = weighted_avg(passing, "completion_percentage_above_expectation", "attempts")
    team_separation = weighted_avg(receiving, "avg_separation", "targets")
    team_yac_oe = weighted_avg(receiving, "avg_yac_above_expectation", "receptions")
    team_ryoe = weighted_avg(rushing, "rush_yards_over_expected_per_att", "rush_attempts")

    # Additional real NGS fields, not used in the originally-validated
    # feature set -- tested separately in model/walk_forward_layer2_extended_test.py
    # to check for further incremental value before trusting them.
    team_cushion = weighted_avg(receiving, "avg_cushion", "targets")
    team_catch_pct = weighted_avg(receiving, "catch_percentage", "targets")
    team_stacked_box_pct = weighted_avg(rushing, "percent_attempts_gte_eight_defenders", "rush_attempts")

    result = pd.DataFrame({
        "team_cpoe": team_cpoe,
        "team_avg_separation": team_separation,
        "team_yac_over_expected": team_yac_oe,
        "team_ryoe": team_ryoe,
        "team_avg_cushion": team_cushion,
        "team_catch_pct": team_catch_pct,
        "team_stacked_box_pct": team_stacked_box_pct,
    })

    if len(result) < min_teams:
        raise ValueError(
            f"NGS data for {season} (through_week={through_week}) only covers "
            f"{len(result)} teams, expected at least {min_teams} -- this season's "
            f"data release is likely incomplete, not a real signal to trust"
        )

    return result


def compute_player_grades(season, through_week=None, min_sample=5):
    passing = load_ngs_data(season, "passing")
    receiving = load_ngs_data(season, "receiving")
    rushing = load_ngs_data(season, "rushing")

    if through_week is not None:
        passing = passing[passing["week"] < through_week]
        receiving = receiving[receiving["week"] < through_week]
        rushing = rushing[rushing["week"] < through_week]

    def zscore_grade(df, value_col, weight_col, min_sample):
        agg = df.groupby("player_display_name").agg(
            value=(value_col, "mean"),
            sample=(weight_col, "sum"),
        )
        agg = agg[agg["sample"] >= min_sample]
        agg["grade"] = 50 + 10 * (agg["value"] - agg["value"].mean()) / agg["value"].std()
        return agg.sort_values("grade", ascending=False)

    # BUG FOUND AND FIXED BY TESTING: the original min_sample thresholds
    # (20 attempts for QB, 5 receptions for WR/TE) let tiny, noisy
    # samples dominate the leaderboard -- the first real test run
    # showed Mason Rudolph (71 attempts) and Hunter Renfrow (8
    # receptions) at the top instead of real 2023 stars, since backup/
    # garbage-time snaps against prevent defenses often post inflated
    # efficiency stats. Real season-long thresholds (~half a season of
    # starts for QB, a meaningful target share for WR/TE) fix this --
    # verified below against known real 2023 names.
    return {
        "QB": zscore_grade(passing, "completion_percentage_above_expectation", "attempts", 200),
        "WR_TE": zscore_grade(receiving, "avg_yac_above_expectation", "receptions", 40),
        "RB": zscore_grade(rushing, "rush_yards_over_expected_per_att", "rush_attempts", 80),
    }


if __name__ == "__main__":
    print("Testing Layer 2 team features against real 2023 data...")
    team_features = compute_team_ngs_features(2023)
    print(f"\n{len(team_features)} teams")
    print("\nBest CPOE (QB accuracy above expectation):")
    print(team_features.sort_values("team_cpoe", ascending=False)[["team_cpoe"]].head(5))
    print("\nBest average separation (receivers getting open):")
    print(team_features.sort_values("team_avg_separation", ascending=False)[["team_avg_separation"]].head(5))
    print("\nBest RYOE (rushing yards over expected per attempt):")
    print(team_features.sort_values("team_ryoe", ascending=False)[["team_ryoe"]].head(5))

    print("\n\nTesting player grades -- checking real known 2023 names...")
    grades = compute_player_grades(2023)
    print("\nTop 5 QBs by CPOE-based grade:")
    print(grades["QB"].head(5))
    print("\nTop 5 WR/TE by YAC-over-expected grade:")
    print(grades["WR_TE"].head(5))
