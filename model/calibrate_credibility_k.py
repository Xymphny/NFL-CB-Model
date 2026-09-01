"""
Calibrates k (Section 11.1's "how many games is the prior worth")
against real historical data — not guessed.

Method: for each test season, compute the true final-season rating
(ground truth), then check how close EARLY-season ratings get to that
truth under two approaches: raw in-season-only vs. blended with the
prior season's final rating. Whichever approach (and k value) gets
closer to the truth, on average across teams and early-season
checkpoints, is the one worth using.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.nfl_pbp import load_season
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
)
from model.preseason_prior import blend_rating

TEST_SEASONS = [2022, 2023]  # each needs season-1 as its prior source
WEEK_CUTOFFS = [2, 4, 6]
CANDIDATE_K_VALUES = [1, 2, 4, 8, 16, 32]


def compute_season_rating(season: int, through_week: int = None) -> pd.DataFrame:
    """Full pipeline for one season, optionally truncated to a week cutoff."""
    df = load_season(season)
    if through_week is not None:
        df = df[df["week"] <= through_week].copy()

    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True)
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    return team_ratings(df, use_recency_weights=False)


def run_backtest():
    print("Computing full-season ratings (priors + ground truth)...")
    full_season_ratings = {}
    for season in [min(TEST_SEASONS) - 1] + TEST_SEASONS:
        print(f"  {season} full season...")
        full_season_ratings[season] = compute_season_rating(season)

    results = []  # rows: season, week, team, method, k, error

    for season in TEST_SEASONS:
        prior = full_season_ratings[season - 1]
        truth = full_season_ratings[season]

        for week in WEEK_CUTOFFS:
            print(f"  {season} through week {week}...")
            in_season = compute_season_rating(season, through_week=week)

            for team in truth.index:
                if team not in prior.index or team not in in_season.index:
                    continue

                true_rating = truth.loc[team, "total_rating"]
                raw_rating = in_season.loc[team, "total_rating"]
                prior_rating = prior.loc[team, "total_rating"]

                # Baseline: no blending at all.
                results.append({
                    "season": season, "week": week, "team": team,
                    "method": "raw", "k": None,
                    "error": abs(raw_rating - true_rating),
                })

                # Blended, across each candidate k.
                for k in CANDIDATE_K_VALUES:
                    blended = blend_rating(prior_rating, raw_rating, games_played=week, k=k)
                    results.append({
                        "season": season, "week": week, "team": team,
                        "method": "blended", "k": k,
                        "error": abs(blended - true_rating),
                    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_backtest()

    print("\n=== Mean absolute error vs. true final-season rating ===")
    raw_mae = df[df["method"] == "raw"]["error"].mean()
    print(f"Raw (no blending):{'':>8} MAE = {raw_mae:.4f}")

    print("\nBlended, by k:")
    for k in CANDIDATE_K_VALUES:
        blended_mae = df[(df["method"] == "blended") & (df["k"] == k)]["error"].mean()
        improvement = (raw_mae - blended_mae) / raw_mae * 100
        print(f"  k={k:>3}: MAE = {blended_mae:.4f}  ({improvement:+.1f}% vs. raw)")

    print("\n=== Broken down by week cutoff (where blending should matter most early) ===")
    for week in WEEK_CUTOFFS:
        week_df = df[df["week"] == week]
        raw_mae_wk = week_df[week_df["method"] == "raw"]["error"].mean()
        print(f"\nWeek {week}: raw MAE = {raw_mae_wk:.4f}")
        for k in CANDIDATE_K_VALUES:
            blended_mae_wk = week_df[(week_df["method"] == "blended") & (week_df["k"] == k)]["error"].mean()
            improvement = (raw_mae_wk - blended_mae_wk) / raw_mae_wk * 100
            print(f"  k={k:>3}: MAE = {blended_mae_wk:.4f}  ({improvement:+.1f}%)")
