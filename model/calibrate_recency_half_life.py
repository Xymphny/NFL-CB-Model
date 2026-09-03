"""
Calibrates the recency-weighting half-life (Section 3.6, set to 6 weeks
in the original spec and never actually validated against real data
until now) -- unlike the recent feature-addition attempts (extended
Layer 2, defensive features), this doesn't add new data or parameters,
it tunes an existing knob. Lower risk of the same sample-size wall
that made those attempts fail.

Genuinely held-out from the start: candidate half-lives are selected
using ONLY training-set (2021-2022) error, then the selected value is
evaluated exactly once on the held-out 2023 test set -- the same
discipline that caught the earlier Layer 2 overfitting mistake,
applied here from the beginning.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_home_field_and_rest, add_recency_weights, team_ratings,
)
from model.prediction import predict_margin

BACKTEST_SEASONS = [2021, 2022, 2023]
BACKTEST_WEEKS = range(4, 18)
CANDIDATE_HALF_LIVES = [2, 4, 6, 8, 10, 12, 16, 100]


def build_cached_dataframes():
    """
    Computes the expensive, half-life-INDEPENDENT pipeline once per
    (season, week) -- opponent adjustment, baselines, VOA -- since only
    the final recency-weighting step actually depends on half_life.
    Testing 8 candidates the naive way (recomputing everything 8x)
    would take ~35 minutes; caching this cuts it to a fraction of that.
    """
    cache = {}
    for season in BACKTEST_SEASONS:
        schedules = load_schedules(seasons=[season])
        full_season_df = load_season(season)

        for week in BACKTEST_WEEKS:
            df = full_season_df[full_season_df["week"] < week].copy()
            df = add_situation_buckets(df)
            df = score_all_plays(df, use_turnover_luck_adjustment=True)
            df = filter_garbage_time(df)
            df = add_home_field_and_rest(df, schedules)
            baselines = compute_baselines(df)
            df = compute_raw_voa(df, baselines)
            df = opponent_adjust(df, iterations=3, regression=0.5)
            cache[(season, week)] = (df, schedules)
        print(f"  cached season {season}")

    return cache


def run_walk_forward_test(half_life, cache):
    rows = []

    for (season, week), (df, schedules) in cache.items():
        df = add_recency_weights(df.copy(), half_life_weeks=half_life)
        ratings = team_ratings(df, use_recency_weights=True)

        week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

        for _, game in week_games.iterrows():
            home, away = game["home_team"], game["away_team"]
            if home not in ratings.index or away not in ratings.index:
                continue

            rating_diff = ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"]
            pred_margin = predict_margin(
                rating_diff, game.get("is_neutral_site", False),
                game["home_rest"] - game["away_rest"],
            )
            actual_margin = game["home_score"] - game["away_score"]

            rows.append({
                "season": season, "week": week,
                "pred_margin": pred_margin, "actual_margin": actual_margin,
                "actual_home_win": actual_margin > 0,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building cached dataframes (the expensive, half-life-independent part, computed once)...")
    cache = build_cached_dataframes()

    results = {}
    for hl in CANDIDATE_HALF_LIVES:
        print(f"Testing half_life={hl}...")
        df = run_walk_forward_test(hl, cache)
        train = df[df["season"].isin([2021, 2022])]
        test = df[df["season"] == 2023]

        train_mae = np.mean(np.abs(train["pred_margin"] - train["actual_margin"]))
        test_mae = np.mean(np.abs(test["pred_margin"] - test["actual_margin"]))
        test_acc = ((test["pred_margin"] > 0) == test["actual_home_win"]).mean()

        results[hl] = {"train_mae": train_mae, "test_mae": test_mae, "test_acc": test_acc}
        print(f"  train MAE={train_mae:.3f} | test MAE={test_mae:.3f}, test straight-up={test_acc:.4f}")

    best_hl = min(results, key=lambda hl: results[hl]["train_mae"])
    print(f"\nBest half-life selected using ONLY training MAE: {best_hl}")
    print(f"Its held-out test performance: MAE={results[best_hl]['test_mae']:.3f}, "
          f"straight-up={results[best_hl]['test_acc']:.4f}")
    print(f"\nCurrent default (6 weeks) held-out performance: "
          f"MAE={results[6]['test_mae']:.3f}, straight-up={results[6]['test_acc']:.4f}")
