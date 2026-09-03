"""
Calibrates opponent-adjustment iterations and regression-toward-mean
(Section 3.5, set to 3 iterations / 50% regression in the original
spec and never validated against real data). Same held-out discipline
as the recency half-life calibration: candidates selected using ONLY
training MAE, evaluated once on the held-out 2023 test set.

Caches the pre-opponent-adjustment steps once per (season, week) since
those don't depend on the parameters being tested -- only
opponent_adjust itself and everything downstream does.
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


def build_cached_dataframes():
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
            cache[(season, week)] = (df, schedules)
        print(f"  cached season {season}")
    return cache


def run_walk_forward_test(iterations, regression, cache):
    rows = []

    for (season, week), (df, schedules) in cache.items():
        adjusted = opponent_adjust(df.copy(), iterations=iterations, regression=regression)
        adjusted = add_recency_weights(adjusted, half_life_weeks=100.0)
        ratings = team_ratings(adjusted, use_recency_weights=True)

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


def evaluate(iterations, regression, cache):
    df = run_walk_forward_test(iterations, regression, cache)
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    train_mae = np.mean(np.abs(train["pred_margin"] - train["actual_margin"]))
    test_mae = np.mean(np.abs(test["pred_margin"] - test["actual_margin"]))
    test_acc = ((test["pred_margin"] > 0) == test["actual_home_win"]).mean()
    return train_mae, test_mae, test_acc


if __name__ == "__main__":
    print("Building cached dataframes...")
    cache = build_cached_dataframes()

    print("\n=== Stage 1: iterations (regression fixed at 0.5) ===")
    iter_results = {}
    for it in [1, 2, 3, 5]:
        train_mae, test_mae, test_acc = evaluate(it, 0.5, cache)
        iter_results[it] = (train_mae, test_mae, test_acc)
        print(f"  iterations={it}: train MAE={train_mae:.3f} | test MAE={test_mae:.3f}, test acc={test_acc:.4f}")

    best_iterations = min(iter_results, key=lambda k: iter_results[k][0])
    print(f"\nBest iterations (by training MAE): {best_iterations}")

    print(f"\n=== Stage 2: regression (iterations fixed at {best_iterations}) ===")
    reg_results = {}
    for reg in [0.3, 0.5, 0.7]:
        train_mae, test_mae, test_acc = evaluate(best_iterations, reg, cache)
        reg_results[reg] = (train_mae, test_mae, test_acc)
        print(f"  regression={reg}: train MAE={train_mae:.3f} | test MAE={test_mae:.3f}, test acc={test_acc:.4f}")

    best_regression = min(reg_results, key=lambda k: reg_results[k][0])
    print(f"\nBest regression (by training MAE): {best_regression}")
    print(f"\nSelected (iterations={best_iterations}, regression={best_regression}) held-out performance: "
          f"MAE={reg_results[best_regression][1]:.3f}, acc={reg_results[best_regression][2]:.4f}")
    print(f"Current default (iterations=3, regression=0.5) held-out performance: "
          f"MAE={iter_results[3][1]:.3f}, acc={iter_results[3][2]:.4f}")
