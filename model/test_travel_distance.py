"""
Tests whether real travel distance (identified as a gap via ESPN FPI's
own published methodology) adds genuine predictive value -- rather
than assume FPI's own ambiguous quotes (their NFL guide says "half a
point" for the Seattle-Miami extreme case; their college guide cites
a different "1 point per 1000 miles" rate) give the right coefficient
to just copy, calibrate it directly against real data with the same
held-out discipline used throughout this session.

Single additional feature -- much lower overfitting risk than the
multi-feature Layer 2 extensions that failed earlier.
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
from model.travel_distance import travel_distance

BACKTEST_SEASONS = [2021, 2022, 2023]
BACKTEST_WEEKS = range(4, 18)


def run_walk_forward_test():
    rows = []

    for season in BACKTEST_SEASONS:
        print(f"Season {season}...")
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
            df = add_recency_weights(df, half_life_weeks=100.0)
            ratings = team_ratings(df, use_recency_weights=True)

            week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue

                actual_margin = game["home_score"] - game["away_score"]
                dist = travel_distance(away, home)

                rows.append({
                    "season": season, "week": week,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "home_field": 0.0 if game.get("is_neutral_site", False) else 1.0,
                    "rest_diff": game["home_rest"] - game["away_rest"],
                    "travel_distance_thousands": dist / 1000.0,
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
                })

    return pd.DataFrame(rows)


def fit_and_score(train, test, feature_cols):
    X_train = np.column_stack([train[feature_cols].values, np.ones(len(train))])
    y_train = train["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack([test[feature_cols].values, np.ones(len(test))])
    y_test = test["actual_margin"].values
    pred_test = X_test @ coef
    mae = np.mean(np.abs(pred_test - y_test))
    acc = ((pred_test > 0) == test["actual_home_win"]).mean()
    return coef, mae, acc


if __name__ == "__main__":
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    print(f"\nTrain: {len(train)} games, Test: {len(test)} games (fully held out)\n")

    baseline_cols = ["rating_diff", "home_field", "rest_diff"]
    _, mae_b, acc_b = fit_and_score(train, test, baseline_cols)
    print(f"Baseline (no travel): MAE={mae_b:.2f}, straight-up={acc_b:.4f}")

    with_travel_cols = baseline_cols + ["travel_distance_thousands"]
    coef, mae_t, acc_t = fit_and_score(train, test, with_travel_cols)
    print(f"With real travel distance: MAE={mae_t:.2f}, straight-up={acc_t:.4f}")

    print(f"\nChange: MAE {mae_b - mae_t:+.3f}, straight-up accuracy {acc_t - acc_b:+.4f}")
    print(f"\nCalibrated travel coefficient: {coef[3]:.4f} points per 1000 miles of away-team travel")
