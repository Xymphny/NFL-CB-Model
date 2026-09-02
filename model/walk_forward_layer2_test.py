"""
Rigorous walk-forward test of Layer 2's real predictive value --
fixes the real methodological weakness the earlier quick test
(test_layer2_value.py) explicitly flagged: using full-season averages
to explain games within that season has lookahead. This uses the
SAME strict walk-forward discipline as model/walk_forward_backtest.py
(ratings AND NGS features computed only through week W-1, never
including the week being predicted).

Tests a real, specific hypothesis: maybe NGS features aren't
redundant with the rating (the earlier test's finding), they're just
FASTER to react to a real change in team quality than the slower-
moving opponent-adjusted rating is. If that's true, this walk-forward
version should show a real improvement even though the full-season
version didn't.
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
    add_home_field_and_rest, team_ratings,
)
from model.layer2_ngs import compute_team_ngs_features, load_ngs_data

BACKTEST_SEASONS = [2021, 2022, 2023]
BACKTEST_WEEKS = range(4, 18)


def run_walk_forward_layer2_test():
    rows = []

    for season in BACKTEST_SEASONS:
        print(f"Season {season}...")
        schedules = load_schedules(seasons=[season])
        full_season_df = load_season(season)

        ngs_preloaded = {
            "passing": load_ngs_data(season, "passing"),
            "receiving": load_ngs_data(season, "receiving"),
            "rushing": load_ngs_data(season, "rushing"),
        }

        for week in BACKTEST_WEEKS:
            df = full_season_df[full_season_df["week"] < week].copy()
            df = add_situation_buckets(df)
            df = score_all_plays(df, use_turnover_luck_adjustment=True)
            df = filter_garbage_time(df)
            df = add_home_field_and_rest(df, schedules)
            baselines = compute_baselines(df)
            df = compute_raw_voa(df, baselines)
            df = opponent_adjust(df, iterations=3, regression=0.5)
            ratings = team_ratings(df, use_recency_weights=True)

            ngs = compute_team_ngs_features(season, through_week=week, preloaded_data=ngs_preloaded)

            week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue
                if home not in ngs.index or away not in ngs.index:
                    continue

                actual_margin = game["home_score"] - game["away_score"]

                rows.append({
                    "season": season, "week": week,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "cpoe_diff": ngs.loc[home, "team_cpoe"] - ngs.loc[away, "team_cpoe"],
                    "separation_diff": ngs.loc[home, "team_avg_separation"] - ngs.loc[away, "team_avg_separation"],
                    "yac_oe_diff": ngs.loc[home, "team_yac_over_expected"] - ngs.loc[away, "team_yac_over_expected"],
                    "ryoe_diff": ngs.loc[home, "team_ryoe"] - ngs.loc[away, "team_ryoe"],
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
                })

    return pd.DataFrame(rows)


def fit_and_score(df, feature_cols):
    X = df[feature_cols].values
    X = np.column_stack([X, np.ones(len(X))])
    y = df["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    mae = np.mean(np.abs(resid))
    straight_up = ((pred > 0) == df["actual_home_win"]).mean()
    return {"r2": r2, "mae": mae, "straight_up_accuracy": straight_up}


if __name__ == "__main__":
    df = run_walk_forward_layer2_test()
    print(f"\n{len(df)} real walk-forward games with rating + Layer 2 features, zero lookahead")

    baseline = fit_and_score(df, ["rating_diff"])
    print(f"\n=== Baseline (walk-forward): margin ~ rating_diff alone ===")
    print(f"  R^2 = {baseline['r2']:.4f}, MAE = {baseline['mae']:.2f}, "
          f"straight-up acc = {baseline['straight_up_accuracy']:.4f}")

    with_layer2 = fit_and_score(df, ["rating_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"])
    print(f"\n=== With Layer 2 features (walk-forward, same discipline) ===")
    print(f"  R^2 = {with_layer2['r2']:.4f}, MAE = {with_layer2['mae']:.2f}, "
          f"straight-up acc = {with_layer2['straight_up_accuracy']:.4f}")

    r2_improvement = (with_layer2['r2'] - baseline['r2']) / baseline['r2'] * 100
    print(f"\n=== R^2 change: {r2_improvement:+.1f}% ===")
    print(f"MAE change: {baseline['mae'] - with_layer2['mae']:+.3f} points")
    print(f"Straight-up accuracy change: {with_layer2['straight_up_accuracy'] - baseline['straight_up_accuracy']:+.4f}")
