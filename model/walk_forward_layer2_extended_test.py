"""
Tests whether the additional NGS fields (avg_cushion, catch_percentage,
percent_attempts_gte_eight_defenders) add further real predictive
value beyond the already-validated feature set (CPOE, separation,
YAC-over-expected, RYOE -- validated in walk_forward_layer2_test.py:
58.22% -> 64.04% straight-up accuracy). Same walk-forward discipline,
zero lookahead, same seasons.
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


def run_walk_forward_test():
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
                    "home_field": 0.0 if game.get("is_neutral_site", False) else 1.0,
                    "rest_diff": game["home_rest"] - game["away_rest"],
                    "cpoe_diff": ngs.loc[home, "team_cpoe"] - ngs.loc[away, "team_cpoe"],
                    "separation_diff": ngs.loc[home, "team_avg_separation"] - ngs.loc[away, "team_avg_separation"],
                    "yac_oe_diff": ngs.loc[home, "team_yac_over_expected"] - ngs.loc[away, "team_yac_over_expected"],
                    "ryoe_diff": ngs.loc[home, "team_ryoe"] - ngs.loc[away, "team_ryoe"],
                    "cushion_diff": ngs.loc[home, "team_avg_cushion"] - ngs.loc[away, "team_avg_cushion"],
                    "catch_pct_diff": ngs.loc[home, "team_catch_pct"] - ngs.loc[away, "team_catch_pct"],
                    "stacked_box_diff": ngs.loc[home, "team_stacked_box_pct"] - ngs.loc[away, "team_stacked_box_pct"],
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
    df = run_walk_forward_test()
    print(f"\n{len(df)} real walk-forward games, zero lookahead")

    validated_features = ["rating_diff", "home_field", "rest_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"]
    validated = fit_and_score(df, validated_features)
    print(f"\n=== Already-validated feature set (baseline for this test) ===")
    print(f"  R^2 = {validated['r2']:.4f}, MAE = {validated['mae']:.2f}, straight-up acc = {validated['straight_up_accuracy']:.4f}")

    extended_features = validated_features + ["cushion_diff", "catch_pct_diff", "stacked_box_diff"]
    extended = fit_and_score(df, extended_features)
    print(f"\n=== Extended: + cushion, catch%, stacked-box rate ===")
    print(f"  R^2 = {extended['r2']:.4f}, MAE = {extended['mae']:.2f}, straight-up acc = {extended['straight_up_accuracy']:.4f}")

    r2_change = (extended['r2'] - validated['r2']) / validated['r2'] * 100
    print(f"\n=== R^2 change: {r2_change:+.1f}% ===")
    print(f"MAE change: {validated['mae'] - extended['mae']:+.3f} points")
    print(f"Straight-up accuracy change: {extended['straight_up_accuracy'] - validated['straight_up_accuracy']:+.4f}")
