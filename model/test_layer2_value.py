"""
Tests whether Layer 2's real tracking-data features (team CPOE,
separation, YAC-over-expected, RYOE) actually add incremental
predictive signal beyond what the existing opponent-adjusted rating
already captures -- rather than assuming real tracking data must help
just because it's real.

CAVEAT, same one calibrate_points_model.py already flagged for its own
prior-season approach: this uses full-season team averages to explain
games WITHIN that same season, which has some lookahead. This is a
legitimate quick diagnostic for "does this feature have any real
signal at all" before investing in a proper walk-forward-disciplined
recalibration -- it is NOT the same rigor as the walk-forward
backtest, and shouldn't be read as a final accuracy number.
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
    compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
)
from model.layer2_ngs import compute_team_ngs_features

TEST_SEASONS = [2021, 2022, 2023]


def build_dataset(season):
    df = load_season(season)
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True)
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    ratings = team_ratings(df, use_recency_weights=False)

    ngs = compute_team_ngs_features(season)

    sched = load_schedules(seasons=[season])
    games = sched.dropna(subset=["home_score", "away_score"])

    rows = []
    for _, g in games.iterrows():
        home, away = g["home_team"], g["away_team"]
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in ngs.index or away not in ngs.index:
            continue

        rows.append({
            "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
            "cpoe_diff": ngs.loc[home, "team_cpoe"] - ngs.loc[away, "team_cpoe"],
            "separation_diff": ngs.loc[home, "team_avg_separation"] - ngs.loc[away, "team_avg_separation"],
            "yac_oe_diff": ngs.loc[home, "team_yac_over_expected"] - ngs.loc[away, "team_yac_over_expected"],
            "ryoe_diff": ngs.loc[home, "team_ryoe"] - ngs.loc[away, "team_ryoe"],
            "actual_margin": g["home_score"] - g["away_score"],
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
    return {"r2": r2, "mae": mae, "coefficients": dict(zip(feature_cols + ["intercept"], coef))}


if __name__ == "__main__":
    print("Gathering real data across seasons (this pulls multiple full seasons)...")
    all_data = []
    for season in TEST_SEASONS:
        print(f"  {season}...")
        all_data.append(build_dataset(season))
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n{len(df)} real games with both rating and Layer 2 NGS features")

    baseline = fit_and_score(df, ["rating_diff"])
    print(f"\n=== Baseline: margin ~ rating_diff alone ===")
    print(f"  R^2 = {baseline['r2']:.4f}, MAE = {baseline['mae']:.2f}")

    with_layer2 = fit_and_score(df, ["rating_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"])
    print(f"\n=== With Layer 2 features added ===")
    print(f"  R^2 = {with_layer2['r2']:.4f}, MAE = {with_layer2['mae']:.2f}")
    print(f"  Coefficients: {with_layer2['coefficients']}")

    improvement = (with_layer2['r2'] - baseline['r2']) / baseline['r2'] * 100
    print(f"\n=== R^2 improvement from adding Layer 2 features: {improvement:+.1f}% ===")
    print(f"MAE change: {baseline['mae'] - with_layer2['mae']:+.3f} points")
