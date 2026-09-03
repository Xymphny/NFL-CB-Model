"""
Tests whether ensembling the play-by-play DVOA-style rating with a
genuinely different model architecture (Elo, using only final scores)
improves accuracy -- using the expanded 2014-2023 dataset (1,945 real
games, more than 3x the largest sample used in any prior test this
project) to give the sample-size-limited hypotheses a fair shot this
time.

Train: 2014-2021 (8 seasons). Test: 2022-2023 (2 seasons, fully held
out) -- both a larger training set AND a larger, more robust test set
than any previous validation this session.

Blend weight and regression coefficients selected using ONLY training
data, applied once to test -- the same discipline established after
the market-blending test's near-miss with test-set peeking.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.nfl_schedules import load_schedules
from model.elo_rating import compute_elo_walk_forward

CACHE_PATH = os.path.join(os.path.dirname(__file__), "expanded_walk_forward_cache.csv")
TRAIN_SEASONS = list(range(2014, 2022))
TEST_SEASONS = [2022, 2023]


def build_combined_dataset():
    dvoa = pd.read_csv(CACHE_PATH)

    schedule = load_schedules(seasons=list(range(2014, 2024)))
    elo_df, _ = compute_elo_walk_forward(schedule)

    # Merge on season/week/home/away -- both are real, independently
    # computed walk-forward datasets over the same real games.
    combined = dvoa.merge(
        elo_df[["season", "week", "home_team", "away_team", "elo_diff", "elo_win_prob_home"]],
        on=["season", "week", "home_team", "away_team"], how="inner",
    )
    return combined


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
    print("Building combined DVOA + Elo dataset...")
    df = build_combined_dataset()
    print(f"{len(df)} real games with both DVOA rating_diff and Elo ratings\n")

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    print(f"Train: {len(train)} games (2014-2021), Test: {len(test)} games (2022-2023, fully held out)\n")

    # 1. DVOA alone (the existing production approach)
    _, mae_dvoa, acc_dvoa = fit_and_score(train, test, ["rating_diff", "home_field", "rest_diff"])
    print(f"DVOA alone:       MAE={mae_dvoa:.2f}, straight-up={acc_dvoa:.4f}")

    # 2. Elo alone
    _, mae_elo, acc_elo = fit_and_score(train, test, ["elo_diff"])
    print(f"Elo alone:        MAE={mae_elo:.2f}, straight-up={acc_elo:.4f}")

    # 3. Ensemble: both models combined in one regression, coefficients
    # fit on train only.
    coef, mae_ens, acc_ens = fit_and_score(train, test, ["rating_diff", "home_field", "rest_diff", "elo_diff"])
    print(f"Ensemble (DVOA+Elo): MAE={mae_ens:.2f}, straight-up={acc_ens:.4f}")
    print(f"  Coefficients: {dict(zip(['rating_diff','home_field','rest_diff','elo_diff','intercept'], coef))}")

    print(f"\n=== Does the ensemble beat DVOA alone? ===")
    print(f"MAE change: {mae_dvoa - mae_ens:+.3f}")
    print(f"Straight-up accuracy change: {acc_ens - acc_dvoa:+.4f}")
