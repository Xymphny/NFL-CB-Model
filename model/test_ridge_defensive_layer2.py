"""
Tests whether ridge regularization lets the defensive Layer 2 features
(which failed with plain least-squares -- model/test_defensive_layer2.py
found they made accuracy WORSE, likely from too many coefficients for
~400 training games) actually add value once the estimation-noise
penalty is controlled for.

Alpha (regularization strength) selected via 5-fold cross-validation
WITHIN the training set only (2021-2022) -- never touches the 2023
test set during selection. The selected alpha is then fit once on all
of training and evaluated exactly once on the held-out test set.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression

from model.test_defensive_layer2 import run_walk_forward_test

FEATURE_COLS = [
    "rating_diff", "home_field", "rest_diff",
    "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff",
    "cpoe_allowed_diff", "separation_allowed_diff", "yac_oe_allowed_diff", "ryoe_allowed_diff",
]


def evaluate_model(model, X_train, y_train, X_test, y_test, actual_home_win_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = np.mean(np.abs(pred - y_test))
    acc = ((pred > 0) == actual_home_win_test).mean()
    return mae, acc


if __name__ == "__main__":
    print("Gathering data (same as the earlier defensive Layer 2 test)...")
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])].dropna(subset=FEATURE_COLS)
    test = df[df["season"] == 2023].dropna(subset=FEATURE_COLS)
    print(f"\nTrain: {len(train)} games, Test: {len(test)} games (fully held out)\n")

    X_train, y_train = train[FEATURE_COLS].values, train["actual_margin"].values
    X_test, y_test = test[FEATURE_COLS].values, test["actual_margin"].values
    home_win_test = test["actual_home_win"].values

    ols = LinearRegression()
    mae_ols, acc_ols = evaluate_model(ols, X_train, y_train, X_test, y_test, home_win_test)
    print(f"Plain OLS, all 11 features (reproducing the earlier failure): MAE={mae_ols:.2f}, acc={acc_ols:.4f}")

    alphas = np.logspace(-1, 3, 20)
    ridge = RidgeCV(alphas=alphas, cv=5)
    mae_ridge, acc_ridge = evaluate_model(ridge, X_train, y_train, X_test, y_test, home_win_test)
    print(f"Ridge (alpha={ridge.alpha_:.2f}, selected via train-only CV): MAE={mae_ridge:.2f}, acc={acc_ridge:.4f}")

    offense_only_cols = ["rating_diff", "home_field", "rest_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"]
    ols_offense = LinearRegression()
    mae_off, acc_off = evaluate_model(
        ols_offense, train[offense_only_cols].values, y_train,
        test[offense_only_cols].values, y_test, home_win_test,
    )
    print(f"Reference -- offense-only base Layer 2 (already in production): MAE={mae_off:.2f}, acc={acc_off:.4f}")

    print(f"\n=== Does ridge rescue the defensive features? ===")
    print(f"Plain OLS w/ defense:  acc={acc_ols:.4f}")
    print(f"Ridge w/ defense:      acc={acc_ridge:.4f}")
    print(f"Offense-only baseline: acc={acc_off:.4f}")
