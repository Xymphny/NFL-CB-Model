"""
Genuinely held-out validation of Layer 2 — fit coefficients on
2021-2022 ONLY, evaluate on 2023 ONLY (data the coefficients never
saw). Built after realizing the original walk-forward tests fit one
set of coefficients across the full 584-game sample and evaluated on
that same sample — the underlying ratings/NGS features have zero
lookahead, but the coefficients themselves got to "see" every outcome
during fitting, which overstates real accuracy.

Real finding from running this: the previously-reported 58.22% ->
64.90% improvement was significantly overstated. The genuinely
held-out numbers are:
  - Rating-only baseline:      59.49% straight-up
  - Base Layer 2 (4 features): 60.51% straight-up (real, ~1pt gain)
  - Extended (+3 features):    60.51% straight-up (IDENTICAL -- the
    extended features added zero genuine value, only appeared to help
    due to overfitting on the in-sample test)

This is why model/prediction.py uses the base 4-feature model, not
the extended one.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.walk_forward_layer2_extended_test import run_walk_forward_test


def held_out_test(train, test, feature_cols, label):
    X_train = np.column_stack([train[feature_cols].values, np.ones(len(train))])
    y_train = train["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack([test[feature_cols].values, np.ones(len(test))])
    y_test = test["actual_margin"].values
    pred_test = X_test @ coef
    straight_up = ((pred_test > 0) == test["actual_home_win"]).mean()
    mae = np.mean(np.abs(y_test - pred_test))
    print(f"{label}: straight-up = {straight_up:.4f}, MAE = {mae:.2f}")
    return straight_up, mae


if __name__ == "__main__":
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    print(f"\nTrain: {len(train)} games (2021-2022), Test: {len(test)} games (2023, fully held out)\n")

    held_out_test(train, test, ["rating_diff", "home_field", "rest_diff"], "Baseline (rating only)      ")
    held_out_test(train, test, ["rating_diff", "home_field", "rest_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"], "Base Layer 2 (4 features)   ")
    held_out_test(train, test, ["rating_diff", "home_field", "rest_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff", "cushion_diff", "catch_pct_diff", "stacked_box_diff"], "Extended Layer 2 (7 features)")
