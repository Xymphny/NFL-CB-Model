"""
Calibrates Elo's own hyperparameters (K-factor, home-field advantage,
season regression) -- these were set to standard, reasonable values
inspired by FiveThirtyEight's published NFL Elo methodology, but never
actually tested against OUR data with the same held-out discipline
applied to everything else in this pipeline. Elo was validated as
USEFUL (the ensemble test), but its own internal parameters were never
calibrated -- a real gap.

Fast to test many candidates: Elo only needs real schedule/score data,
not full play-by-play processing, so a real held-out grid search is
cheap here unlike the DVOA-side calibrations.

Held-out discipline: candidates selected using ONLY training-set
(2014-2021) accuracy, evaluated once on the held-out 2022-2023 test
set -- same as every other calibration this session.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ingest.nfl_schedules import load_schedules
from model.elo_rating import compute_elo_walk_forward

TRAIN_SEASONS = list(range(2014, 2022))
TEST_SEASONS = [2022, 2023]

CANDIDATE_K = [10, 15, 20, 25, 32]
CANDIDATE_HOME_ADV = [35, 55, 65, 75, 100]
CANDIDATE_REGRESSION = [0.2, 0.33, 0.5, 0.66]


def evaluate(schedule, k, home_adv, regression):
    import model.elo_rating as elo_module
    original_regression = elo_module.SEASON_REGRESSION
    elo_module.SEASON_REGRESSION = regression
    try:
        elo_df, _ = compute_elo_walk_forward(schedule, k=k, home_advantage=home_adv)
    finally:
        elo_module.SEASON_REGRESSION = original_regression

    train = elo_df[elo_df["season"].isin(TRAIN_SEASONS)]
    test = elo_df[elo_df["season"].isin(TEST_SEASONS)]

    train_acc = ((train["elo_win_prob_home"] > 0.5) == train["actual_home_win"]).mean()
    test_acc = ((test["elo_win_prob_home"] > 0.5) == test["actual_home_win"]).mean()
    test_brier = np.mean((test["elo_win_prob_home"] - test["actual_home_win"].astype(float)) ** 2)
    return train_acc, test_acc, test_brier


if __name__ == "__main__":
    schedule = load_schedules(seasons=list(range(2014, 2024)))

    print("=== Stage 1: K-factor (home_adv=65, regression=0.33 held at current defaults) ===")
    k_results = {}
    for k in CANDIDATE_K:
        train_acc, test_acc, test_brier = evaluate(schedule, k, 65, 0.33)
        k_results[k] = (train_acc, test_acc, test_brier)
        print(f"  K={k}: train acc={train_acc:.4f} | test acc={test_acc:.4f}, test Brier={test_brier:.4f}")
    best_k = max(k_results, key=lambda k: k_results[k][0])
    print(f"\nBest K (by training accuracy): {best_k}")

    print(f"\n=== Stage 2: home advantage (K={best_k} fixed, regression=0.33) ===")
    ha_results = {}
    for ha in CANDIDATE_HOME_ADV:
        train_acc, test_acc, test_brier = evaluate(schedule, best_k, ha, 0.33)
        ha_results[ha] = (train_acc, test_acc, test_brier)
        print(f"  home_adv={ha}: train acc={train_acc:.4f} | test acc={test_acc:.4f}, test Brier={test_brier:.4f}")
    best_ha = max(ha_results, key=lambda ha: ha_results[ha][0])
    print(f"\nBest home advantage (by training accuracy): {best_ha}")

    print(f"\n=== Stage 3: season regression (K={best_k}, home_adv={best_ha} fixed) ===")
    reg_results = {}
    for reg in CANDIDATE_REGRESSION:
        train_acc, test_acc, test_brier = evaluate(schedule, best_k, best_ha, reg)
        reg_results[reg] = (train_acc, test_acc, test_brier)
        print(f"  regression={reg}: train acc={train_acc:.4f} | test acc={test_acc:.4f}, test Brier={test_brier:.4f}")
    best_reg = max(reg_results, key=lambda r: reg_results[r][0])
    print(f"\nBest regression (by training accuracy): {best_reg}")

    print(f"\n=== Final selected (K={best_k}, home_adv={best_ha}, regression={best_reg}) held-out performance ===")
    print(f"test acc={reg_results[best_reg][1]:.4f}, test Brier={reg_results[best_reg][2]:.4f}")
    print(f"\n=== Current defaults (K=20, home_adv=65, regression=0.33) held-out performance ===")
    _, default_test_acc, default_test_brier = evaluate(schedule, 20, 65, 0.33)
    print(f"test acc={default_test_acc:.4f}, test Brier={default_test_brier:.4f}")
