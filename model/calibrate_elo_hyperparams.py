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
from model.holdout_discipline import HoldoutVault
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

    # Each stage selects on TRAINING accuracy. The held-out column stays
    # sealed until that selection exists, so it cannot talk anyone out of
    # the argmin -- see model/holdout_discipline.py.
    def run_stage(name, candidates, build):
        vault = HoldoutVault(select_by="train_acc", minimize=False)
        print(f"\n=== {name} ===")
        for c in candidates:
            train_acc, test_acc, test_brier = evaluate(schedule, *build(c))
            vault.record(c, train_acc=train_acc, test_acc=test_acc,
                         test_brier=test_brier)
            print(f"  {c}: train acc={train_acc:.4f}")
        best = vault.select()
        print(f"  -> selected by training accuracy: {best}")
        return best, vault

    best_k, _ = run_stage("Stage 1: K-factor (home_adv=65, regression=0.33)",
                          CANDIDATE_K, lambda k: (k, 65, 0.33))
    best_ha, _ = run_stage(f"Stage 2: home advantage (K={best_k}, regression=0.33)",
                           CANDIDATE_HOME_ADV, lambda ha: (best_k, ha, 0.33))
    best_reg, reg_vault = run_stage(
        f"Stage 3: season regression (K={best_k}, home_adv={best_ha})",
        CANDIDATE_REGRESSION, lambda r: (best_k, best_ha, r))

    print(f"\n=== Final selected (K={best_k}, home_adv={best_ha}, "
          f"regression={best_reg}) held-out performance ===")
    print(f"  {reg_vault.held_out(best_reg)}")
    print("\n=== Current defaults (K=20, home_adv=65, regression=0.33) held-out ===")
    _, default_test_acc, default_test_brier = evaluate(schedule, 20, 65, 0.33)
    print(f"  test acc={default_test_acc:.4f}, test Brier={default_test_brier:.4f}")
    print("\nThis comparison is a grade, not a search. If the incumbent wins, that")
    print("is a fact to record -- not a reason to reopen the grid.")
