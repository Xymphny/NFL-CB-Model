"""
Calibrates opponent-adjustment iterations/regression for CFB -- never
checked before (CFB has been silently reusing NFL's calibrated values,
3 iterations/0.5 regression). Sequential search, memory-conscious
given the earlier real OOM kill.
"""

import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.cfb_pbp import load_cfb_season, derive_cfb_schedule
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
)

BACKTEST_SEASONS = [2021, 2022, 2023]
CHECKPOINT_WEEKS = [6, 10, 14]
CANDIDATE_ITERATIONS = [1, 2, 3, 5]
CANDIDATE_REGRESSION = [0.3, 0.5, 0.7]


def build_dataset(fixed_iterations=None, stage="iterations"):
    rows = []
    for season in BACKTEST_SEASONS:
        print(f"Season {season}: downloading and parsing...")
        full_season_df, raw = load_cfb_season(season)
        schedule = derive_cfb_schedule(raw)
        del raw
        gc.collect()

        for week in CHECKPOINT_WEEKS:
            base_df = full_season_df[full_season_df["week"] < week].copy()
            base_df = add_situation_buckets(base_df)
            base_df = score_all_plays(base_df, use_turnover_luck_adjustment=True, league="CFB")
            base_df = filter_garbage_time(base_df)
            baselines = compute_baselines(base_df)
            base_df = compute_raw_voa(base_df, baselines)

            week_games = schedule[schedule["week"] == week]

            if stage == "iterations":
                candidates = [(it, 0.5) for it in CANDIDATE_ITERATIONS]
            else:
                candidates = [(fixed_iterations, reg) for reg in CANDIDATE_REGRESSION]

            for iterations, regression in candidates:
                adjusted = opponent_adjust(base_df.copy(), iterations=iterations, regression=regression)
                ratings = team_ratings(adjusted, use_recency_weights=False)

                for _, game in week_games.iterrows():
                    home, away = game["home_team"], game["away_team"]
                    if home not in ratings.index or away not in ratings.index:
                        continue
                    actual_margin = game["home_score"] - game["away_score"]
                    rows.append({
                        "season": season, "week": week,
                        "iterations": iterations, "regression": regression,
                        "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                        "actual_margin": actual_margin, "actual_home_win": actual_margin > 0,
                    })

        del full_season_df, schedule
        gc.collect()

    return pd.DataFrame(rows)


def evaluate(df, key_col, key_val):
    subset = df[df[key_col] == key_val]
    train = subset[subset["season"].isin([2021, 2022])]
    test = subset[subset["season"] == 2023]

    coef = np.sum(train["rating_diff"] * train["actual_margin"]) / np.sum(train["rating_diff"] ** 2)
    train_mae = np.mean(np.abs(coef * train["rating_diff"] - train["actual_margin"]))
    test_pred = coef * test["rating_diff"]
    test_mae = np.mean(np.abs(test_pred - test["actual_margin"]))
    test_acc = ((test_pred > 0) == test["actual_home_win"]).mean()
    return train_mae, test_mae, test_acc


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "iterations"
    cache_path = os.path.join(os.path.dirname(__file__), f"cfb_opponent_adj_{stage}_cache.csv")

    if stage == "iterations":
        df = build_dataset(stage="iterations")
    else:
        fixed_iterations = int(sys.argv[2])
        df = build_dataset(fixed_iterations=fixed_iterations, stage="regression")

    df.to_csv(cache_path, index=False)
    print(f"\nWrote {cache_path} ({len(df)} rows)")

    key_col = "iterations" if stage == "iterations" else "regression"
    candidates = CANDIDATE_ITERATIONS if stage == "iterations" else CANDIDATE_REGRESSION
    results = {}
    for val in candidates:
        train_mae, test_mae, test_acc = evaluate(df, key_col, val)
        results[val] = (train_mae, test_mae, test_acc)
        print(f"  {key_col}={val}: train MAE={train_mae:.3f} | test MAE={test_mae:.3f}, test acc={test_acc:.4f}")
    best_val = min(results, key=lambda k: results[k][0])
    print(f"\nBest {key_col} (by training MAE): {best_val}")
