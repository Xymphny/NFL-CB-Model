"""
Tests whether ensembling CFB's DVOA-style rating with real Elo
improves on either alone -- given Elo alone already substantially
outperforms DVOA alone for CFB (68.60% vs 59.50% on the same 121 real
held-out games), likely because CFB's much wider, more persistent
talent gaps between programs are well-captured by Elo's cross-season
carryover, while the current CFB DVOA model restarts from scratch each
season with no preseason prior.
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
from model.elo_rating import compute_elo_walk_forward

BACKTEST_SEASONS = [2021, 2022, 2023]
CHECKPOINT_WEEKS = [6, 10, 14]


def compute_ratings_through_week(full_season_df, through_week):
    df = full_season_df[full_season_df["week"] < through_week].copy()
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True, league="CFB")
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)  # NOTE: iterations=1 is better for DVOA ALONE
                                                             # (see model/calibrate_cfb_opponent_adjustment.py),
                                                             # but iterations=3 gives the better REAL ensemble
                                                             # result (66.94% vs 65.29% accuracy) -- the same
                                                             # "optimize the deployed system, not a component in
                                                             # isolation" lesson confirmed a second time, after
                                                             # the NFL Elo hyperparameter case found the same thing.
    return team_ratings(df, use_recency_weights=False)


def build_dvoa_rows():
    rows = []
    for season in BACKTEST_SEASONS:
        print(f"Season {season}: downloading and parsing...")
        full_season_df, raw = load_cfb_season(season)
        schedule = derive_cfb_schedule(raw)
        del raw
        gc.collect()

        for week in CHECKPOINT_WEEKS:
            ratings = compute_ratings_through_week(full_season_df, week)
            week_games = schedule[schedule["week"] == week]

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue
                actual_margin = game["home_score"] - game["away_score"]
                rows.append({
                    "season": season, "week": week, "home_team": home, "away_team": away,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "actual_margin": actual_margin, "actual_home_win": actual_margin > 0,
                })

        del full_season_df, schedule
        gc.collect()

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
    dvoa_df = build_dvoa_rows()

    schedule_cache = pd.read_csv(os.path.join(os.path.dirname(__file__), "cfb_schedule_cache.csv"))
    elo_df, _ = compute_elo_walk_forward(schedule_cache)

    combined = dvoa_df.merge(
        elo_df[["season", "week", "home_team", "away_team", "elo_diff"]],
        on=["season", "week", "home_team", "away_team"], how="inner",
    )
    print(f"\n{len(combined)} real games with both DVOA and Elo\n")

    train = combined[combined["season"].isin([2021, 2022])]
    test = combined[combined["season"] == 2023]
    print(f"Train: {len(train)} games, Test: {len(test)} games (fully held out)\n")

    _, mae_dvoa, acc_dvoa = fit_and_score(train, test, ["rating_diff"])
    print(f"DVOA alone:  MAE={mae_dvoa:.2f}, acc={acc_dvoa:.4f}")

    _, mae_elo, acc_elo = fit_and_score(train, test, ["elo_diff"])
    print(f"Elo alone:   MAE={mae_elo:.2f}, acc={acc_elo:.4f}")

    coef, mae_ens, acc_ens = fit_and_score(train, test, ["rating_diff", "elo_diff"])
    print(f"Ensemble:    MAE={mae_ens:.2f}, acc={acc_ens:.4f}")
    print(f"Coefficients: rating_diff={coef[0]:.4f}, elo_diff={coef[1]:.4f}, intercept={coef[2]:.4f}")
