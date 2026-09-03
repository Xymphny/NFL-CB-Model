"""
Real CFB points-prediction calibration -- the first real predictive
model for college football in this project, matching (at a smaller
scale, given each season's data is far larger than NFL's) the same
walk-forward, zero-lookahead discipline used throughout the NFL work.

Scoped to 3 checkpoint weeks per season (6, 10, 14) rather than every
week 4-17 like the NFL backtests -- CFB's per-season data is far
larger (91MB vs NFL's ~15MB) and involves 133 teams vs 32, so a full
week-by-week backtest would take proportionally much longer. This is
a smaller but still genuinely real, zero-lookahead test, not a
shortcut around the discipline itself.
"""

import sys
import os

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


def compute_ratings_through_week(full_season_df, through_week):
    df = full_season_df[full_season_df["week"] < through_week].copy()
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True, league="CFB")
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    return team_ratings(df, use_recency_weights=False)


def run_walk_forward_test():
    import gc
    rows = []

    for season in BACKTEST_SEASONS:
        print(f"Season {season}: downloading and parsing (this is the slow part)...")
        full_season_df, raw = load_cfb_season(season)
        schedule = derive_cfb_schedule(raw)
        print(f"  {len(full_season_df)} plays, {len(schedule)} games")

        # Free the raw 362-column dataframe immediately -- only needed
        # for schedule derivation, and holding 3 seasons of it
        # simultaneously in memory caused an OOM kill on the first
        # attempt at this test.
        del raw
        gc.collect()

        for week in CHECKPOINT_WEEKS:
            print(f"  computing ratings through week {week}...")
            ratings = compute_ratings_through_week(full_season_df, week)

            week_games = schedule[schedule["week"] == week]

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue

                actual_margin = game["home_score"] - game["away_score"]

                rows.append({
                    "season": season, "week": week,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
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
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    print(f"\nTrain: {len(train)} games (2021-2022), Test: {len(test)} games (2023, fully held out)\n")

    coef, mae, acc = fit_and_score(train, test, ["rating_diff"])
    print(f"CFB margin model (rating_diff only): MAE={mae:.2f}, straight-up accuracy={acc:.4f}")
    print(f"Calibrated coefficient: rating_diff={coef[0]:.4f}, intercept={coef[1]:.4f}")

    home_win_rate = train["actual_home_win"].mean()
    print(f"\nReal home-field win rate in CFB (2021-2022 training data): {home_win_rate:.4f}")
