"""
Calibrates recency-weighting half-life specifically for CFB -- until
now, CFB was silently reusing NFL's calibrated value (half-life=100,
found via NFL's own real walk-forward test). Never validated for CFB's
very different competitive structure (133 teams vs 32, far wider
talent spread, real home-field finding already differed from NFL's).
Given how large and real the NFL half-life finding was (a ~2.9 point
accuracy swing from an untested assumption), blindly reusing it for a
structurally different competition is a real gap worth checking.

Same held-out discipline: candidates selected using ONLY training-set
(2021-2022) MAE, evaluated once on the held-out 2023 test set.

Memory-conscious given the earlier OOM: caches only the checkpoint-week
dataframes needed (not the full raw 362-column season data) and frees
each season's data immediately after use.
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
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_recency_weights, team_ratings,
)

BACKTEST_SEASONS = [2021, 2022, 2023]
CHECKPOINT_WEEKS = [6, 10, 14]
CANDIDATE_HALF_LIVES = [2, 4, 6, 10, 100]


def build_cache():
    cache = {}
    for season in BACKTEST_SEASONS:
        print(f"Season {season}: downloading and parsing...")
        full_season_df, raw = load_cfb_season(season)
        schedule = derive_cfb_schedule(raw)
        del raw
        gc.collect()

        for week in CHECKPOINT_WEEKS:
            df = full_season_df[full_season_df["week"] < week].copy()
            df = add_situation_buckets(df)
            df = score_all_plays(df, use_turnover_luck_adjustment=True, league="CFB")
            df = filter_garbage_time(df)
            baselines = compute_baselines(df)
            df = compute_raw_voa(df, baselines)
            df = opponent_adjust(df, iterations=3, regression=0.5)
            cache[(season, week)] = (df, schedule[schedule["week"] == week].copy())
            print(f"  cached week {week}")

        del full_season_df, schedule
        gc.collect()

    return cache


def evaluate(half_life, cache):
    rows = []
    for (season, week), (df, week_games) in cache.items():
        weighted = add_recency_weights(df.copy(), half_life_weeks=half_life)
        ratings = team_ratings(weighted, use_recency_weights=True)

        for _, game in week_games.iterrows():
            home, away = game["home_team"], game["away_team"]
            if home not in ratings.index or away not in ratings.index:
                continue
            actual_margin = game["home_score"] - game["away_score"]
            rating_diff = ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"]
            rows.append({"season": season, "rating_diff": rating_diff,
                         "actual_margin": actual_margin, "actual_home_win": actual_margin > 0})

    result_df = pd.DataFrame(rows)
    train = result_df[result_df["season"].isin([2021, 2022])]
    test = result_df[result_df["season"] == 2023]

    coef = np.sum(train["rating_diff"] * train["actual_margin"]) / np.sum(train["rating_diff"] ** 2)
    train_mae = np.mean(np.abs(coef * train["rating_diff"] - train["actual_margin"]))
    test_pred = coef * test["rating_diff"]
    test_mae = np.mean(np.abs(test_pred - test["actual_margin"]))
    test_acc = ((test_pred > 0) == test["actual_home_win"]).mean()
    return train_mae, test_mae, test_acc


if __name__ == "__main__":
    cache = build_cache()

    results = {}
    for hl in CANDIDATE_HALF_LIVES:
        train_mae, test_mae, test_acc = evaluate(hl, cache)
        results[hl] = (train_mae, test_mae, test_acc)
        print(f"half_life={hl}: train MAE={train_mae:.3f} | test MAE={test_mae:.3f}, test acc={test_acc:.4f}")

    best_hl = min(results, key=lambda hl: results[hl][0])
    print(f"\nBest half-life selected using ONLY training MAE: {best_hl}")
    print(f"Its held-out test performance: MAE={results[best_hl][1]:.3f}, acc={results[best_hl][2]:.4f}")
    print(f"\nCurrently-deployed default (100, borrowed from NFL) held-out performance: "
          f"MAE={results[100][1]:.3f}, acc={results[100][2]:.4f}")
