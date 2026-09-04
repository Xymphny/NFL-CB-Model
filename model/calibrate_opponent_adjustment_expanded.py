"""
Recalibrates opponent-adjustment iterations/regression on the
EXPANDED 2014-2023 dataset, sequential search (iterations first, then
regression) matching the original smaller-sample calibration's
approach.

Usage: python3 calibrate_opponent_adjustment_expanded.py <season> <stage> [fixed_iterations]
stage: "iterations" or "regression"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_home_field_and_rest, add_recency_weights, team_ratings,
)

BACKTEST_WEEKS = range(4, 18)
CANDIDATE_ITERATIONS = [1, 2, 3, 5]
CANDIDATE_REGRESSION = [0.3, 0.5, 0.7]
BEST_ITERATIONS_FROM_STAGE1 = 3


def process_season(season, stage, fixed_iterations=None):
    print(f"Processing season {season}, stage={stage}...")
    schedules = load_schedules(seasons=[season])
    full_season_df = load_season(season)

    rows = []
    for week in BACKTEST_WEEKS:
        base_df = full_season_df[full_season_df["week"] < week].copy()
        base_df = add_situation_buckets(base_df)
        base_df = score_all_plays(base_df, use_turnover_luck_adjustment=True)
        base_df = filter_garbage_time(base_df)
        base_df = add_home_field_and_rest(base_df, schedules)
        baselines = compute_baselines(base_df)
        base_df = compute_raw_voa(base_df, baselines)

        week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

        if stage == "iterations":
            candidates = [(it, 0.5) for it in CANDIDATE_ITERATIONS]
        else:
            candidates = [(fixed_iterations, reg) for reg in CANDIDATE_REGRESSION]

        for iterations, regression in candidates:
            adjusted = opponent_adjust(base_df.copy(), iterations=iterations, regression=regression)
            adjusted = add_recency_weights(adjusted, half_life_weeks=100.0)
            ratings = team_ratings(adjusted, use_recency_weights=True)

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

    return pd.DataFrame(rows)


if __name__ == "__main__":
    season = int(sys.argv[1])
    stage = sys.argv[2]
    fixed_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else BEST_ITERATIONS_FROM_STAGE1

    cache_path = os.path.join(os.path.dirname(__file__), f"opponent_adj_expanded_{stage}_cache.csv")

    result = process_season(season, stage, fixed_iterations)

    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        existing = existing[existing["season"] != season]
        combined = pd.concat([existing, result], ignore_index=True)
    else:
        combined = result

    combined.to_csv(cache_path, index=False)
    print(f"Cache ({stage}) now has {len(combined)} total rows across seasons: {sorted(combined['season'].unique())}")
