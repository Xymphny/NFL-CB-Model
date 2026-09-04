"""
Full weekly CFB walk-forward backtest -- expands from the earlier
3-checkpoint approximation (weeks 6/10/14) to every real week 4-13,
giving a much more precise accuracy baseline. Uses the validated DVOA
+ Elo ensemble settings (iterations=3, regression=0.5 -- confirmed
best for the real deployed ensemble).

Processes one season at a time, caching results, appending across
multiple runs.

Usage: python3 cfb_full_walk_forward.py <season>
"""

import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingest.cfb_pbp import load_cfb_season, derive_cfb_schedule
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cfb_full_walk_forward_cache.csv")
BACKTEST_WEEKS = range(4, 14)


def process_season(season):
    print(f"Season {season}: downloading and parsing...")
    full_season_df, raw = load_cfb_season(season)
    schedule = derive_cfb_schedule(raw)
    del raw
    gc.collect()

    rows = []
    for week in BACKTEST_WEEKS:
        print(f"  computing ratings through week {week}...")
        df = full_season_df[full_season_df["week"] < week].copy()
        df = add_situation_buckets(df)
        df = score_all_plays(df, use_turnover_luck_adjustment=True, league="CFB")
        df = filter_garbage_time(df)
        baselines = compute_baselines(df)
        df = compute_raw_voa(df, baselines)
        df = opponent_adjust(df, iterations=3, regression=0.5)
        ratings = team_ratings(df, use_recency_weights=False)

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


if __name__ == "__main__":
    season = int(sys.argv[1])

    result = process_season(season)

    if os.path.exists(CACHE_PATH):
        existing = pd.read_csv(CACHE_PATH)
        existing = existing[existing["season"] != season]
        combined = pd.concat([existing, result], ignore_index=True)
    else:
        combined = result

    combined.to_csv(CACHE_PATH, index=False)
    print(f"Cache now has {len(combined)} total games across seasons: {sorted(combined['season'].unique())}")
