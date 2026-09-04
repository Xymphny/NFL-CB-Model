"""
Builds a combined real CFB schedule (final scores only) across
multiple seasons for Elo testing -- reuses the existing
ingest/cfb_pbp.py infrastructure (derive_cfb_schedule already produces
data in the exact (season, week, home_team, away_team, home_score,
away_score) shape model/elo_rating.py's compute_elo_walk_forward
already expects, so no CFB-specific Elo code is needed at all).

Memory-conscious given the earlier real OOM kill: processes one
season at a time, discards the large raw play-by-play dataframe
immediately after deriving the schedule from it.

Usage: python3 build_cfb_schedule_cache.py <season>
"""

import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingest.cfb_pbp import load_cfb_season, derive_cfb_schedule

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cfb_schedule_cache.csv")


if __name__ == "__main__":
    season = int(sys.argv[1])

    print(f"Downloading and deriving real schedule for CFB {season} (including postseason)...")
    _, raw = load_cfb_season(season)
    schedule = derive_cfb_schedule(raw, include_postseason=True)
    del raw
    gc.collect()

    print(f"{len(schedule)} real games derived")

    if os.path.exists(CACHE_PATH):
        existing = pd.read_csv(CACHE_PATH)
        existing = existing[existing["season"] != season]
        combined = pd.concat([existing, schedule], ignore_index=True)
    else:
        combined = schedule

    combined.to_csv(CACHE_PATH, index=False)
    print(f"Cache now has {len(combined)} total games across seasons: {sorted(combined['season'].unique())}")
