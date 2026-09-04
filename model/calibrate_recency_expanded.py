"""
Recalibrates recency-weighting half-life on the EXPANDED 2014-2023
dataset (vs the original 3-season 2021-2023 calibration) -- given how
much the larger sample changed conclusions elsewhere this session
(DVOA-alone accuracy improved just from more data; the Elo ensemble
only showed its full value at this scale), it's a real, open question
whether the half-life conclusion (100 weeks, i.e. near-zero weighting)
still holds, or whether a different value wins with 3x the data.

Processes ONE season at a time (same real computational-cost lesson
as the CFB work and the original expanded-cache build), caching
results per candidate half-life to disk, appending across multiple
runs.

Usage: python3 calibrate_recency_expanded.py <season>
Run once per season (2014 through 2023).
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

CACHE_PATH = os.path.join(os.path.dirname(__file__), "recency_expanded_cache.csv")
BACKTEST_WEEKS = range(4, 18)
CANDIDATE_HALF_LIVES = [2, 4, 6, 10, 20, 50, 100]


def process_season(season):
    print(f"Processing season {season}...")
    schedules = load_schedules(seasons=[season])
    full_season_df = load_season(season)

    rows = []
    for week in BACKTEST_WEEKS:
        df = full_season_df[full_season_df["week"] < week].copy()
        df = add_situation_buckets(df)
        df = score_all_plays(df, use_turnover_luck_adjustment=True)
        df = filter_garbage_time(df)
        df = add_home_field_and_rest(df, schedules)
        baselines = compute_baselines(df)
        df = compute_raw_voa(df, baselines)
        df = opponent_adjust(df, iterations=3, regression=0.5)

        week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

        for hl in CANDIDATE_HALF_LIVES:
            weighted = add_recency_weights(df.copy(), half_life_weeks=hl)
            ratings = team_ratings(weighted, use_recency_weights=True)

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue

                actual_margin = game["home_score"] - game["away_score"]
                rows.append({
                    "season": season, "week": week, "half_life": hl,
                    "home_team": home, "away_team": away,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "home_field": 0.0 if game.get("is_neutral_site", False) else 1.0,
                    "rest_diff": game["home_rest"] - game["away_rest"],
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
                })

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
    print(f"Cache now has {len(combined)} total rows across seasons: {sorted(combined['season'].unique())}")
