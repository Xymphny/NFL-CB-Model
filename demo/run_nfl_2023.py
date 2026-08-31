"""
End-to-end demo: load real 2023 NFL play-by-play, run it through the
full Layer 1 pipeline, print resulting team ratings.

Run from the football_model/ directory:
    python3 demo/run_nfl_2023.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.nfl_pbp import load_season
from model.ratings import (
    add_situation_buckets,
    score_all_plays,
    compute_baselines,
    compute_raw_voa,
    opponent_adjust,
    team_ratings,
)


def main():
    print("Loading 2023 play-by-play from nflverse...")
    df = load_season(2023)
    print(f"  {len(df)} scrimmage plays loaded")

    print("Bucketing plays...")
    df = add_situation_buckets(df)

    print("Scoring plays (with turnover-luck adjustment)...")
    df = score_all_plays(df, use_turnover_luck_adjustment=True)

    print("Computing league baselines...")
    baselines = compute_baselines(df)
    print(f"  {len(baselines)} distinct situation buckets")

    print("Computing raw VOA...")
    df = compute_raw_voa(df, baselines)

    print("Running opponent adjustment (3 iterations)...")
    df = opponent_adjust(df, iterations=3, regression=0.5)

    print("Aggregating team ratings (recency-weighted)...")
    ratings = team_ratings(df, use_recency_weights=True)

    print("\n=== 2023 Team Ratings (Top 10, opponent-adjusted, recency-weighted) ===")
    print(ratings.head(10).to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\n=== Bottom 5 ===")
    print(ratings.tail(5).to_string(float_format=lambda x: f"{x:+.3f}"))

    return ratings


if __name__ == "__main__":
    main()
