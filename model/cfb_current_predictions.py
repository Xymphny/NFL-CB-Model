"""
Generates real CFB 2026 predictions using Elo carryover from
2021-2025 -- the only real signal available right now, since 2026 CFB
play-by-play isn't published yet (confirmed directly: 404 on the same
source used for every other CFB season this project). DVOA's
rating_diff is honestly set to 0 (not fabricated), since there's no
real current-season data to compute it from.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from model.elo_rating import compute_elo_walk_forward
from model.cfb_prediction import predict_margin

SCHEDULE_CACHE = os.path.join(os.path.dirname(__file__), "cfb_schedule_cache.csv")


def get_current_cfb_elo_ratings():
    schedule = pd.read_csv(SCHEDULE_CACHE)
    _, final_elo = compute_elo_walk_forward(schedule)
    return final_elo


def predict_cfb_game(home_team, away_team, elo_ratings):
    if home_team not in elo_ratings or away_team not in elo_ratings:
        return None
    elo_diff = elo_ratings[home_team] - elo_ratings[away_team]
    margin = predict_margin(rating_diff=0.0, elo_diff=elo_diff)
    return {"home_team": home_team, "away_team": away_team, "predicted_margin": margin, "elo_diff": elo_diff}


if __name__ == "__main__":
    print("Real CFB Elo ratings, carried forward through 2025 (2026 play-by-play not yet published)...")
    elo_ratings = get_current_cfb_elo_ratings()

    print("\nExample real predictions (rating_diff=0, Elo-only signal):")
    example_matchups = [
        ("Georgia", "Ohio State"),
        ("Alabama", "Texas"),
        ("Oregon", "Notre Dame"),
    ]
    for home, away in example_matchups:
        result = predict_cfb_game(home, away, elo_ratings)
        if result:
            print(f"  {away} @ {home}: predicted margin = {result['predicted_margin']:+.1f} (home perspective)")
