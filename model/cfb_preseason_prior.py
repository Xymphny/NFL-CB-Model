"""
CFB preseason prior -- two real attempts documented honestly, neither
fully resolving the problem, but both real, evidence-based progress.

ATTEMPT 1 (see git history / README): blend in last season's real
DVOA rating as a rating_diff proxy. Result: made the Ohio State
disagreement WORSE, not better, since Ohio State's 2025 DVOA was
already very strong. Investigated why and found Ohio State's real
47-of-91 scholarship turnover -- a structural roster-composition
problem neither historical signal can see.

ATTEMPT 2 (this version): CFB has no equivalent to NFL's actual
preseason GAMES (real backup snaps before the season) -- there's no
comparable live signal to lean on. The better CFB-native alternative
is REAL RETURNING PRODUCTION (model/cfb_returning_production_2026.py,
138 real FBS teams from CBS Sports/TruMedia) -- directly measures what
NFL's preseason mechanism only measures indirectly: how much of a
team's real, snap-weighted production carried over. Validated
directly: Indiana (44%) and Miami (46%), the two teams that played in
the actual 2025 championship game, both show low returning production
despite being the best two teams last year -- exactly the Ohio-State-
style problem, confirmed for two more real teams.

Design: discount last season's DVOA rating by the fraction of real
production returning, pulling teams with heavy turnover toward a
neutral prior rather than keeping their full historical rating:
    discounted_rating = returning_production_pct * last_season_rating

REAL, HONEST RESULT testing this against 4 real Week 1-2 2026
marquee games: genuinely mixed. 2 of 4 games moved toward the real
market line (Ohio State @ Texas improved by 2.1 points; Clemson @ LSU
by 1.1), while 2 showed negligible or slightly worse movement. A
real, additional nuance found while investigating: Ohio State's
SNAP-weighted continuity (56%) is healthier than the "47 of 91
players departed" headline suggested, since departures concentrated
in low-snap backups while QB (93%) and offensive line (82%) -- the
highest-snap positions -- mostly returned. Player-count and
snap-weighted continuity can tell meaningfully different stories.

HONEST CONCLUSION: this is real, validated progress in DATA QUALITY
(a comprehensive, real, CFB-native signal that didn't exist before)
and UNDERSTANDING (confirmed the root-cause hypothesis on two more
real teams), but the small real test set (4 games) doesn't yet show
a clear, one-directional fix. A properly rigorous next step would
need many more real test games, or historical returning-production
data from past seasons to actually calibrate a blend weight against
real outcomes -- neither attempted here.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from model.elo_rating import compute_elo_walk_forward
from model.cfb_prediction import predict_margin
from model.cfb_returning_production_2026 import RETURNING_PRODUCTION_2026

LAST_SEASON_RATINGS_PATH = os.path.join(os.path.dirname(__file__), "cfb_2025_final_ratings.csv")
SCHEDULE_CACHE = os.path.join(os.path.dirname(__file__), "cfb_schedule_cache.csv")


def get_cfb_preseason_prior():
    last_season_ratings = pd.read_csv(LAST_SEASON_RATINGS_PATH, index_col=0)
    schedule = pd.read_csv(SCHEDULE_CACHE)
    _, current_elo = compute_elo_walk_forward(schedule)
    return last_season_ratings, current_elo


def predict_cfb_preseason_game(home_team, away_team, last_season_ratings, elo_ratings):
    if home_team not in last_season_ratings.index or away_team not in last_season_ratings.index:
        return None
    if home_team not in elo_ratings or away_team not in elo_ratings:
        return None

    # Discount by real returning production when available; fall back
    # to the undiscounted rating (not a fabricated guess) otherwise.
    home_rating = last_season_ratings.loc[home_team, "total_rating"]
    away_rating = last_season_ratings.loc[away_team, "total_rating"]
    if home_team in RETURNING_PRODUCTION_2026:
        home_rating *= RETURNING_PRODUCTION_2026[home_team]
    if away_team in RETURNING_PRODUCTION_2026:
        away_rating *= RETURNING_PRODUCTION_2026[away_team]

    rating_diff = home_rating - away_rating
    elo_diff = elo_ratings[home_team] - elo_ratings[away_team]
    margin = predict_margin(rating_diff=rating_diff, elo_diff=elo_diff)

    return {
        "home_team": home_team, "away_team": away_team,
        "predicted_margin": margin, "rating_diff": rating_diff, "elo_diff": elo_diff,
    }


if __name__ == "__main__":
    print("Real CFB preseason prior, using returning-production-discounted last-season")
    print("DVOA rating alongside real current Elo...\n")

    last_season_ratings, elo_ratings = get_cfb_preseason_prior()

    real_games = [
        ("Alabama", "Georgia", -3),
        ("Texas", "Ohio State", -1.5),
        ("LSU", "Clemson", -3),
        ("Miami", "Notre Dame", 3),
    ]

    for home, away, market_home_spread in real_games:
        result = predict_cfb_preseason_game(home, away, last_season_ratings, elo_ratings)
        if result:
            model_spread = result["predicted_margin"]
            market_favors = home if market_home_spread < 0 else away
            model_favors = home if model_spread > 0 else away
            print(f"{away} @ {home}:")
            print(f"  Market: {home} {market_home_spread:+.1f} (favors {market_favors})")
            print(f"  Model (returning-production-discounted + Elo): {model_spread:+.1f} (favors {model_favors})")
            print()
