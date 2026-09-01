"""
Points-prediction layer — Section 11.4. Translates team ratings into
predicted spread, total, and win probability, using the coefficients
calibrated in calibrate_points_model.py against 5 real seasons.

These coefficients are the actual measured output of that calibration
run (see model/calibrate_points_model.py's printed results) — not
re-derived here. Recalibrate periodically (e.g. once a season, once
more data accumulates) by rerunning that script and updating the
constants below.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# From calibrate_points_model.py's real run against 2019-2023 data.
# CAVEAT (same one printed by that script): prior-season-only, no
# within-season walk-forward update yet — treat as a baseline, not a
# fully validated production model.
MARGIN_COEFFICIENTS = {
    "rating_diff": 22.7091,
    "home_field": 2.8321,
    "rest_diff": 0.1310,
    "intercept": -1.1811,
}

TOTAL_COEFFICIENTS = {
    "combined_offense": 12.4186,
    "wind": -0.2801,
    "intercept": 47.1185,
}


def predict_margin(rating_diff: float, is_neutral_site: bool, rest_diff: float) -> float:
    home_field = 0.0 if is_neutral_site else 1.0
    return (
        MARGIN_COEFFICIENTS["rating_diff"] * rating_diff
        + MARGIN_COEFFICIENTS["home_field"] * home_field
        + MARGIN_COEFFICIENTS["rest_diff"] * rest_diff
        + MARGIN_COEFFICIENTS["intercept"]
    )


def predict_total(combined_offense: float, wind: float = 0.0) -> float:
    # Wind is unknown for games more than a few days out (confirmed:
    # nflverse's schedule data has NaN wind for future games) — default
    # to 0.0 (no wind penalty) rather than failing, since weather is
    # genuinely unknowable that far ahead.
    wind = 0.0 if pd.isna(wind) else wind
    return (
        TOTAL_COEFFICIENTS["combined_offense"] * combined_offense
        + TOTAL_COEFFICIENTS["wind"] * wind
        + TOTAL_COEFFICIENTS["intercept"]
    )


def margin_to_win_probability(predicted_margin: float, margin_std: float = 13.0) -> float:
    """
    Converts a predicted margin into a win probability via a normal CDF,
    using the same residual standard deviation used in season_simulation.py
    (~13 points, consistent with calibrate_points_model.py's measured MAE
    of ~10-11 points).
    """
    from scipy.stats import norm
    return float(norm.cdf(predicted_margin / margin_std))


def predict_game(
    home_rating: float, away_rating: float,
    home_offense: float, away_offense: float,
    is_neutral_site: bool, rest_diff: float, wind: float = 0.0,
) -> dict:
    """
    Full prediction for one game: predicted points for each team, spread,
    total, and win probability — the single translation layer that
    serves spread, moneyline, and totals all at once (Section 11.4's
    design).
    """
    rating_diff = home_rating - away_rating
    margin = predict_margin(rating_diff, is_neutral_site, rest_diff)
    total = predict_total(home_offense + away_offense, wind)

    home_points = (total + margin) / 2
    away_points = (total - margin) / 2
    win_prob_home = margin_to_win_probability(margin)

    return {
        "predicted_home_points": home_points,
        "predicted_away_points": away_points,
        "spread": margin,       # positive = home favored, matches nflverse's spread_line convention direction
        "total": total,
        "win_prob_home": win_prob_home,
    }


def find_latest_ratings_snapshot(repo_data_path: str, season: int) -> str | None:
    """
    Finds the most recent ratings snapshot for a season, now that
    weekly_job.py writes immutable per-week files (data/ratings/{season}-
    week-{week}.json) instead of overwriting a single ratings.json —
    returns the path to the highest week number available, or None if
    no snapshot exists yet for this season.
    """
    import glob
    import re

    ratings_dir = os.path.join(repo_data_path, "ratings")
    pattern = os.path.join(ratings_dir, f"{season}-week-*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def week_number(path):
        match = re.search(r"week-(\d+)\.json$", path)
        return int(match.group(1)) if match else -1

    return max(candidates, key=week_number)


def load_current_ratings(ratings_json_path: str) -> pd.DataFrame:
    """
    Loads the ratings.json committed by weekly_job.py, so odds_watch_job.py
    can predict this week's games without recomputing Layer 1 from scratch.
    """
    import json
    with open(ratings_json_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["ratings"]).set_index("team")
    return df


def build_week_predictions(ratings: pd.DataFrame, upcoming_games: pd.DataFrame) -> dict:
    """
    Builds the model_predictions dict odds_watch_job.py's compute_divergences()
    expects, keyed by home_team.
    """
    predictions = {}
    for _, game in upcoming_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        if home not in ratings.index or away not in ratings.index:
            continue

        result = predict_game(
            home_rating=ratings.loc[home, "total_rating"],
            away_rating=ratings.loc[away, "total_rating"],
            home_offense=ratings.loc[home, "offense_voa"],
            away_offense=ratings.loc[away, "offense_voa"],
            is_neutral_site=game.get("is_neutral_site", False),
            rest_diff=game.get("home_rest", 7) - game.get("away_rest", 7),
            wind=game.get("wind", 0.0),
        )
        predictions[home] = result

    return predictions


if __name__ == "__main__":
    # Sanity check against the real Week 1 2026 slate, using placeholder
    # ratings (no real 2026 ratings exist yet — season hasn't started).
    from ingest.nfl_schedules import load_schedules

    sched = load_schedules(seasons=[2026])
    week1 = sched[sched["week"] == 1]

    fake_ratings = pd.DataFrame({
        "offense_voa": np.random.uniform(-0.1, 0.1, 32),
        "defense_voa": np.random.uniform(-0.1, 0.1, 32),
        "total_rating": np.random.uniform(-0.2, 0.2, 32),
    }, index=week1["home_team"].tolist() + [t for t in week1["away_team"] if t not in week1["home_team"].tolist()])

    predictions = build_week_predictions(fake_ratings, week1)
    print(f"Built predictions for {len(predictions)} of {len(week1)} Week 1 games (placeholder ratings)")
    for team, pred in list(predictions.items())[:3]:
        print(f"  {team}: spread={pred['spread']:+.1f}, total={pred['total']:.1f}, win_prob={pred['win_prob_home']:.2f}")
