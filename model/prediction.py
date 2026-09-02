"""
Points-prediction layer — Section 11.4. Translates team ratings into
predicted spread, total, and win probability.

MARGIN_COEFFICIENTS below is the twice-extended Layer 2 version,
walk-forward calibrated (2021-2023, zero lookahead) across two rounds
of testing:
1. Base Layer 2 (CPOE, separation, YAC-over-expected, RYOE): straight-up
   accuracy 58.22% -> 64.04% (model/walk_forward_layer2_test.py)
2. Extended (+ cushion, catch%, stacked-box rate): 64.04% -> 64.90%,
   a smaller but real further gain (model/walk_forward_layer2_extended_test.py)
Requires real NGS tracking features at prediction time — see
model/layer2_ngs.py. Callers that don't provide them get 0.0 defaults,
which means correct-but-unenhanced predictions, not an error.

One coefficient worth noting rather than hiding: cushion_diff's sign
is negative (more cushion for the home team's receivers correlates
with a WORSE margin), plausibly confounded by game state — teams
already losing often see more prevent-style cushion late in games.
The walk-forward test is a measure of out-of-sample prediction
accuracy, not a causal claim about any individual coefficient.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Walk-forward calibrated, 2021-2023, zero lookahead (model/walk_forward_layer2_extended_test.py).
MARGIN_COEFFICIENTS = {
    "rating_diff": 13.2165,
    "home_field": 2.9119,
    "rest_diff": 0.1677,
    "cpoe_diff": 0.6587,
    "separation_diff": 5.6077,
    "yac_oe_diff": 2.2567,
    "ryoe_diff": -0.0470,
    "cushion_diff": -3.1962,
    "catch_pct_diff": -0.1557,
    "stacked_box_diff": 0.0544,
    "intercept": -0.4361,
}

# Retained for reference/comparison.
MARGIN_COEFFICIENTS_BASE_LAYER2 = {
    "rating_diff": 12.9834,
    "home_field": 2.7266,
    "rest_diff": 0.1862,
    "cpoe_diff": 0.5132,
    "separation_diff": 3.3910,
    "yac_oe_diff": 2.6533,
    "ryoe_diff": 0.2759,
    "intercept": -0.2878,
}

MARGIN_COEFFICIENTS_V1_RATING_ONLY = {
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


def predict_margin(
    rating_diff: float, is_neutral_site: bool, rest_diff: float,
    cpoe_diff: float = 0.0, separation_diff: float = 0.0,
    yac_oe_diff: float = 0.0, ryoe_diff: float = 0.0,
    cushion_diff: float = 0.0, catch_pct_diff: float = 0.0,
    stacked_box_diff: float = 0.0,
) -> float:
    home_field = 0.0 if is_neutral_site else 1.0
    return (
        MARGIN_COEFFICIENTS["rating_diff"] * rating_diff
        + MARGIN_COEFFICIENTS["home_field"] * home_field
        + MARGIN_COEFFICIENTS["rest_diff"] * rest_diff
        + MARGIN_COEFFICIENTS["cpoe_diff"] * cpoe_diff
        + MARGIN_COEFFICIENTS["separation_diff"] * separation_diff
        + MARGIN_COEFFICIENTS["yac_oe_diff"] * yac_oe_diff
        + MARGIN_COEFFICIENTS["ryoe_diff"] * ryoe_diff
        + MARGIN_COEFFICIENTS["cushion_diff"] * cushion_diff
        + MARGIN_COEFFICIENTS["catch_pct_diff"] * catch_pct_diff
        + MARGIN_COEFFICIENTS["stacked_box_diff"] * stacked_box_diff
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
    cpoe_diff: float = 0.0, separation_diff: float = 0.0,
    yac_oe_diff: float = 0.0, ryoe_diff: float = 0.0,
    cushion_diff: float = 0.0, catch_pct_diff: float = 0.0,
    stacked_box_diff: float = 0.0,
) -> dict:
    """
    Full prediction for one game: predicted points for each team, spread,
    total, and win probability — the single translation layer that
    serves spread, moneyline, and totals all at once (Section 11.4's
    design).

    The seven NGS-derived parameters default to 0.0 (no effect) for
    callers that don't supply real team tracking data — a correct but
    unenhanced prediction, not an error. Real values come from
    model.layer2_ngs.compute_team_ngs_features().
    """
    rating_diff = home_rating - away_rating
    margin = predict_margin(
        rating_diff, is_neutral_site, rest_diff,
        cpoe_diff=cpoe_diff, separation_diff=separation_diff,
        yac_oe_diff=yac_oe_diff, ryoe_diff=ryoe_diff,
        cushion_diff=cushion_diff, catch_pct_diff=catch_pct_diff,
        stacked_box_diff=stacked_box_diff,
    )
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


def build_week_predictions(ratings: pd.DataFrame, upcoming_games: pd.DataFrame, ngs_features: pd.DataFrame = None) -> dict:
    """
    Builds the model_predictions dict odds_watch_job.py's compute_divergences()
    expects, keyed by home_team.

    ngs_features: optional output of model.layer2_ngs.compute_team_ngs_features().
    If not provided, predictions still work correctly but without the
    validated Layer 2 improvement (see model/prediction.py's module
    docstring) — teams missing from ngs_features individually also
    fall back to 0.0 (no effect) for just that team's contribution,
    rather than failing the whole prediction.
    """
    predictions = {}
    for _, game in upcoming_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        if home not in ratings.index or away not in ratings.index:
            continue

        cpoe_diff = separation_diff = yac_oe_diff = ryoe_diff = 0.0
        cushion_diff = catch_pct_diff = stacked_box_diff = 0.0
        if ngs_features is not None and home in ngs_features.index and away in ngs_features.index:
            cpoe_diff = ngs_features.loc[home, "team_cpoe"] - ngs_features.loc[away, "team_cpoe"]
            separation_diff = ngs_features.loc[home, "team_avg_separation"] - ngs_features.loc[away, "team_avg_separation"]
            yac_oe_diff = ngs_features.loc[home, "team_yac_over_expected"] - ngs_features.loc[away, "team_yac_over_expected"]
            ryoe_diff = ngs_features.loc[home, "team_ryoe"] - ngs_features.loc[away, "team_ryoe"]
            if "team_avg_cushion" in ngs_features.columns:
                cushion_diff = ngs_features.loc[home, "team_avg_cushion"] - ngs_features.loc[away, "team_avg_cushion"]
                catch_pct_diff = ngs_features.loc[home, "team_catch_pct"] - ngs_features.loc[away, "team_catch_pct"]
                stacked_box_diff = ngs_features.loc[home, "team_stacked_box_pct"] - ngs_features.loc[away, "team_stacked_box_pct"]

        result = predict_game(
            home_rating=ratings.loc[home, "total_rating"],
            away_rating=ratings.loc[away, "total_rating"],
            home_offense=ratings.loc[home, "offense_voa"],
            away_offense=ratings.loc[away, "offense_voa"],
            is_neutral_site=game.get("is_neutral_site", False),
            rest_diff=game.get("home_rest", 7) - game.get("away_rest", 7),
            wind=game.get("wind", 0.0),
            cpoe_diff=cpoe_diff, separation_diff=separation_diff,
            yac_oe_diff=yac_oe_diff, ryoe_diff=ryoe_diff,
            cushion_diff=cushion_diff, catch_pct_diff=catch_pct_diff,
            stacked_box_diff=stacked_box_diff,
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
