"""
Points-prediction layer — Section 11.4. Translates team ratings into
predicted spread, total, and win probability.

MARGIN_COEFFICIENTS below is the BASE Layer 2 model (CPOE, separation,
YAC-over-expected, RYOE) — NOT the further-extended version (+ cushion,
catch%, stacked-box rate) that was briefly in production. Real,
important correction found by testing: fitting one set of coefficients
on all 584 games at once (as the original walk-forward test did)
overstates real accuracy, because the coefficients themselves get to
"see" every outcome during fitting even though the underlying ratings/
NGS features have zero lookahead. A genuinely held-out test (fit on
2021-2022 only, evaluate on 2023 -- data the coefficients never saw)
showed:
  - Rating-only baseline: 59.49% straight-up
  - Base Layer 2 (this model): 60.51% straight-up -- a real, if far
    more modest than previously reported, ~1-point improvement
  - Extended Layer 2 (+3 more features): ALSO 60.51% -- identical,
    meaning the extended features added ZERO genuine value and were
    only appearing to help due to overfitting on the in-sample test.
The originally-reported 58.22% -> 64.90% figures were real numbers
from a real test, but that test's methodology (one coefficient fit
across the full sample, then evaluated on the same sample) overstated
what to expect on genuinely new data. This is the honest, corrected
number. See model/walk_forward_layer2_test.py and the held-out
validation in model/test_layer2_held_out.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Base Layer 2, fit on all of 2021-2023 for production (standard
# practice: validate via held-out split first, then refit on all
# available data for the final model) -- validated above to
# genuinely generalize, unlike the extended version.
# Properly co-calibrated on the expanded 2016-2021 training set (NGS
# data's earliest available year), tested on 2022-2023 (389 games,
# fully held out) -- a substantial, real, validated improvement:
# straight-up accuracy 60.93% (rating alone) -> 62.72% (+ Layer 2) ->
# 65.55% (+ Elo ensemble). See model/test_full_ensemble.py.
#
# rating_diff's own coefficient is small (0.108) because Elo captures
# much of the same "team quality" signal -- real multicollinearity
# between two different measures of the same underlying thing, not a
# sign either one is broken. The COMBINED model's held-out performance
# is what was validated, not any single coefficient's size.
MARGIN_COEFFICIENTS = {
    "rating_diff": 0.1078,
    "home_field": 5.5271,
    "rest_diff": 0.0229,
    "cpoe_diff": 0.4622,
    "separation_diff": 2.7941,
    "yac_oe_diff": 0.1230,
    "ryoe_diff": 0.3432,
    "elo_diff": 0.0348,
    "intercept": -6.6607,
}

# Retained for reference/comparison -- fit under the OLD half_life=6
# recency default, before the audit caught the staleness, and before
# the Elo ensemble was added.
MARGIN_COEFFICIENTS_STALE_PRE_RECENCY_FIX = {
    "rating_diff": 12.9834,
    "home_field": 2.7266,
    "rest_diff": 0.1862,
    "cpoe_diff": 0.5132,
    "separation_diff": 3.3910,
    "yac_oe_diff": 2.6533,
    "ryoe_diff": 0.2759,
    "intercept": -0.2878,
}

# Retained for reference -- the version between the recency fix and
# the Elo ensemble addition (rating+Layer2 only, no Elo term).
MARGIN_COEFFICIENTS_PRE_ELO = {
    "rating_diff": 21.4704,
    "home_field": 3.7442,
    "rest_diff": 0.0925,
    "cpoe_diff": 0.3625,
    "separation_diff": 2.4554,
    "yac_oe_diff": 2.1408,
    "ryoe_diff": 1.6545,
    "intercept": -1.6367,
}

# Retained for reference/comparison ONLY -- do NOT use in production.
# Tested and found to add zero genuine value over the base model above
# under honest held-out validation (identical 60.51% straight-up
# accuracy on 2023 data neither version's coefficients had seen).
MARGIN_COEFFICIENTS_EXTENDED_NOT_RECOMMENDED = {
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
    stacked_box_diff: float = 0.0, elo_diff: float = None,
) -> float:
    """
    cushion_diff, catch_pct_diff, stacked_box_diff kept as accepted
    parameters for backward compatibility with existing callers, but
    have NO effect on the prediction -- MARGIN_COEFFICIENTS (the base
    model) doesn't include them, per the held-out validation above.

    elo_diff defaults to None (not 0.0) -- REAL BUG FOUND AND FIXED:
    MARGIN_COEFFICIENTS was co-calibrated WITH Elo present, which
    redistributed weight away from rating_diff onto elo_diff
    (rating_diff's own coefficient shrank from ~21 to ~0.11, since Elo
    now carries most of that signal). Silently defaulting elo_diff to
    0.0 and using the SAME co-calibrated coefficients anyway would make
    predictions almost entirely flat whenever Elo is unavailable for
    any reason -- a 200x understatement of rating_diff's real effect,
    not a small graceful loss of one bonus feature (found and measured
    directly: a real rating_diff of 0.1 would predict 0.01 points
    under the co-calibrated coefficients vs 2.15 points under the
    correct pre-Elo ones). Using None as the sentinel for "Elo
    unavailable" switches to MARGIN_COEFFICIENTS_PRE_ELO instead, which
    was properly calibrated for exactly this case.
    """
    coefficients = MARGIN_COEFFICIENTS if elo_diff is not None else MARGIN_COEFFICIENTS_PRE_ELO
    elo_diff = elo_diff if elo_diff is not None else 0.0

    home_field = 0.0 if is_neutral_site else 1.0
    return (
        coefficients["rating_diff"] * rating_diff
        + coefficients["home_field"] * home_field
        + coefficients["rest_diff"] * rest_diff
        + coefficients["cpoe_diff"] * cpoe_diff
        + coefficients["separation_diff"] * separation_diff
        + coefficients["yac_oe_diff"] * yac_oe_diff
        + coefficients["ryoe_diff"] * ryoe_diff
        + coefficients.get("elo_diff", 0.0) * elo_diff
        + coefficients["intercept"]
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
    stacked_box_diff: float = 0.0, elo_diff: float = None,
) -> dict:
    """
    Full prediction for one game: predicted points for each team, spread,
    total, and win probability — the single translation layer that
    serves spread, moneyline, and totals all at once (Section 11.4's
    design).

    The NGS-derived parameters default to 0.0 (no effect, small
    additive features -- graceful degradation is correct for these).
    elo_diff defaults to None, not 0.0 -- see predict_margin's
    docstring for why: unlike the NGS features, Elo isn't a small
    bonus signal, the production coefficients were co-calibrated
    assuming it's present, so "unavailable" and "available but zero"
    need to be distinguished, not silently treated the same.
    """
    rating_diff = home_rating - away_rating
    margin = predict_margin(
        rating_diff, is_neutral_site, rest_diff,
        cpoe_diff=cpoe_diff, separation_diff=separation_diff,
        yac_oe_diff=yac_oe_diff, ryoe_diff=ryoe_diff,
        cushion_diff=cushion_diff, catch_pct_diff=catch_pct_diff,
        stacked_box_diff=stacked_box_diff, elo_diff=elo_diff,
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


def build_week_predictions(ratings: pd.DataFrame, upcoming_games: pd.DataFrame, ngs_features: pd.DataFrame = None, elo_ratings: dict = None) -> dict:
    """
    Builds the model_predictions dict odds_watch_job.py's compute_divergences()
    expects, keyed by home_team.

    ngs_features: optional output of model.layer2_ngs.compute_team_ngs_features().
    elo_ratings: optional {team: current_elo} dict, the last row's
    post-game ratings from model.elo_rating.compute_elo_walk_forward()
    (or equivalent live-updated state). If not provided, predictions
    still work correctly but without the validated Elo-ensemble
    improvement — teams missing from either source individually also
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

        elo_diff = None
        if elo_ratings is not None and home in elo_ratings and away in elo_ratings:
            elo_diff = elo_ratings[home] - elo_ratings[away]

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
            stacked_box_diff=stacked_box_diff, elo_diff=elo_diff,
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
