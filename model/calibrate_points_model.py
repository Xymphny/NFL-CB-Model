"""
Points-prediction layer calibration — Section 11.4.

Approach: use each team's FINAL rating from season N to predict games in
season N+1. This is a deliberate choice to avoid lookahead bias in this
first pass — using a season's own final rating to "predict" games within
that same season would leak the outcome being predicted into the
predictor (Section 11's no-lookahead-bias discipline). A proper
within-season walk-forward version (updating ratings week by week and
only using data available before each game) is real future work, not
built here — see the caveat printed at the end of this script.

Output: regression coefficients for
    margin = a * rating_diff + b * home_field + c * rest_diff
    total  = d * combined_offense + e * wind + f

These two equations ARE the "predicted-points-per-team" layer described
in the spec: home_points = (total + margin) / 2, away_points =
(total - margin) / 2.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (
    add_situation_buckets,
    score_all_plays,
    compute_baselines,
    compute_raw_voa,
    opponent_adjust,
    filter_garbage_time,
    team_ratings,
)

SEASONS_TO_CALIBRATE = [2019, 2020, 2021, 2022, 2023]


def compute_season_ratings(season: int) -> pd.DataFrame:
    """Run the full Layer 1 pipeline for one season, no recency weighting
    (we want the final, full-season rating here, not a recent-form snapshot)."""
    df = load_season(season)
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True)
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    ratings = team_ratings(df, use_recency_weights=False)
    ratings["season"] = season
    return ratings


def build_calibration_dataset() -> pd.DataFrame:
    """
    For each season N+1 game, attach both teams' season-N final ratings.
    """
    print("Computing ratings for each season (this pulls real data per season)...")
    ratings_by_season = {}
    for season in SEASONS_TO_CALIBRATE:
        print(f"  season {season}...")
        ratings_by_season[season] = compute_season_ratings(season)

    rows = []
    for season in SEASONS_TO_CALIBRATE[1:]:
        prior_season = season - 1
        if prior_season not in ratings_by_season:
            continue
        prior = ratings_by_season[prior_season]

        sched = load_schedules(seasons=[season])
        for _, game in sched.iterrows():
            home, away = game["home_team"], game["away_team"]
            if home not in prior.index or away not in prior.index:
                continue
            if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
                continue

            rows.append({
                "season": season,
                "home_team": home,
                "away_team": away,
                "home_prior_off": prior.loc[home, "offense_voa"],
                "home_prior_def": prior.loc[home, "defense_voa"],
                "away_prior_off": prior.loc[away, "offense_voa"],
                "away_prior_def": prior.loc[away, "defense_voa"],
                "rating_diff": prior.loc[home, "total_rating"] - prior.loc[away, "total_rating"],
                "combined_offense": prior.loc[home, "offense_voa"] + prior.loc[away, "offense_voa"],
                "is_neutral_site": game["is_neutral_site"],
                "rest_diff": game["home_rest"] - game["away_rest"],
                "wind": game["wind"] if pd.notna(game["wind"]) else 0.0,
                "actual_margin": game["home_score"] - game["away_score"],
                "actual_total": game["home_score"] + game["away_score"],
            })

    return pd.DataFrame(rows)


def calibrate(dataset: pd.DataFrame):
    """Simple OLS via numpy least squares — no new dependency needed."""
    dataset = dataset.copy()
    dataset["home_field"] = (~dataset["is_neutral_site"]).astype(float)

    # Margin model: rating_diff, home_field, rest_diff
    X_margin = dataset[["rating_diff", "home_field", "rest_diff"]].values
    X_margin = np.column_stack([X_margin, np.ones(len(X_margin))])  # intercept
    y_margin = dataset["actual_margin"].values
    coef_margin, _, _, _ = np.linalg.lstsq(X_margin, y_margin, rcond=None)

    pred_margin = X_margin @ coef_margin
    resid_margin = y_margin - pred_margin
    r2_margin = 1 - np.sum(resid_margin**2) / np.sum((y_margin - y_margin.mean())**2)

    # Total model: combined_offense, wind
    X_total = dataset[["combined_offense", "wind"]].values
    X_total = np.column_stack([X_total, np.ones(len(X_total))])
    y_total = dataset["actual_total"].values
    coef_total, _, _, _ = np.linalg.lstsq(X_total, y_total, rcond=None)

    pred_total = X_total @ coef_total
    resid_total = y_total - pred_total
    r2_total = 1 - np.sum(resid_total**2) / np.sum((y_total - y_total.mean())**2)

    return {
        "margin_coefficients": {
            "rating_diff": coef_margin[0],
            "home_field": coef_margin[1],
            "rest_diff": coef_margin[2],
            "intercept": coef_margin[3],
        },
        "margin_r2": r2_margin,
        "margin_mae": np.mean(np.abs(resid_margin)),
        "total_coefficients": {
            "combined_offense": coef_total[0],
            "wind": coef_total[1],
            "intercept": coef_total[2],
        },
        "total_r2": r2_total,
        "total_mae": np.mean(np.abs(resid_total)),
        "n_games": len(dataset),
    }


if __name__ == "__main__":
    dataset = build_calibration_dataset()
    print(f"\nCalibration dataset: {len(dataset)} games across seasons {SEASONS_TO_CALIBRATE[1:]}")

    results = calibrate(dataset)

    print("\n=== Margin model: actual_margin ~ rating_diff + home_field + rest_diff ===")
    for k, v in results["margin_coefficients"].items():
        print(f"  {k}: {v:+.4f}")
    print(f"  R^2: {results['margin_r2']:.4f}")
    print(f"  MAE: {results['margin_mae']:.2f} points")

    print("\n=== Total model: actual_total ~ combined_offense + wind ===")
    for k, v in results["total_coefficients"].items():
        print(f"  {k}: {v:+.4f}")
    print(f"  R^2: {results['total_r2']:.4f}")
    print(f"  MAE: {results['total_mae']:.2f} points")

    print(f"\nn = {results['n_games']} games")
    print(
        "\nCAVEAT: this calibration uses each team's prior-SEASON final rating "
        "to predict next-season games, chosen specifically to avoid lookahead "
        "bias in this first pass. It does NOT yet reflect a proper within-season "
        "walk-forward setup (updating ratings week-by-week using only data "
        "available before each game), which is real future work per the "
        "spec's Section 11 backtesting protocol. Treat this R^2/MAE as a "
        "baseline sanity check, not the model's true in-season accuracy."
    )
