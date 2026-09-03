"""
Tests whether real ESPN win-rate metrics (Pass Rush/Run Stop/Pass
Block/Run Block Win Rate) add genuine predictive value -- the closest
free, real answer found to the PFF-charting gap. Uses PRIOR season's
final win rates as a feature predicting the CURRENT season's games
(analogous to how last-season rating already serves as a prior) --
the only honest design given this data is a season-end aggregate, not
weekly like the NGS Layer 2 features.

Genuinely held-out validation FROM THE START (fit on 2021-2022 games,
evaluate only on 2023 games) -- the lesson from the earlier Layer 2
overfitting mistake, applied here from the beginning rather than
discovered partway through.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_home_field_and_rest, team_ratings,
)
from model.win_rate_history_2020_2022 import WIN_RATES_BY_SEASON

BACKTEST_SEASONS = [2021, 2022, 2023]
BACKTEST_WEEKS = range(4, 18)


def run_walk_forward_test():
    rows = []

    for season in BACKTEST_SEASONS:
        print(f"Season {season}...")
        prior_win_rates = WIN_RATES_BY_SEASON[season - 1]

        schedules = load_schedules(seasons=[season])
        full_season_df = load_season(season)

        for week in BACKTEST_WEEKS:
            df = full_season_df[full_season_df["week"] < week].copy()
            df = add_situation_buckets(df)
            df = score_all_plays(df, use_turnover_luck_adjustment=True)
            df = filter_garbage_time(df)
            df = add_home_field_and_rest(df, schedules)
            baselines = compute_baselines(df)
            df = compute_raw_voa(df, baselines)
            df = opponent_adjust(df, iterations=3, regression=0.5)
            ratings = team_ratings(df, use_recency_weights=True)

            week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue
                if home not in prior_win_rates["prwr"] or away not in prior_win_rates["prwr"]:
                    continue

                actual_margin = game["home_score"] - game["away_score"]

                rows.append({
                    "season": season, "week": week,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "home_field": 0.0 if game.get("is_neutral_site", False) else 1.0,
                    "rest_diff": game["home_rest"] - game["away_rest"],
                    "prwr_diff": prior_win_rates["prwr"][home] - prior_win_rates["prwr"][away],
                    "rswr_diff": prior_win_rates["rswr"][home] - prior_win_rates["rswr"][away],
                    "pbwr_diff": prior_win_rates["pbwr"][home] - prior_win_rates["pbwr"][away],
                    "rbwr_diff": prior_win_rates["rbwr"][home] - prior_win_rates["rbwr"][away],
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
                })

    return pd.DataFrame(rows)


def fit_and_score(train, test, feature_cols):
    train = train.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)
    X_train = np.column_stack([train[feature_cols].values, np.ones(len(train))])
    y_train = train["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack([test[feature_cols].values, np.ones(len(test))])
    y_test = test["actual_margin"].values
    pred_test = X_test @ coef
    mae = np.mean(np.abs(y_test - pred_test))
    straight_up = ((pred_test > 0) == test["actual_home_win"]).mean()
    return coef, mae, straight_up


if __name__ == "__main__":
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    print(f"\nTrain: {len(train)} games (2021-2022), Test: {len(test)} games (2023, fully held out)\n")

    baseline_features = ["rating_diff", "home_field", "rest_diff"]
    _, mae_b, acc_b = fit_and_score(train, test, baseline_features)
    print(f"Baseline (rating only): MAE = {mae_b:.2f}, straight-up = {acc_b:.4f}")

    with_win_rates = baseline_features + ["prwr_diff", "rswr_diff", "pbwr_diff", "rbwr_diff"]
    coef, mae_w, acc_w = fit_and_score(train, test, with_win_rates)
    print(f"With real prior-season win rates: MAE = {mae_w:.2f}, straight-up = {acc_w:.4f}")

    print(f"\nChange: MAE {mae_b - mae_w:+.3f}, straight-up accuracy {acc_w - acc_b:+.4f}")
    print(f"\nCoefficients: {dict(zip(with_win_rates + ['intercept'], coef))}")
