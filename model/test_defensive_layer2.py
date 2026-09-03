"""
Tests whether DEFENSIVE NGS features (CPOE/separation/YAC-OE allowed)
add real value beyond the offensive Layer 2 features already
validated -- using genuinely held-out validation FROM THE START this
time (fit on 2021-2022, evaluate only on 2023), after the earlier
in-sample-coefficient mistake found in walk_forward_layer2_test.py.

Every prior Layer 2 feature measures a team's own offensive skill.
This tests the opposite: how much of these same skills a team's
defense ALLOWS to opponents -- a genuinely different signal, not a
rehash of the offensive ones.
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
from model.layer2_ngs import compute_team_ngs_features, compute_defensive_ngs_features, load_ngs_data

BACKTEST_SEASONS = [2021, 2022, 2023]
BACKTEST_WEEKS = range(4, 18)


def run_walk_forward_test():
    rows = []

    for season in BACKTEST_SEASONS:
        print(f"Season {season}...")
        schedules = load_schedules(seasons=[season])
        full_season_df = load_season(season)

        ngs_preloaded = {
            "passing": load_ngs_data(season, "passing"),
            "receiving": load_ngs_data(season, "receiving"),
            "rushing": load_ngs_data(season, "rushing"),
        }

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

            ngs_off = compute_team_ngs_features(season, through_week=week, preloaded_data=ngs_preloaded)
            try:
                ngs_def = compute_defensive_ngs_features(season, through_week=week, preloaded_data=ngs_preloaded)
            except ValueError:
                continue

            week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue
                if home not in ngs_off.index or away not in ngs_off.index:
                    continue
                if home not in ngs_def.index or away not in ngs_def.index:
                    continue

                actual_margin = game["home_score"] - game["away_score"]

                rows.append({
                    "season": season, "week": week,
                    "rating_diff": ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"],
                    "home_field": 0.0 if game.get("is_neutral_site", False) else 1.0,
                    "rest_diff": game["home_rest"] - game["away_rest"],
                    "cpoe_diff": ngs_off.loc[home, "team_cpoe"] - ngs_off.loc[away, "team_cpoe"],
                    "separation_diff": ngs_off.loc[home, "team_avg_separation"] - ngs_off.loc[away, "team_avg_separation"],
                    "yac_oe_diff": ngs_off.loc[home, "team_yac_over_expected"] - ngs_off.loc[away, "team_yac_over_expected"],
                    "ryoe_diff": ngs_off.loc[home, "team_ryoe"] - ngs_off.loc[away, "team_ryoe"],
                    "cpoe_allowed_diff": ngs_def.loc[home, "team_cpoe_allowed"] - ngs_def.loc[away, "team_cpoe_allowed"],
                    "separation_allowed_diff": ngs_def.loc[home, "team_separation_allowed"] - ngs_def.loc[away, "team_separation_allowed"],
                    "yac_oe_allowed_diff": ngs_def.loc[home, "team_yac_oe_allowed"] - ngs_def.loc[away, "team_yac_oe_allowed"],
                    "ryoe_allowed_diff": ngs_def.loc[home, "team_ryoe_allowed"] - ngs_def.loc[away, "team_ryoe_allowed"],
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
                })

    return pd.DataFrame(rows)


def held_out_test(train, test, feature_cols, label):
    train = train.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)
    X_train = np.column_stack([train[feature_cols].values, np.ones(len(train))])
    y_train = train["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack([test[feature_cols].values, np.ones(len(test))])
    y_test = test["actual_margin"].values
    pred_test = X_test @ coef
    straight_up = ((pred_test > 0) == test["actual_home_win"]).mean()
    mae = np.mean(np.abs(y_test - pred_test))
    print(f"{label}: straight-up = {straight_up:.4f}, MAE = {mae:.2f} (n_train={len(train)}, n_test={len(test)})")
    return coef, straight_up, mae


if __name__ == "__main__":
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    print(f"\nTrain: {len(train)} games (2021-2022), Test: {len(test)} games (2023, fully held out)\n")

    offense_only = ["rating_diff", "home_field", "rest_diff", "cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"]
    held_out_test(train, test, offense_only, "Offense-only Layer 2 (already in production)")

    with_defense = offense_only + ["cpoe_allowed_diff", "separation_allowed_diff", "yac_oe_allowed_diff", "ryoe_allowed_diff"]
    coef, acc, mae = held_out_test(train, test, with_defense, "Offense + Defense Layer 2           ")

    print("\nDefensive coefficients (fit on 2021-2022):")
    for name, val in zip(with_defense + ["intercept"], coef):
        if "allowed" in name:
            print(f"  {name}: {val:.4f}")
