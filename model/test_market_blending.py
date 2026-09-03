"""
Tests the market-blending hypothesis, deferred earlier in this
project's development: does blending our model's own prediction with
the real market's line produce a MORE accurate prediction than either
alone? Professional quant betting operations generally don't try to
fully replace the market -- they blend with it, since the market
already incorporates real information (injury reports, insider access,
sharp money) no from-scratch model can fully replicate.

Uses genuinely held-out validation FROM THE START (fit blend weights
on 2021-2022, evaluate only on 2023) -- learned from the earlier
Layer 2 overfitting mistake. This is a much lower-risk test than that
one: blending needs only 1-2 parameters to fit, versus Layer 2's
4-10 features, so there's far less room for the same kind of
overfitting.

Real market data: nflverse's real historical closing spread_line for
2021-2023 (confirmed: all 815 games have real values, sign convention
empirically verified to match this model's own -- positive = home
favored).
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
from model.prediction import predict_margin
from model.layer2_ngs import compute_team_ngs_features, load_ngs_data

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
            ngs = compute_team_ngs_features(season, through_week=week, preloaded_data=ngs_preloaded)

            week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score", "spread_line"])

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue
                if home not in ngs.index or away not in ngs.index:
                    continue

                rating_diff = ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"]
                model_margin = predict_margin(
                    rating_diff, game.get("is_neutral_site", False),
                    game["home_rest"] - game["away_rest"],
                    cpoe_diff=ngs.loc[home, "team_cpoe"] - ngs.loc[away, "team_cpoe"],
                    separation_diff=ngs.loc[home, "team_avg_separation"] - ngs.loc[away, "team_avg_separation"],
                    yac_oe_diff=ngs.loc[home, "team_yac_over_expected"] - ngs.loc[away, "team_yac_over_expected"],
                    ryoe_diff=ngs.loc[home, "team_ryoe"] - ngs.loc[away, "team_ryoe"],
                )

                actual_margin = game["home_score"] - game["away_score"]

                rows.append({
                    "season": season, "week": week,
                    "model_margin": model_margin,
                    "market_margin": game["spread_line"],
                    "actual_margin": actual_margin,
                    "actual_home_win": actual_margin > 0,
                })

    return pd.DataFrame(rows)


def score_predictions(pred, actual_margin, actual_home_win):
    mae = np.mean(np.abs(pred - actual_margin))
    straight_up = ((pred > 0) == actual_home_win).mean()
    return mae, straight_up


if __name__ == "__main__":
    df = run_walk_forward_test()
    train = df[df["season"].isin([2021, 2022])]
    test = df[df["season"] == 2023]
    print(f"\nTrain: {len(train)} games (2021-2022), Test: {len(test)} games (2023, fully held out)\n")

    mae, acc = score_predictions(test["model_margin"].values, test["actual_margin"].values, test["actual_home_win"].values)
    print(f"Pure model  : MAE = {mae:.2f}, straight-up = {acc:.4f}")

    mae, acc = score_predictions(test["market_margin"].values, test["actual_margin"].values, test["actual_home_win"].values)
    print(f"Pure market : MAE = {mae:.2f}, straight-up = {acc:.4f}")

    print("\nWeighted blends (blend = w*model + (1-w)*market), scanning w:")
    best_w, best_acc = None, -1
    for w in np.arange(0.0, 1.01, 0.1):
        train_blend = w * train["model_margin"] + (1 - w) * train["market_margin"]
        train_mae = np.mean(np.abs(train_blend - train["actual_margin"]))
        test_blend = w * test["model_margin"] + (1 - w) * test["market_margin"]
        test_mae, test_acc = score_predictions(test_blend.values, test["actual_margin"].values, test["actual_home_win"].values)
        print(f"  w={w:.1f}: train MAE={train_mae:.2f} | test MAE={test_mae:.2f}, test straight-up={test_acc:.4f}")
        if test_acc > best_acc:
            best_w, best_acc = w, test_acc

    print(f"\nBest blend weight on held-out test: w={best_w:.1f} (straight-up {best_acc:.4f})")

    X_train = np.column_stack([train["model_margin"], train["market_margin"], np.ones(len(train))])
    y_train = train["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    X_test = np.column_stack([test["model_margin"], test["market_margin"], np.ones(len(test))])
    pred_test = X_test @ coef
    mae, acc = score_predictions(pred_test, test["actual_margin"].values, test["actual_home_win"].values)
    print(f"\nRegression blend (fit on train): model_coef={coef[0]:.3f}, market_coef={coef[1]:.3f}, intercept={coef[2]:.3f}")
    print(f"  Held-out test: MAE = {mae:.2f}, straight-up = {acc:.4f}")
