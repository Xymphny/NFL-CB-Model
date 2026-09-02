"""
Walk-forward backtesting harness -- Section 11's validation protocol,
finally turned into an actual procedure rather than just a stated
principle.

For each week W in each season, ratings are computed using ONLY data
through week W-1 (in-season, recency-weighted -- the same computation
weekly_job.py does live each week), then used to predict week W's real
games via the points-prediction layer, then checked against what
actually happened. This is strict walk-forward: no future information
ever leaks into a prediction, unlike calibrate_points_model.py's
prior-season-only approach (which that script's own output explicitly
caveats as a simpler stand-in for this).
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
from model.prediction import predict_game, margin_to_win_probability

BACKTEST_SEASONS = [2021, 2022, 2023]
BACKTEST_WEEKS = range(4, 18)


def compute_ratings_through_week(full_season_df: pd.DataFrame, through_week: int, schedules: pd.DataFrame) -> pd.DataFrame:
    df = full_season_df[full_season_df["week"] < through_week].copy()
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True)
    df = filter_garbage_time(df)
    df = add_home_field_and_rest(df, schedules)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    return team_ratings(df, use_recency_weights=True)


def run_backtest():
    rows = []

    for season in BACKTEST_SEASONS:
        print(f"Backtesting {season}...")
        schedules = load_schedules(seasons=[season])
        full_season_df = load_season(season)  # loaded ONCE per season, not once per week

        for week in BACKTEST_WEEKS:
            ratings = compute_ratings_through_week(full_season_df, week, schedules)
            week_games = schedules[schedules["week"] == week].dropna(subset=["home_score", "away_score"])

            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ratings.index or away not in ratings.index:
                    continue

                pred = predict_game(
                    home_rating=ratings.loc[home, "total_rating"],
                    away_rating=ratings.loc[away, "total_rating"],
                    home_offense=ratings.loc[home, "offense_voa"],
                    away_offense=ratings.loc[away, "offense_voa"],
                    is_neutral_site=bool(game.get("is_neutral_site", False)),
                    rest_diff=float(game["home_rest"] - game["away_rest"]),
                    wind=game.get("wind", 0.0),
                )

                actual_margin = game["home_score"] - game["away_score"]
                actual_total = game["home_score"] + game["away_score"]
                actual_home_win = actual_margin > 0

                rows.append({
                    "season": season, "week": week,
                    "home_team": home, "away_team": away,
                    "predicted_spread": pred["spread"], "predicted_total": pred["total"],
                    "predicted_win_prob_home": pred["win_prob_home"],
                    "actual_margin": actual_margin, "actual_total": actual_total,
                    "actual_home_win": actual_home_win,
                    "spread_error": pred["spread"] - actual_margin,
                    "total_error": pred["total"] - actual_total,
                })

    return pd.DataFrame(rows)


def score_backtest(df: pd.DataFrame) -> dict:
    spread_mae = df["spread_error"].abs().mean()
    total_mae = df["total_error"].abs().mean()
    brier = ((df["predicted_win_prob_home"] - df["actual_home_win"].astype(float)) ** 2).mean()
    predicted_home_win = df["predicted_spread"] > 0
    straight_up_accuracy = (predicted_home_win == df["actual_home_win"]).mean()

    return {
        "n_games": len(df),
        "spread_mae": spread_mae,
        "total_mae": total_mae,
        "brier_score": brier,
        "straight_up_accuracy": straight_up_accuracy,
    }


if __name__ == "__main__":
    df = run_backtest()
    print(f"\n{len(df)} games backtested across {BACKTEST_SEASONS}, weeks {list(BACKTEST_WEEKS)}")

    overall = score_backtest(df)
    print("\n=== Overall walk-forward backtest results ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== By season ===")
    for season in BACKTEST_SEASONS:
        season_df = df[df["season"] == season]
        season_results = score_backtest(season_df)
        print(f"  {season}: MAE(spread)={season_results['spread_mae']:.2f}, "
              f"MAE(total)={season_results['total_mae']:.2f}, "
              f"Brier={season_results['brier_score']:.4f}, "
              f"straight-up acc={season_results['straight_up_accuracy']:.3f}")

    print("\n=== By week (does accuracy improve as the season progresses?) ===")
    for week in [4, 8, 12, 16]:
        week_df = df[df["week"] == week]
        if len(week_df) == 0:
            continue
        week_results = score_backtest(week_df)
        print(f"  Week {week}: MAE(spread)={week_results['spread_mae']:.2f}, "
              f"straight-up acc={week_results['straight_up_accuracy']:.3f}")
