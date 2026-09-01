"""
Generates a real-data preview of the dashboard using the completed
2023 season — NOT fabricated data. Every number here comes from the
actual tested pipeline running against real historical games, including
real closing lines from nflverse's schedule data for the market
comparison. This exists purely to let the site be visually/functionally
checked with real content before the 2026 season provides its own.

Clearly labeled as 2023 throughout — this is a demo of real historical
output, not a simulation of what 2026 will look like.
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
)
from model.team_profile import build_team_profile
from model.prediction import predict_game, margin_to_win_probability
from model.market_comparison import flag_divergence

PREVIEW_SEASON = 2023
PREVIEW_WEEKS = [4, 9, 14, 18]  # a spread across the season, to also demonstrate multi-week snapshot history
DIVERGENCE_WEEK = 18


def compute_and_write_ratings(season: int, week: int, output_dir: str):
    df = load_season(season)
    df = df[df["week"] <= week].copy()
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True)
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    ratings = team_ratings(df, use_recency_weights=True)
    profile = build_team_profile(df, ratings)

    ratings_dir = os.path.join(output_dir, "ratings")
    os.makedirs(ratings_dir, exist_ok=True)
    output_file = os.path.join(ratings_dir, f"{season}-week-{week:02d}.json")

    payload = {
        "season": season,
        "week": week,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "prior_source": "none (2023 demo data, real historical season)",
        "ratings": profile.reset_index().rename(columns={"index": "team"}).to_dict(orient="records"),
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  wrote {output_file} ({len(profile)} teams)")
    return profile


def compute_and_write_divergence(season: int, week: int, ratings, output_dir: str):
    """
    Uses REAL closing lines from nflverse's schedule data — not
    synthetic odds. Market win probability is derived from the real
    spread via the same margin-to-probability conversion the model
    itself uses (a standard normal-CDF approximation), since real
    historical moneyline data isn't in the schedule file — everything
    else here is genuinely real.
    """
    sched = load_schedules(seasons=[season])
    week_games = sched[sched["week"] == week].dropna(subset=["spread_line", "total_line"])

    divergences = []
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

        # Real closing line — the market's actual point spread/total for this real game.
        market_spread = -float(game["spread_line"])  # nflverse convention flip to match ours (positive = home favored)
        market_total = float(game["total_line"])
        market_win_prob_home = margin_to_win_probability(market_spread)

        divergence = flag_divergence(
            model_spread=pred["spread"], model_total=pred["total"], model_win_prob_home=pred["win_prob_home"],
            market_spread=market_spread, market_total=market_total,
            market_odds_home=market_win_prob_home, market_odds_away=1 - market_win_prob_home,
        )

        divergences.append({
            "home_team": home, "away_team": away,
            "market_win_prob_home_fair": market_win_prob_home,
            "market_spread": market_spread,
            "market_total": market_total,
            **divergence,
        })

    divergence_dir = os.path.join(output_dir, "divergence")
    os.makedirs(divergence_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = os.path.join(divergence_dir, f"{season}-week-{week:02d}-{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump({
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "season": season,
            "week": week,
            "note": "2023 demo — real closing lines, real model predictions, not live 2026 data",
            "divergences": divergences,
        }, f, indent=2)

    print(f"  wrote {output_file} ({len(divergences)} real games, using real closing lines)")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./preview_data"

    print(f"Generating real 2023 season preview data into {output_dir}...")
    ratings_by_week = {}
    for week in PREVIEW_WEEKS:
        print(f"Computing week {week} ratings...")
        ratings_by_week[week] = compute_and_write_ratings(PREVIEW_SEASON, week, output_dir)

    print(f"Computing week {DIVERGENCE_WEEK} divergence (real closing lines)...")
    compute_and_write_divergence(PREVIEW_SEASON, DIVERGENCE_WEEK, ratings_by_week[DIVERGENCE_WEEK], output_dir)

    print("\nDone. This is real 2023 season output for preview purposes — not live 2026 data.")
