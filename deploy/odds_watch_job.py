"""
Odds/injury watch cron job — Section 9.1's lighter job, 3-4 refreshes
per game day.

UNTESTED beyond structure — needs a real Odds API key. The de-vig and
divergence math this calls (model/market_comparison.py) IS tested; what
isn't tested here is the live HTTP fetch and the actual credit cost per
call, which Section 9.3 already flagged as needing a real test call
before finalizing the exact schedule.
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from model.market_comparison import american_to_implied_prob, devig_two_way, flag_divergence
from model.prediction import load_current_ratings, build_week_predictions
from deploy.validate import ValidationError
from deploy.notify import report_success, report_failure
from deploy.git_utils import git_commit_and_push
from ingest.nfl_schedules import is_game_day, load_schedules, get_current_week

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
REPO_DATA_PATH = os.environ.get("REPO_DATA_PATH", "./data")
GIT_REPO_URL = os.environ.get("GIT_REPO_URL")


def fetch_current_odds(sport_key: str = "americanfootball_nfl") -> list:
    """
    Pulls the full current slate in one call (Section 9.1's discovery
    that one call covers every scheduled game, not one call per game).
    """
    if not ODDS_API_KEY:
        raise ValidationError("ODDS_API_KEY not set")

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"[odds_watch] API credits used this call: {used}, remaining this period: {remaining}")

    data = resp.json()
    print(f"[odds_watch] {len(data)} games returned with posted odds "
          f"(sportsbooks open lines gradually as kickoff approaches, so this "
          f"is normally far fewer than the full season's schedule)")
    return data


def compute_divergences(odds_data: list, model_predictions: dict) -> list:
    """
    For each game, de-vig the market's line and compare to the model's
    own prediction (Section 9.3's kept-separate design — model_predictions
    must come from the points-prediction layer, not from this function).
    """
    results = []
    for game in odds_data:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        if home_team not in model_predictions:
            continue

        # Use the first available bookmaker as a simple starting point —
        # averaging across books (a proper consensus line) is a
        # reasonable refinement not done in this pass.
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        markets = {m["key"]: m for m in bookmakers[0].get("markets", [])}

        h2h_market = markets.get("h2h")
        if not h2h_market:
            continue
        outcomes = {o["name"]: o["price"] for o in h2h_market["outcomes"]}
        if home_team not in outcomes or away_team not in outcomes:
            continue

        home_raw = american_to_implied_prob(outcomes[home_team])
        away_raw = american_to_implied_prob(outcomes[away_team])
        home_fair, away_fair = devig_two_way(home_raw, away_raw)

        # Fixed bug: market_spread/market_total previously came from the
        # model's own prediction dict (a placeholder oversight) instead
        # of the actual spreads/totals markets — parse them for real.
        spreads_market = markets.get("spreads")
        market_spread = None
        if spreads_market:
            home_spread_outcome = next(
                (o for o in spreads_market["outcomes"] if o["name"] == home_team), None
            )
            if home_spread_outcome:
                market_spread = -home_spread_outcome["point"]  # API convention: negative = favored;
                # flipped to match this project's "positive = home favored" convention (Section 11.4)

        totals_market = markets.get("totals")
        market_total = totals_market["outcomes"][0]["point"] if totals_market and totals_market.get("outcomes") else None

        if market_spread is None or market_total is None:
            continue  # book hasn't posted a full line yet — skip rather than compare against a partial one

        pred = model_predictions[home_team]
        divergence = flag_divergence(
            model_spread=pred["spread"], model_total=pred["total"], model_win_prob_home=pred["win_prob_home"],
            market_spread=market_spread, market_total=market_total,
            market_odds_home=home_fair, market_odds_away=away_fair,
        )

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "market_win_prob_home_fair": home_fair,
            "market_spread": market_spread,
            "market_total": market_total,
            **divergence,
        })

    return results


def main():
    season = int(os.environ.get("SEASON", "2026"))

    # Real production measurement (6 credits/call) showed the fixed
    # every-4-hours/every-day schedule would exceed The Odds API's free
    # tier in ~2 weeks instead of a month. Gate on actual game days
    # rather than changing the cron schedule itself — cheaper to skip
    # the API call in code than to fight cron syntax for "game days only".
    if not is_game_day(season):
        print(f"[odds_watch_job] not a game day, skipping API call to conserve credits")
        report_success("odds_watch_job", summary="skipped, not a game day")
        return

    try:
        odds_data = fetch_current_odds()

        # Load the current ratings weekly_job.py already committed, and
        # build this week's predictions from them — the piece that was
        # previously an empty placeholder.
        ratings_path = os.path.join(REPO_DATA_PATH, "ratings.json")
        if not os.path.exists(ratings_path):
            raise ValidationError(
                f"No ratings.json found at {ratings_path} — weekly_job.py needs to "
                f"have run at least once before odds_watch_job.py can predict anything"
            )
        ratings = load_current_ratings(ratings_path)

        current_week = get_current_week(season)
        sched = load_schedules(seasons=[season])
        upcoming_games = sched[sched["week"] == current_week]

        model_predictions = build_week_predictions(ratings, upcoming_games)
        print(f"[odds_watch_job] built predictions for {len(model_predictions)} of "
              f"{len(upcoming_games)} week {current_week} games")

        divergences = compute_divergences(odds_data, model_predictions)

        os.makedirs(REPO_DATA_PATH, exist_ok=True)
        output_file = os.path.join(REPO_DATA_PATH, "divergence.json")
        with open(output_file, "w") as f:
            json.dump({
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "season": season,
                "week": current_week,
                "divergences": divergences,
            }, f, indent=2)

        # This was missing entirely in the original version — the job
        # wrote divergence.json locally but never committed it, so
        # nothing ever reached the repo or triggered the static site's
        # auto-deploy. Same shared git logic as weekly_job.py, already
        # hardened against the missing-origin and detached-HEAD failures
        # found in production there.
        if GIT_REPO_URL:
            git_commit_and_push(output_file, commit_message=f"Update divergence: {len(divergences)} games")
        else:
            print("[odds_watch_job] GIT_REPO_URL not set, skipping commit/push (local-only run)")

        report_success("odds_watch_job", summary=f"{len(divergences)} games compared")

    except ValidationError as e:
        report_failure("odds_watch_job", error=f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        report_failure("odds_watch_job", error=f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
