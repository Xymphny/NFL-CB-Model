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
from deploy.validate import ValidationError
from deploy.notify import report_success, report_failure
from deploy.git_utils import git_commit_and_push

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

    return resp.json()


def compute_divergences(odds_data: list, model_ratings: dict, model_predictions: dict) -> list:
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
        h2h_market = next((m for m in bookmakers[0].get("markets", []) if m["key"] == "h2h"), None)
        if not h2h_market:
            continue

        outcomes = {o["name"]: o["price"] for o in h2h_market["outcomes"]}
        if home_team not in outcomes or away_team not in outcomes:
            continue

        home_raw = american_to_implied_prob(outcomes[home_team])
        away_raw = american_to_implied_prob(outcomes[away_team])
        home_fair, away_fair = devig_two_way(home_raw, away_raw)

        pred = model_predictions[home_team]
        divergence = flag_divergence(
            model_spread=pred["spread"], model_total=pred["total"], model_win_prob_home=pred["win_prob_home"],
            market_spread=pred.get("market_spread", 0), market_total=pred.get("market_total", 0),
            market_odds_home=home_fair, market_odds_away=away_fair,
        )

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "market_win_prob_home_fair": home_fair,
            **divergence,
        })

    return results


def main():
    try:
        odds_data = fetch_current_odds()
        # model_predictions would come from the calibrated points-prediction
        # layer for this week's games — not wired up in this pass, since
        # that requires a live current-week rating, not the historical
        # calibration exercise calibrate_points_model.py already does.
        model_predictions = {}
        divergences = compute_divergences(odds_data, model_ratings={}, model_predictions=model_predictions)

        os.makedirs(REPO_DATA_PATH, exist_ok=True)
        output_file = os.path.join(REPO_DATA_PATH, "divergence.json")
        with open(output_file, "w") as f:
            json.dump({
                "computed_at": datetime.now(timezone.utc).isoformat(),
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
