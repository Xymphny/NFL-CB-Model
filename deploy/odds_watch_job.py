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
import pandas as pd

from model.market_comparison import american_to_implied_prob, devig_two_way, flag_divergence
from model.prediction import load_current_ratings, build_week_predictions, find_latest_ratings_snapshot
from model.layer2_ngs import compute_team_ngs_features
from model.elo_rating import compute_elo_walk_forward
from model.weather import fetch_forecast
from model.version import METHODOLOGY_VERSION
from deploy.validate import ValidationError
from deploy.notify import report_success, report_failure, send_webhook_alert
from deploy.git_utils import git_commit_and_push
from ingest.nfl_schedules import is_game_day, load_schedules, get_next_upcoming_week

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


# The Odds API names NFL teams in full ("Detroit Lions"); every other
# part of this system -- ratings, schedules, the performance grader,
# the frontend -- speaks nflverse abbreviations ("DET"). This map is
# the bridge. Its absence was a latent day-one bug: the first live
# forced run matched 0 of 272 games because "Detroit Lions" was looked
# up in a dict keyed by "DET". Discovered 2026-09-04, pre-season,
# exactly what forced runs are for.
ODDS_TEAM_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


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

        # Full names from the API -> nflverse abbreviations for
        # everything downstream (see ODDS_TEAM_TO_ABBR note above).
        full_home, full_away = home_team, away_team
        home_team = ODDS_TEAM_TO_ABBR.get(home_team, home_team)
        away_team = ODDS_TEAM_TO_ABBR.get(away_team, away_team)
        if home_team not in model_predictions:
            continue
        expected_away = model_predictions[home_team].get("away_team")
        if expected_away is not None and expected_away != away_team:
            continue  # same home team, different (future-week) matchup

        # ALL bookmakers now parsed (line shopping upgrade). Divergence
        # math runs against the CONSENSUS line (median across books --
        # sturdier than any single book), while per-book best prices are
        # kept so the dashboard can show where each side is cheapest.
        # Half a point of shopping is worth more than most model
        # improvements -- this is the feature that captures it.
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        h2h_prices, spread_rows, total_rows = [], [], []
        for book in bookmakers:
            book_name = book.get("title") or book.get("key")
            markets = {m["key"]: m for m in book.get("markets", [])}

            h2h = markets.get("h2h")
            if h2h:
                outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
                if full_home in outcomes and full_away in outcomes:
                    h2h_prices.append({"book": book_name, "home": outcomes[full_home], "away": outcomes[full_away]})

            spreads = markets.get("spreads")
            if spreads:
                for o in spreads["outcomes"]:
                    if o.get("point") is None:
                        continue
                    spread_rows.append({
                        "book": book_name, "side": "home" if o["name"] == full_home else "away",
                        # API convention: negative = that side favored. Home
                        # rows flipped to this project's positive-=-home-favored
                        # convention; away rows keep the side's own number.
                        "point": -o["point"] if o["name"] == full_home else o["point"],
                        "price": o.get("price"),
                    })

            totals = markets.get("totals")
            if totals:
                for o in totals["outcomes"]:
                    if o.get("point") is None:
                        continue
                    total_rows.append({"book": book_name, "side": o["name"].lower(), "point": o["point"], "price": o.get("price")})

        if not h2h_prices:
            continue
        # De-vig the consensus book-by-book, then take the median fair prob.
        fair_probs = []
        for hp in h2h_prices:
            hr = american_to_implied_prob(hp["home"])
            ar = american_to_implied_prob(hp["away"])
            hf, _ = devig_two_way(hr, ar)
            fair_probs.append(hf)
        home_fair = float(pd.Series(fair_probs).median())
        away_fair = 1 - home_fair

        home_spreads = [r for r in spread_rows if r["side"] == "home"]
        market_spread = float(pd.Series([r["point"] for r in home_spreads]).median()) if home_spreads else None
        if market_spread == 0:
            market_spread = 0.0  # normalize -0.0 from medians of pick-em lines
        market_total = float(pd.Series([r["point"] for r in total_rows]).median()) if total_rows else None

        def _best(rows, better):
            """Best available price: the friendliest point first, then
            the cheapest juice at that point."""
            if not rows:
                return None
            best_point = better(r["point"] for r in rows)
            at_point = [r for r in rows if r["point"] == best_point and r["price"] is not None]
            if not at_point:
                return {"point": best_point, "price": None, "book": None}
            top = max(at_point, key=lambda r: r["price"])
            return {"point": top["point"], "price": top["price"], "book": top["book"]}

        best_prices = {
            # Home side wants to lay the FEWEST points (min of our
            # positive-=-favored convention); away side wants to GET the most.
            "home_spread": _best(home_spreads, min),
            "away_spread": _best([r for r in spread_rows if r["side"] == "away"], max),
            "over": _best([r for r in total_rows if r["side"] == "over"], min),
            "under": _best([r for r in total_rows if r["side"] == "under"], max),
            "home_ml": max(h2h_prices, key=lambda r: r["home"])["home"] if h2h_prices else None,
            "away_ml": max(h2h_prices, key=lambda r: r["away"])["away"] if h2h_prices else None,
            "n_books": len(bookmakers),
        }

        # Sharp-book anchoring: when a sharp book is in the feed, its
        # number is the best single estimate of the true line -- square
        # books sitting a half point or more off it are stale, and
        # betting the side the sharp number implies at the stale shop
        # is the most reliable +EV pattern in the market, model-free.
        # Activates automatically iff the Odds API plan returns one of
        # these books; costs nothing when absent.
        SHARP_BOOKS = ("Pinnacle", "Circa Sports", "BetOnline.ag")
        sharp_anchor = None
        sharp_rows = [r for r in home_spreads if r["book"] in SHARP_BOOKS]
        if sharp_rows:
            sharp_book = sharp_rows[0]["book"]
            sharp_line = sharp_rows[0]["point"] + 0.0 if sharp_rows[0]["point"] != 0 else 0.0
            stale = []
            for r in home_spreads:
                if r["book"] in SHARP_BOOKS:
                    continue
                gap = r["point"] - sharp_line
                if abs(gap) >= 0.5:
                    stale.append({
                        "book": r["book"], "point": r["point"], "price": r.get("price"),
                        # gap > 0: this book asks the home side to lay
                        # MORE than sharp -- home side stale-cheap side
                        # is at the sharp-favoring shop; value side is
                        # the one getting the extra points.
                        "value_side": "away" if gap > 0 else "home",
                        "gap_pts": round(gap, 1),
                    })
            sharp_anchor = {"book": sharp_book, "spread": sharp_line, "stale_books": stale}

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
            "home_name": full_home,
            "away_name": full_away,
            "kickoff": game.get("commence_time"),
            "market_win_prob_home_fair": home_fair,
            "market_spread": market_spread,
            "market_total": market_total,
            "best_prices": best_prices,
            "sharp_anchor": sharp_anchor,
            **divergence,
        })

    return results


def _json_sanitize(obj):
    """Recursively replace NaN/inf (Python json writes them; JavaScript
    JSON.parse rejects them) with None. Paired with allow_nan=False at
    the dump so any future leak crashes the job loudly instead of
    silently breaking the site."""
    import math
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    season = int(os.environ.get("SEASON", "2026"))

    # Real production measurement (6 credits/call) showed the fixed
    # every-4-hours/every-day schedule would exceed The Odds API's free
    # tier in ~2 weeks instead of a month. Gate on actual game days
    # rather than changing the cron schedule itself — cheaper to skip
    # the API call in code than to fight cron syntax for "game days only".
    #
    # FORCE_ODDS_CHECK bypasses this gate for on-demand manual runs —
    # sportsbooks post lines days or weeks before kickoff, not just on
    # game day itself, so a manual "check what's posted right now" is a
    # legitimate need the automated schedule shouldn't be widened for
    # (that would blow the credit budget back out — see the schedule
    # comment above). Use this for a one-off check, not as a standing
    # override.
    force_run = os.environ.get("FORCE_ODDS_CHECK", "false").lower() == "true"

    if not force_run and not is_game_day(season):
        print(f"[odds_watch_job] not a game day, skipping API call to conserve credits "
              f"(set FORCE_ODDS_CHECK=true to check current lines on demand)")
        report_success("odds_watch_job", summary="skipped, not a game day")
        return

    if force_run:
        print("[odds_watch_job] FORCE_ODDS_CHECK set, checking current lines regardless of game day")

    try:
        odds_data = fetch_current_odds()

        # Find the most recent ratings snapshot weekly_job.py already
        # committed (now stored as immutable per-week files, not a single
        # overwritten ratings.json), and build this week's predictions
        # from them.
        ratings_path = find_latest_ratings_snapshot(REPO_DATA_PATH, season)
        if ratings_path is None:
            raise ValidationError(
                f"No ratings snapshot found for {season} in {REPO_DATA_PATH}/ratings/ — "
                f"weekly_job.py needs to have run at least once before odds_watch_job.py "
                f"can predict anything"
            )
        print(f"[odds_watch_job] using ratings snapshot: {ratings_path}")
        ratings = load_current_ratings(ratings_path)
        # Week of the ratings snapshot itself (filename is {season}-week-NN),
        # used to attach an honest preseason caution to the board when
        # predictions are running on week-00 priors.
        import re as _re
        _m = _re.search(r"week-(\d+)", os.path.basename(ratings_path))
        ratings_snapshot_week = int(_m.group(1)) if _m else None

        current_week = get_next_upcoming_week(season)
        sched = load_schedules(seasons=[season])
        upcoming_games = sched[sched["week"] == current_week]

        # Layer 2: real player-tracking features (Section 4/model/layer2_ngs.py),
        # validated via walk-forward testing to meaningfully improve
        # prediction accuracy (straight-up 58.22% -> 64.04% in
        # backtesting — see model/walk_forward_layer2_test.py). Falls
        # back gracefully to unenhanced predictions if NGS data can't
        # be fetched (e.g. very early in a season before enough data exists).
        try:
            ngs_features = compute_team_ngs_features(season, through_week=current_week)
        except Exception as e:
            print(f"[odds_watch_job] Layer 2 NGS features unavailable ({e}), predicting without them")
            ngs_features = None

        # Real Elo ensemble (model/elo_rating.py) -- validated to give
        # a substantial additional accuracy improvement on top of Layer
        # 2 (straight-up 62.72% -> 65.55% in backtesting on a large,
        # held-out 2022-2023 test set -- see model/test_full_ensemble.py).
        # Computed from real historical schedule data (10 real prior
        # seasons plus the current season's completed games) -- cheap,
        # since Elo only needs final scores, not full play-by-play.
        try:
            historical_schedule = load_schedules(seasons=list(range(season - 10, season + 1)))
            _, elo_ratings = compute_elo_walk_forward(historical_schedule)
        except Exception as e:
            print(f"[odds_watch_job] Elo ratings unavailable ({e}), predicting without them")
            elo_ratings = None

        # Real wind forecast (model/weather.py) -- built and its parsing
        # logic validated against api.weather.gov's real documented
        # format, but never actually wired into a prediction until now
        # (the same "built but invisible" gap found repeatedly this
        # session). Schedule data's own "wind" column is NaN for
        # upcoming games (weather isn't knowable that far ahead at
        # publish time) -- fetches a real near-kickoff forecast instead of
        # silently defaulting to 0 for every outdoor game.
        upcoming_games = upcoming_games.copy()
        for idx, game in upcoming_games.iterrows():
            forecast = fetch_forecast(game["home_team"])
            upcoming_games.loc[idx, "wind"] = forecast.get("wind", 0.0)

        model_predictions = build_week_predictions(ratings, upcoming_games, ngs_features=ngs_features, elo_ratings=elo_ratings)
        # Tag each prediction with its intended opponent. Without this,
        # a home-team-keyed lookup matches EVERY future home game of
        # that team in the API's multi-week feed -- the first live run
        # compared week-1 predictions against 135 games across the
        # whole season (discovered 2026-09-04, second forced-run catch).
        for _, gm in upcoming_games.iterrows():
            if gm["home_team"] in model_predictions:
                model_predictions[gm["home_team"]]["away_team"] = gm["away_team"]
        print(f"[odds_watch_job] built predictions for {len(model_predictions)} of "
              f"{len(upcoming_games)} week {current_week} games")

        # Preseason scale alignment -- same cure as the CFB board's
        # all-underdogs artifact, same disease: week-00 ratings compress
        # the rating-driven share of home margins, and the margin
        # equation's collinear home_field/intercept pair nets a -1.13pt
        # home edge (see the note beside MARGIN_COEFFICIENTS), so the
        # preseason board leaned away on 15 of 16 games. Aligning model
        # spreads to the slate's market spreads (slope+intercept)
        # removes the systematic component and leaves cross-sectional
        # disagreement only. STRICTLY week-00: the in-season path is
        # backtest-validated at native scale and stays untouched.
        if ratings_snapshot_week == 0 and len(model_predictions) >= 8:
            probe = compute_divergences(odds_data, model_predictions)
            spread_rows = [(d_p["market_spread"] + d_p["spread_gap"], d_p["market_spread"]) for d_p in probe]
            total_rows = [(d_p["market_total"] + d_p["total_gap"], d_p["market_total"])
                          for d_p in probe if d_p.get("total_gap") is not None and d_p.get("market_total") is not None]

            def _fit(pairs):
                xs = pd.Series([r[0] for r in pairs]); ys = pd.Series([r[1] for r in pairs])
                sl = ((xs - xs.mean()) * (ys - ys.mean())).sum() / max(((xs - xs.mean()) ** 2).sum(), 1e-9)
                return sl, ys.mean() - sl * xs.mean()

            def _robust_align(pairs, label):
                """Robust two-pass fit (drop the 2 largest residuals and
                refit, so a genuine outlier survives de-biasing at full
                size) with a degeneracy guard: a slope outside (0.1, 5)
                means the fit is pathological -- skip rather than apply
                nonsense."""
                if len(pairs) < 8:
                    return None
                sl, ic = _fit(pairs)
                trimmed = sorted(pairs, key=lambda r: abs((sl * r[0] + ic) - r[1]))[:-2]
                sl, ic = _fit(trimmed)
                if not (0.1 < sl < 5):
                    print(f"[odds_watch] {label} alignment skipped: degenerate slope {sl:.2f}")
                    return None
                print(f"[odds_watch] preseason {label} alignment: x' = {sl:.2f}*x {ic:+.2f} over {len(pairs)} games")
                return sl, ic

            s_align = _robust_align(spread_rows, "spread")
            t_align = _robust_align(total_rows, "total")
            if s_align or t_align:
                from model.prediction import margin_to_win_probability
                for pred in model_predictions.values():
                    if s_align:
                        pred["spread"] = s_align[0] * pred["spread"] + s_align[1]
                        # Keep win prob consistent with the aligned line.
                        pred["win_prob_home"] = margin_to_win_probability(pred["spread"])
                    if t_align and pred.get("total") is not None:
                        pred["total"] = t_align[0] * pred["total"] + t_align[1]

        divergences = compute_divergences(odds_data, model_predictions)

        # QB-status annotation (free nflverse data; annotation-only by
        # design -- see deploy/qb_status.py for the held-out evidence
        # against auto-demotion). Soft-fail: never blocks the pipeline.
        try:
            from deploy.qb_status import get_qb_alerts
            qb_alerts = get_qb_alerts(season, current_week)
            for d in divergences:
                d["qb_alert"] = {
                    "home": qb_alerts.get(d["home_team"]),
                    "away": qb_alerts.get(d["away_team"]),
                }
            from deploy.qb_status import get_projected_starters
            qb1_map = get_projected_starters(season)
        except Exception as qb_err:
            qb1_map = {}
            print(f"[odds_watch] qb alerts skipped: {qb_err}")

        # ESPN FPI cross-reference: an independent public model's win
        # probability shown as a third opinion. Not edge (public =
        # priced in) -- transparency. Soft-fail.
        try:
            from deploy.espn_extras import fetch_fpi_map
            fpi_map = fetch_fpi_map()
            for d in divergences:
                d["fpi_home_prob"] = fpi_map.get((d["home_team"], d["away_team"]))
            if fpi_map:
                print(f"[odds_watch] FPI attached for {sum(1 for d in divergences if d.get('fpi_home_prob') is not None)} games")
        except Exception as fpi_err:
            print(f"[odds_watch] FPI skipped: {fpi_err}")

        # Per-game context: full injury reports + kickoff weather.
        # Transparency, not edge (the residual experiment showed both
        # are priced in) -- soft-fail like everything context-shaped.
        try:
            from deploy.game_context import attach_context
            attach_context(divergences, season, current_week)
        except Exception as ctx_err:
            print(f"[odds_watch] game context skipped: {ctx_err}")

        os.makedirs(REPO_DATA_PATH, exist_ok=True)
        divergence_dir = os.path.join(REPO_DATA_PATH, "divergence")
        os.makedirs(divergence_dir, exist_ok=True)

        # Alert on NEWLY flagged plays only -- diffed against the most
        # recent prior snapshot for this same week, so the webhook fires
        # when a play first appears (or a line moves enough to flag a
        # game that wasn't), not on every 4-hour re-check of the same
        # board. This is the "play posted" alert a subscriber Discord
        # actually wants; a webhook that repeats the full board six
        # times a day trains everyone to mute it.
        import glob as _glob
        prior_files = sorted(_glob.glob(os.path.join(divergence_dir, f"{season}-week-{current_week:02d}-*.json")))
        previously_flagged = set()
        if prior_files:
            with open(prior_files[-1]) as f:
                for d in json.load(f).get("divergences", []):
                    if d.get("spread_flagged") or d.get("total_flagged"):
                        previously_flagged.add((d["home_team"], d["away_team"]))

        new_plays = []
        for d in divergences:
            if not (d.get("spread_flagged") or d.get("total_flagged")):
                continue
            if (d["home_team"], d["away_team"]) in previously_flagged:
                continue
            bp = (d.get("best_prices") or {})
            if d.get("spread_flagged"):
                side = d["home_team"] if d["spread_gap"] > 0 else d["away_team"]
                line = -d["market_spread"] if d["spread_gap"] > 0 else d["market_spread"]
                best = bp.get("home_spread") if d["spread_gap"] > 0 else bp.get("away_spread")
                desc = f"{side} {line:+.1f}, gap {abs(d['spread_gap']):.1f} pts"
            else:  # total-only flag: describe the total, not a spread
                over = d["total_gap"] > 0
                best = bp.get("over") if over else bp.get("under")
                desc = (f"{d['away_team']}/{d['home_team']} {'over' if over else 'under'} "
                        f"{d['market_total']:g}, gap {abs(d['total_gap']):.1f} pts")
            price_note = f" (best {best['price']:+d} at {best['book']})" if best and best.get("price") is not None else ""
            new_plays.append(desc + price_note)
        if new_plays:
            send_webhook_alert(
                "New board plays -- " + "; ".join(new_plays[:6])
                + (f" (+{len(new_plays) - 6} more)" if len(new_plays) > 6 else "")
            )

        # Timestamped snapshot rather than overwriting one divergence.json
        # — same fix as weekly_job.py's ratings snapshots, for the same
        # reason (repeated overwrites of one file is what causes git
        # friction; a new file per check is a pure addition). This has an
        # even stronger case here: odds-watch runs multiple times per game
        # day, so keeping every snapshot is exactly what Section 9.3's
        # closing-line-value tracking needs — the full line movement from
        # open to close, not just the latest number.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = os.path.join(divergence_dir, f"{season}-week-{current_week:02d}-{timestamp}.json")
        with open(output_file, "w") as f:
            preseason_note = None
            if ratings_snapshot_week == 0:
                preseason_note = (
                    "Early-week board: model numbers come from preseason ratings, "
                    "which are the least reliable of the season -- large gaps here "
                    "are more likely model error than market error. Treat Week 1 "
                    "as observation, not opportunity.")
            json.dump(_json_sanitize({
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "season": season,
                "week": current_week,
                "methodology_version": METHODOLOGY_VERSION,
                "note": preseason_note,
                "qb1_map": qb1_map,
                "divergences": divergences,
            }), f, indent=2, allow_nan=False)

        # This was missing entirely in the original version — the job
        # wrote divergence.json locally but never committed it, so
        # nothing ever reached the repo or triggered the static site's
        # auto-deploy. Same shared git logic as weekly_job.py, already
        # hardened against the missing-origin and detached-HEAD failures
        # found in production there.
        if GIT_REPO_URL:
            git_commit_and_push(output_file, commit_message=f"Update divergence: {season} week {current_week}, {len(divergences)} games")
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
