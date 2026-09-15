"""
MLB odds watch: The Odds API (baseball_mlb, h2h + totals) joined to
the day's probables -> data/mlb_divergence/{date}-{stamp}.json.

Board semantics differ from football on purpose: the flag statistic
is EV EDGE (model win prob minus de-vigged market prob), and every
row stores the probables it was priced against -- a later scratch
means the stored line describes a different game, so the frontend and
grader treat probable-mismatch rows as void, the moneyline analog of
the kickoff freeze. Verdict thresholds ship as OBSERVATION-ONLY until
model/mlb_backtest.py produces the evidence table (same Lean-cap
philosophy as CFB weeks 1-4: no Play badge without a backtest row
justifying it).

First-pitch freeze reuses the football machinery verbatim:
split_started_and_carry + load_prior_snapshots from odds_watch_job.
"""

import json
import os
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.odds_watch_job import split_started_and_carry, load_prior_snapshots, _json_sanitize
from deploy.mlb_probables import fetch_probables, apply_overrides
from model.mlb_backtest import devig_home_prob

ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
REPO_DATA_PATH = os.environ.get("REPO_DATA_PATH", "./data")


def parse_game_markets(game):
    """Best home/away ML across books + consensus total. Returns None
    if no moneyline present."""
    best = {"home": None, "away": None}
    totals = []
    full_h, full_a = game.get("home_team"), game.get("away_team")
    for bk in game.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] == "h2h":
                for oc in mk.get("outcomes", []):
                    side = "home" if oc["name"] == full_h else ("away" if oc["name"] == full_a else None)
                    if side and oc.get("price") is not None:
                        if best[side] is None or oc["price"] > best[side]["price"]:
                            best[side] = {"price": oc["price"], "book": bk.get("title")}
            elif mk["key"] == "totals":
                for oc in mk.get("outcomes", []):
                    if oc.get("point") is not None:
                        totals.append(oc["point"])
    if not best["home"] or not best["away"]:
        return None
    return {
        "home_ml": best["home"]["price"], "home_ml_book": best["home"]["book"],
        "away_ml": best["away"]["price"], "away_ml_book": best["away"]["book"],
        "market_total": sorted(totals)[len(totals) // 2] if totals else None,
    }


def build_snapshot(api_key, model_probs=None):
    """model_probs: {(home_abbr, away_abbr): p_home} from the live
    model runner (separate job); absent -> odds-only observation rows."""
    resp = requests.get(ODDS_URL, params={"apiKey": api_key, "regions": "us",
                                          "markets": "h2h,totals", "oddsFormat": "american"}, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    odds_data = resp.json()
    print(f"[mlb_odds_watch] {len(odds_data)} games with odds; credits remaining {remaining}")

    probables = {("%s" % g["home_name"], "%s" % g["away_name"]): g for g in apply_overrides(fetch_probables())}

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = now_iso[:10]
    prior = load_prior_snapshots(REPO_DATA_PATH, "mlb_divergence", 0, 0)  # date-keyed family; season/week unused
    # date-keyed prior loading: fall back to glob by date prefix
    import glob as _g
    prior = []
    for path in sorted(_g.glob(os.path.join(REPO_DATA_PATH, "mlb_divergence", f"{today}-*.json"))):
        with open(path) as f:
            snap = json.load(f)
        prior.append((snap.get("computed_at"), snap.get("divergences", [])))
    odds_data, carried = split_started_and_carry(odds_data, prior, now_iso)
    if carried:
        print(f"[mlb_odds_watch] {len(carried)} started game(s): lines frozen at first pitch")

    rows = []
    for game in odds_data:
        parsed = parse_game_markets(game)
        if not parsed:
            continue
        pb = probables.get((game.get("home_team"), game.get("away_team")), {})
        market_p = devig_home_prob(parsed["home_ml"], parsed["away_ml"])
        row = {
            "home_team": pb.get("home_team") or game.get("home_team"),
            "away_team": pb.get("away_team") or game.get("away_team"),
            "home_name": game.get("home_team"), "away_name": game.get("away_team"),
            "kickoff": game.get("commence_time"), "line_status": "open",
            "home_probable": pb.get("home_probable"), "away_probable": pb.get("away_probable"),
            "qb_alert": {"home": pb.get("home_alert"), "away": pb.get("away_alert")},
            **parsed,
            "market_home_prob": round(market_p, 4),
        }
        key = (row["home_team"], row["away_team"])
        if model_probs and key in model_probs:
            row["model_home_prob"] = round(model_probs[key], 4)
            row["ev_gap"] = round(model_probs[key] - market_p, 4)
        rows.append(row)

    out = _json_sanitize({
        "computed_at": now_iso, "date": today,
        "note": ("MLB board is OBSERVATION-ONLY until the archive backtest publishes its "
                 "threshold table. Lines are quoted conditional on the listed probables shown; "
                 "a scratched starter voids the stored line for grading."),
        "divergences": rows + carried,
    })
    out_dir = os.path.join(REPO_DATA_PATH, "mlb_divergence")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{today}-{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(f"[mlb_odds_watch] wrote {path}: {len(rows)} open + {len(carried)} frozen")
    return path


def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("[mlb_odds_watch] ODDS_API_KEY not set; exiting")
        return
    path = build_snapshot(api_key)
    if os.environ.get("GIT_REPO_URL"):
        from deploy.git_utils import git_commit_and_push
        git_commit_and_push(path, commit_message=f"MLB odds snapshot {os.path.basename(path)}")


if __name__ == "__main__":
    main()
