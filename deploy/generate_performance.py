"""
Grades every flagged play against final scores and closing lines, and
writes data/performance.json -- the file the dashboard's Track record
tab, KPI scorecard, and confidence-meter tier pip all read.

Grading rules (deliberately mirrors frontend/src/staking.js exactly --
if these drift apart, the site advertises one system and grades
another):
  Play = |gap| >= 4.0 points, staked 1u
  Lean = 2.5 <= |gap| < 4.0, staked 0.5u
  Spread gaps and total gaps graded against their own markets;
  the bet is taken at the EARLIEST snapshot's market number for the
  week (you bet when the model first flags, not at the close).

CLV per play: your number vs nflverse's closing spread_line, signed so
positive = beat the close. Units assume -110 both ways -- the odds
snapshots don't yet retain per-side prices (a known refinement).

Preseason projections (week 0) are informational only and are never
graded. A week is graded only once nflverse shows final scores for it.
"""

import sys
import os
import glob
import json
import re
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

PLAY_GAP = 4.0
LEAN_GAP = 2.5
WIN_PAYOUT = 100 / 110  # -110 both sides
STAKES = {"play": 1.0, "lean": 0.5}

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def load_week_snapshots(data_dir, season):
    """Earliest snapshot per real week (week >= 1), keyed by week."""
    pattern = os.path.join(data_dir, "divergence", f"{season}-week-*.json")
    by_week = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf"{season}-week-(\d+)", os.path.basename(path))
        if not m:
            continue
        week = int(m.group(1))
        if week < 1 or week in by_week:
            continue  # sorted() => first file per week is the earliest
        with open(path) as f:
            by_week[week] = json.load(f)
    return by_week


def grade_divergence(d, week, results_by_game):
    """Returns a list of graded play dicts (0, 1, or 2 -- spread and/or total)."""
    key = (week, d["home_team"], d["away_team"])
    game = results_by_game.get(key)
    if game is None:
        return []

    graded = []
    for market, gap in (("spread", d.get("spread_gap")), ("total", d.get("total_gap"))):
        if gap is None:
            continue
        edge = abs(gap)
        if edge >= PLAY_GAP:
            tier = "play"
        elif edge >= LEAN_GAP:
            tier = "lean"
        else:
            continue

        actual_margin = game["home_score"] - game["away_score"]
        actual_total = game["home_score"] + game["away_score"]

        if market == "spread":
            line = d["market_spread"]
            picked_home = gap > 0
            if actual_margin == line:
                result = "push"
            else:
                result = "win" if (actual_margin > line) == picked_home else "loss"
            close = game.get("spread_line")
            # CLV from the picked side's perspective: positive = beat the close.
            clv = None if close is None or pd.isna(close) else (close - line if picked_home else line - close)
            label = f"{d['home_team'] if picked_home else d['away_team']} {'-' if (picked_home and line > 0) or (not picked_home and line < 0) else '+'}{abs(line):g}"
        else:
            line = d["market_total"]
            over = gap > 0
            if actual_total == line:
                result = "push"
            else:
                result = "win" if (actual_total > line) == over else "loss"
            close = game.get("total_line")
            clv = None if close is None or pd.isna(close) else (close - line if over else line - close)
            label = f"{d['away_team']}/{d['home_team']} {'over' if over else 'under'} {line:g}"

        stake = STAKES[tier]
        units = stake * WIN_PAYOUT if result == "win" else -stake if result == "loss" else 0.0
        graded.append({
            "week": week, "market": market, "tier": tier, "label": label,
            "line": line, "close": None if clv is None else float(close),
            "result": result, "units": round(units, 3),
            "clv": None if clv is None else round(float(clv), 2),
            "edge": round(edge, 2),
            "model_margin": (d["market_spread"] + d["spread_gap"]) if market == "spread" else None,
            "actual_margin": int(actual_margin) if market == "spread" else None,
            "market_spread": d["market_spread"] if market == "spread" else None,
        })
    return graded


def summarize(plays):
    graded = [p for p in plays if p["result"] in ("win", "loss")]
    wins = sum(1 for p in graded if p["result"] == "win")
    losses = len(graded) - wins
    units = sum(p["units"] for p in plays)
    staked = sum(STAKES[p["tier"]] for p in plays if p["result"] != "push")
    clvs = [p["clv"] for p in plays if p["clv"] is not None]

    spreads = [p for p in plays if p["market"] == "spread" and p["model_margin"] is not None and p["actual_margin"] is not None]
    model_mae = (sum(abs(p["model_margin"] - p["actual_margin"]) for p in spreads) / len(spreads)) if spreads else None
    market_mae = (sum(abs(p["market_spread"] - p["actual_margin"]) for p in spreads) / len(spreads)) if spreads else None

    def tier_block(tier):
        t = [p for p in graded if p["tier"] == tier]
        w = sum(1 for p in t if p["result"] == "win")
        return {"n_plays": len(t), "ats_pct": round(w / len(t), 4) if t else None}

    return {
        "ats_wins": wins,
        "ats_losses": losses,
        "pushes": sum(1 for p in plays if p["result"] == "push"),
        "units": round(units, 2),
        "roi": round(units / staked, 4) if staked > 0 else None,
        "avg_clv": round(sum(clvs) / len(clvs), 2) if clvs else None,
        "n_clv_bets": len(clvs),
        "model_mae": round(model_mae, 2) if model_mae is not None else None,
        "market_mae": round(market_mae, 2) if market_mae is not None else None,
        "tier_stats": tier_block("play"),
        "tier_stats_by_tier": {"play": tier_block("play"), "lean": tier_block("lean")},
    }


def generate(data_dir, season, games_df=None):
    if games_df is None:
        games_df = pd.read_csv(GAMES_URL)
    season_games = games_df[(games_df["season"] == season)].dropna(subset=["home_score", "away_score"])
    results_by_game = {
        (int(g["week"]), g["home_team"], g["away_team"]): g
        for _, g in season_games.iterrows()
    }

    snapshots = load_week_snapshots(data_dir, season)
    plays = []
    for week, snap in sorted(snapshots.items()):
        for d in snap.get("divergences", []):
            plays.extend(grade_divergence(d, week, results_by_game))

    perf = {
        "season": season,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "graded_weeks": sorted({p["week"] for p in plays}),
        **summarize(plays),
        "plays": plays,
    }

    out_path = os.path.join(data_dir, "performance.json")
    with open(out_path, "w") as f:
        json.dump(perf, f, indent=2)
    n = perf["ats_wins"] + perf["ats_losses"]
    print(f"[generate_performance] wrote {out_path}: {perf['ats_wins']}-{perf['ats_losses']} "
          f"({n} graded plays), {perf['units']:+.2f}u, avg CLV {perf['avg_clv']}")
    return out_path


if __name__ == "__main__":
    season = int(os.environ.get("SEASON", datetime.now().year))
    data_dir = os.environ.get("REPO_DATA_PATH", "./data")
    generate(data_dir, season)
