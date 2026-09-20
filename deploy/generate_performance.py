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
    """Per real week (week >= 1), a synthetic snapshot whose rows are
    each game's FIRST APPEARANCE AS A NON-PASS.

    Previously this took only the week's earliest file, so a game that
    the board flagged later in the week never entered the record at
    all -- week 1 lost BUF@HOU (a lean that would have LOST) and
    GB@MIN (a 1u Play total that would have WON). The docstring's own
    rule is "you bet when the model first flags it", which was only
    true for games flagged at the open. Entry is the first snapshot
    where the row grades non-pass, so the price recorded is the price
    that was actually showing when the claim was made.
    """
    pattern = os.path.join(data_dir, "divergence", f"{season}-week-*.json")
    by_week = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf"{season}-week-(\d+)", os.path.basename(path))
        if not m:
            continue
        week = int(m.group(1))
        if week < 1:
            continue
        try:
            with open(path) as f:
                snap = json.load(f)
        except Exception as e:                            # noqa: BLE001
            print(f"[performance] skipping unreadable snapshot {os.path.basename(path)}: {e}")
            continue
        slot = by_week.setdefault(week, {"_meta": snap, "_rows": {}, "_all": {}})
        for row in snap.get("divergences", []):
            gkey = (row.get("home_team"), row.get("away_team"))
            if gkey not in slot["_all"] and row.get("spread_gap") is not None:
                slot["_all"][gkey] = row                  # first PRICED appearance
            if gkey in slot["_rows"]:
                continue                                  # already entered earlier
            if _flags_non_pass(row):
                slot["_rows"][gkey] = row
    out, allrows = {}, {}
    for week, slot in by_week.items():
        snap = dict(slot["_meta"])
        snap["divergences"] = list(slot["_rows"].values())
        out[week] = snap
        full = dict(slot["_meta"])
        full["divergences"] = list(slot["_all"].values())
        allrows[week] = full
    return out, allrows


def _flags_non_pass(d):
    """True when this row would have shown a Play or Lean on the board."""
    sg, tg = d.get("spread_gap"), d.get("total_gap")
    se = abs(sg) if sg is not None else 0.0
    te = abs(tg) if tg is not None else 0.0
    return se >= LEAN_GAP or te >= LEAN_GAP + 1


def grade_divergence(d, week, results_by_game):
    """Returns a list of graded play dicts (0, 1, or 2 -- spread and/or total)."""
    key = (week, d["home_team"], d["away_team"])
    game = results_by_game.get(key)
    if game is None:
        return []

    # VERDICT SELECTION -- must mirror gradeGameInner in
    # frontend/src/staking.js exactly. Two drifts were found on
    # 2026-09-20 and are fixed here:
    #   1. totals were graded at PLAY_GAP/LEAN_GAP while the board
    #      required PLAY_GAP+1 / LEAN_GAP+1, so the grader booked
    #      claims the board never made (a fabricated week-1 ATL@PIT
    #      loss among them);
    #   2. both markets were graded per game while the board shows
    #      ONE verdict, spread taking priority.
    # Replaying both versions over the committed snapshots, 6 of 16
    # week-1 games and 5 of 16 week-2 games graded differently.
    spread_gap = d.get("spread_gap")
    total_gap = d.get("total_gap")
    spread_edge = abs(spread_gap) if spread_gap is not None else 0.0
    total_edge = abs(total_gap) if total_gap is not None else 0.0

    if spread_edge >= PLAY_GAP or total_edge >= PLAY_GAP + 1:
        tier = "play"
        market = "spread" if spread_edge >= PLAY_GAP else "total"
    elif spread_edge >= LEAN_GAP or total_edge >= LEAN_GAP + 1:
        tier = "lean"
        market = "spread" if spread_edge >= LEAN_GAP else "total"
    else:
        return []

    graded = []
    for market, gap in ((market, spread_gap if market == "spread" else total_gap),):
        if gap is None:
            continue
        edge = abs(gap)
        # Regime cap: the board showed these at Lean stakes (see
        # apply_regime_layer) -- grade what the board showed.
        if market == "spread" and d.get("tier_cap") == "lean":
            tier = "lean"

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


def calibration_stats(by_week, results_by_game):
    """Weekly-updated calibration slope over ALL priced games with
    finals (not just flags): regress actual home margins on the
    board's model margins, and on the market's, pooled season to
    date. Slope 1.0 = honest amplitude; below 1 = stated margins
    carry less signal per point than they claim. Added 2026-09-18
    after the week 1-2 diagnostic measured model 0.68 vs market 1.14
    (n=17) -- expected while NGS inputs are unpublished and the
    in-season blend is young. The number to watch: if the model
    slope still sits near 0.7 by ~week 6 WITH tracking data flowing,
    it graduates from "young season" to a real offseason finding."""
    pairs = []
    for week, snap in by_week.items():
        for d in snap.get("divergences", []):
            game = results_by_game.get((week, d.get("home_team"), d.get("away_team")))
            if game is None or d.get("spread_gap") is None or d.get("market_spread") is None:
                continue
            actual = game["home_score"] - game["away_score"]
            pairs.append((d["market_spread"] + d["spread_gap"], d["market_spread"], actual))
    if len(pairs) < 8:
        return None

    def slope(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        if var < 1e-9:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var

    model_p, market_p, actual = zip(*pairs)
    sm, sk = slope(model_p, actual), slope(market_p, actual)
    return {"slope_model": round(sm, 3) if sm is not None else None,
            "slope_market": round(sk, 3) if sk is not None else None,
            "n_games": len(pairs)}


def generate(data_dir, season, games_df=None):
    if games_df is None:
        games_df = pd.read_csv(GAMES_URL)
    season_games = games_df[(games_df["season"] == season)].dropna(subset=["home_score", "away_score"])
    results_by_game = {
        (int(g["week"]), g["home_team"], g["away_team"]): g
        for _, g in season_games.iterrows()
    }

    snapshots, all_priced = load_week_snapshots(data_dir, season)
    plays = []
    for week, snap in sorted(snapshots.items()):
        for d in snap.get("divergences", []):
            plays.extend(grade_divergence(d, week, results_by_game))

    perf = {
        "season": season,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "graded_weeks": sorted({p["week"] for p in plays}),
        **summarize(plays),
        # Calibration measures AMPLITUDE over every priced game, not
        # only flagged ones -- narrowing it to flags would make the
        # instrument report on its own selection.
        "calibration": calibration_stats(all_priced, results_by_game),
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
