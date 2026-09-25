#!/usr/bin/env python3
"""Write the dashboard's record and gates, generated -- never hand-written.

    python3 scripts/export_record.py

data/site/record.json
    Per league, from data/ledger: settled paper games against the 150-game
    floor, how the model's preferred side has done at the closing price, its
    CLV, the market grade, and the most recent settled games.

data/site/gates.json
    Every check that decides what ships, read from where it is decided:
      - each fitted artifact's own held-out grade (a model that fails its gate
        is refused by its loader; the site shows the same verdict)
      - the validation files that withhold or disclose a market
      - each league's market grade (ADR 0024) and tier bands (ADR 0025)
      - every decision record's status and title (docs/decisions)
    The old board hard-coded its gates as text in JSX, which is how a site
    ends up describing checks the code no longer runs.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from coverline.core.market_weight import staking_weight  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402

SITE = ROOT / "data" / "site"
LEDGER = ROOT / "data" / "ledger"
LEAGUES = ("nfl", "cfb", "mlb", "nhl", "nba")
FLOOR = 150
BREAK_EVEN_110 = 0.5238
MARKET_ORDER = ("spreads", "h2h", "totals")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ record --

def record(ledger_dir: Path = LEDGER) -> dict:
    led = BetLedger(ledger_dir)
    sigs = led.signals()
    outcome = {o.signal_id: o for o in led.outcomes()}
    closes = {c.signal_id: c for c in led.closes()}

    out = {"generated_at": _now(), "floor": FLOOR, "leagues": {}}
    by_league: dict[str, list] = defaultdict(list)
    for s in sigs:
        by_league[s.league].append(s)

    for league in LEAGUES:
        w, grade = staking_weight(league)
        rows = by_league.get(league, [])
        # One row per game, as the market grade counts them: the preferred
        # side (model above market), on the first market in priority order,
        # at the latest pricing before the game.
        per_game: dict[str, object] = {}
        for s in rows:
            if s.edge_claimed is None or s.p_model <= s.p_market or s.game_id is None:
                continue
            rank = MARKET_ORDER.index(s.market) if s.market in MARKET_ORDER else 9
            cur = per_game.get(s.event_id)
            if cur is None or rank < cur[0] or (rank == cur[0] and s.at > cur[1].at):
                per_game[s.event_id] = (rank, s)
        # CLV comes from the game's EARLIEST trade priced well before its
        # close. The latest trade (kept above for the result) is priced from
        # the closing poll itself, so its CLV is zero by construction and
        # averaging it in measured nothing (grade.priced_before_close).
        from coverline.execution.grade import priced_before_close
        early: dict = {}
        for s in rows:
            if s.edge_claimed is None or s.p_model <= s.p_market or s.game_id is None:
                continue
            if not priced_before_close(s, closes.get(s.signal_id)):
                continue
            rank = MARKET_ORDER.index(s.market) if s.market in MARKET_ORDER else 9
            cur = early.get(s.event_id)
            if cur is None or rank < cur[0] or (rank == cur[0] and s.at < cur[1].at):
                early[s.event_id] = (rank, s)
        early_clv = {}
        clv_pts, line_pts = [], []
        for ev, (_, s) in early.items():
            clv = led.clv_of(s, closes.get(s.signal_id))
            if clv is not None and clv.valid:
                early_clv[ev] = clv
                clv_pts.append(clv.prob_points)
                if clv.line_points is not None:
                    line_pts.append(clv.line_points)
        settled, recent = [], []
        for _, s in per_game.values():
            o = outcome.get(s.signal_id)
            clv = early_clv.get(s.event_id)
            if o is None or o.result == "push":
                continue
            settled.append(o.result == "win")
            recent.append({"at": s.at, "game_id": s.game_id, "market": s.market,
                           "selection": s.selection, "line": s.line,
                           "price_decimal": s.price_decimal, "book": s.book,
                           "p_model": round(s.p_model, 4), "p_market": round(s.p_market, 4),
                           "result": o.result, "score": f"{o.home_score}-{o.away_score}",
                           "clv_prob_points": round(clv.prob_points, 4) if clv else None})
        n = len(settled)
        recent.sort(key=lambda r: r["at"], reverse=True)
        out["leagues"][league] = {
            "grade": {"weight": w, **({k: grade.get(k) for k in ("w_hat", "se", "n", "source")}
                                      if grade else {"source": None})},
            "signals": len(rows),
            "settled_games": n,
            "progress_to_floor": round(min(n / FLOOR, 1.0), 4),
            "preferred_side_wins": sum(settled),
            "preferred_side_hit_rate": round(sum(settled) / n, 4) if n else None,
            "break_even_at_minus_110": BREAK_EVEN_110,
            "mean_clv_prob_points": round(sum(clv_pts) / len(clv_pts), 5) if clv_pts else None,
            "mean_clv_line_points": round(sum(line_pts) / len(line_pts), 4) if line_pts else None,
            "clv_graded": len(clv_pts),
            "recent": recent[:50],
        }
    return out


# ------------------------------------------------------------------- gates --

def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


#: (league, what it gates, file). The verdict is read from the file.
ARTIFACT_GATES = (
    ("nba", "Team ratings (walk-forward)", "data/nba_fitted.json"),
    ("nhl", "Goal rates (walk-forward)", "data/nhl_fitted.json"),
    ("nhl", "Goalie-pull and overtime rules layer (puck line)", "data/nhl_rules.json"),
    ("mlb", "Ninth-inning rules layer (moneyline)", "data/mlb_rules.json"),
    ("nfl", "Key-number correction (integer spreads)", "data/nfl_key_numbers.json"),
    ("cfb", "Key-number correction (integer spreads)", "data/cfb_key_numbers.json"),
)
#: The old board's spread Play threshold (data/spread_validation.json) gated
#: PLAY_GAP in the frontend, which no longer exists; its successor is the tier
#: bands' measured hit rates, reported under "market" below (ADR 0025).
VALIDATION_GATES = (
    ("nfl", "Totals", "data/totals_validation.json"),
)


def _decision(path: Path) -> dict:
    text = path.read_text()
    front = dict(re.findall(r"^(\w+):\s*(.*?)\s*(?:#.*)?$",
                            text.split("---")[1], flags=re.M)) if text.startswith("---") else {}
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
    return {"id": path.stem[:4], "file": f"docs/decisions/{path.name}", "title": title,
            "status": front.get("status"), "date": front.get("date"),
            "superseded_by": None if front.get("superseded_by") in (None, "null") else front.get("superseded_by")}


def gates() -> dict:
    out = {"generated_at": _now(), "fitted": [], "validation": [], "market": [], "rules": [],
           "decisions": []}
    for league, what, rel in ARTIFACT_GATES:
        art = _json(ROOT / rel)
        g = (art or {}).get("holdout_grade") or {}
        out["fitted"].append({"league": league, "gate": what, "file": rel,
                              "passed": bool(g.get("supported")), "t": g.get("t"),
                              "paired_gain": g.get("paired_gain"),
                              "missing": art is None})
    for league, what, rel in VALIDATION_GATES:
        art = _json(ROOT / rel) or {}
        out["validation"].append({"league": league, "gate": what, "file": rel,
                                  "supported": art.get("supported"),
                                  "action": art.get("action") or ("WITHHELD" if art.get("supported") is False else None)})
    tiers = (_json(ROOT / "data" / "tier_thresholds.json") or {}).get("leagues", {})
    for league in LEAGUES:
        w, grade = staking_weight(league)
        t = tiers.get(league, {})
        out["market"].append({"league": league, "staking_weight": w,
                              "w_hat": (grade or {}).get("w_hat"), "se": (grade or {}).get("se"),
                              "n": (grade or {}).get("n"), "source": (grade or {}).get("source"),
                              "tiers_source": t.get("source"), "tiers_provisional": t.get("provisional"),
                              # Per band, how the model's preferred side did in
                              # the history the bands came from. None was
                              # distinguishable from break-even when built.
                              "tiers_by_band": t.get("by_band"),
                              "break_even_at_minus_110": BREAK_EVEN_110})
    import export_board
    q = export_board.regime_evidence()
    out["rules"] = [{"league": "nfl", "rule": "Regime cap",
                     "text": (f"Early-season flags backing a first-year external head coach went "
                              f"{q} ATS in 2016-2023; such flags are capped at Lean with half "
                              "stake, never removed."),
                     "file": "model/coach_regime_results.json"}]
    for p in sorted((ROOT / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md")):
        if p.name.startswith("0000"):
            continue
        out["decisions"].append(_decision(p))
    return out


def main(argv: list[str] | None = None) -> int:
    SITE.mkdir(parents=True, exist_ok=True)
    rec = record()
    (SITE / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    g = gates()
    (SITE / "gates.json").write_text(json.dumps(g, indent=1) + "\n")
    for league, r in rec["leagues"].items():
        print(f"{league}: {r['settled_games']}/{FLOOR} settled, weight {r['grade']['weight']}")
    print(f"gates: {len(g['fitted'])} fitted, {len(g['validation'])} validation, "
          f"{len(g['decisions'])} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
