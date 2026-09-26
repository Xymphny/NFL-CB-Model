#!/usr/bin/env python3
"""The matchup detail the board exports: the model's margin distribution and
the line since open (dashboard v2 items 9 and 10).

The site draws bars and a sparkline from these and computes nothing: no pmf,
no fair line, no interpolation.

MARGIN PMF. From the distribution the board priced with, over each league's
window, home margin. Discrete leagues give mass per integer margin; NBA is
modelled continuously, so its mass is the CDF binned to integers and says so.
Mass beyond the window is reported as tails, never dropped. Key numbers are
the margins the league's MEASURED key-number table weights at 1.5x or more.

LINE HISTORY. One point per capture holding the game: the median line and
price across books, and the median devigged probability, from the side the
board shows. Read one snapshot at a time, summarised as it goes -- the store
grows by dozens of snapshots a day and this runs inside a 512 MB job. Missed
windows come from the gap log and are listed, never filled in: "1 capture so
far" is a valid state. The close is marked once the game has started: the last
pre-start capture that was not an early poll.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

#: Home-margin window per league (points, goals, runs).
PMF_WINDOW = {"nfl": 28, "cfb": 28, "nba": 40, "nhl": 8, "mlb": 12}
#: A key number is a margin the measured table weights at least this much.
KEY_WEIGHT = 1.5
#: How far back a game's line history reads.
HISTORY_DAYS = 10

TOTALS_REASON = {
    "nfl": "Not priced: the totals model failed its grade.",
    "cfb": "Not priced: no totals model has been graded.",
    "nba": "Not priced: no totals model has been graded.",
}


def key_numbers(league: str) -> list[int]:
    try:
        if league == "nfl":
            from coverline.leagues.nfl.model import _load_key_number_weights
            w = _load_key_number_weights() or {}
        elif league == "cfb":
            from coverline.leagues.cfb.model import KEY_NUMBER_WEIGHTS
            w = dict(KEY_NUMBER_WEIGHTS or {})
        else:
            return []
    except Exception:
        return []
    lim = PMF_WINDOW[league]
    return sorted(k for k, v in w.items() if v >= KEY_WEIGHT and 0 < abs(k) <= lim)


def margin_pmf(league: str, dist) -> dict | None:
    """{from, to, mass[], binned, tail_below, tail_above, key_numbers}."""
    lim = PMF_WINDOW.get(league)
    if lim is None:
        return None
    try:
        d = getattr(dist, "is_discrete", True)
        discrete = bool(d() if callable(d) else d)
        ks = range(-lim, lim + 1)
        if discrete:
            mass = [float(dist.margin_pmf(k)) for k in ks]
            below = float(dist.margin_cdf(-lim - 1))
        else:
            mass = [float(dist.margin_cdf(k + 0.5) - dist.margin_cdf(k - 0.5)) for k in ks]
            below = float(dist.margin_cdf(-lim - 0.5))
        above = max(0.0, 1.0 - below - sum(mass))
        return {"from": -lim, "to": lim, "mass": [round(m, 5) for m in mass],
                "binned": not discrete, "tail_below": round(below, 5),
                "tail_above": round(above, 5), "key_numbers": key_numbers(league)}
    except Exception:
        return None


def totals_view(league: str, dist) -> dict:
    ok = getattr(dist, "total_validated", True) is not False
    return {"priced": ok, "reason": None if ok else TOTALS_REASON.get(
        league, "Not priced: the totals model is not validated.")}


def fair_line(dist, side: str) -> float | None:
    """The spread at which the model is even, from `side`'s view: minus the
    model's mean home margin for home, plus it for away."""
    try:
        mu = float(dist.margin_mean())
    except Exception:
        return None
    return round(-mu if side == "home" else mu, 2)


def _devig_home(h: float, a: float) -> float | None:
    try:
        from coverline.core import pricing as P
        return float(P.devig([h, a], "power")[0])
    except Exception:
        return None


def summarise_event(event: dict, market_key: str) -> dict | None:
    """One snapshot's view of one game: median home point, prices, and
    devigged home probability across books. Pure."""
    home, away = event.get("home_team"), event.get("away_team")
    points, hp, ap, ps = [], [], [], []
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            by = {o.get("name"): o for o in m.get("outcomes", [])}
            h, a = by.get(home), by.get(away)
            if not h or not a:
                continue
            if h.get("point") is not None:
                points.append(float(h["point"]))
            hp.append(float(h["price"]))
            ap.append(float(a["price"]))
            p = _devig_home(float(h["price"]), float(a["price"]))
            if p is not None:
                ps.append(p)
    if not hp:
        return None
    med = statistics.median
    return {"home_line": med(points) if points else None, "home_price": round(med(hp), 3),
            "away_price": round(med(ap), 3), "p_home": round(med(ps), 5) if ps else None,
            "books": len(hp)}


def history(store, sport: str, want: dict[str, str], now: pd.Timestamp) -> dict[str, list]:
    """{event_id: [points oldest first]} for `want` = {event_id: market key}."""
    out: dict[str, list] = {e: [] for e in want}
    if not want:
        return out
    since = (now - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in store.snapshots(sport):
        if row.get("captured_at", "") < since:
            continue
        path = store.root / row["path"]
        try:
            rec = store.read_snapshot(path)
        except Exception:
            continue
        for e in rec.get("payload", []):
            eid = e.get("id")
            if eid not in want:
                continue
            s = summarise_event(e, want[eid])
            if s:
                out[eid].append({"captured_at": rec["_meta"]["captured_at"],
                                 "kind": row.get("kind", "current"), **s})
        del rec
    for pts in out.values():
        pts.sort(key=lambda p: p["captured_at"])
    return out


def for_side(points: list[dict], side: str, start: str | None, gaps: list[dict]) -> dict:
    """A game's history from the side the board shows, with the close marked
    once the game has started and the missed windows listed."""
    rows = []
    for p in points:
        line = p["home_line"]
        if side == "away" and line is not None:
            line = -line
        prob = p["p_home"] if side == "home" else (None if p["p_home"] is None else 1 - p["p_home"])
        rows.append({"captured_at": p["captured_at"], "kind": p["kind"], "line": line,
                     "price": p["home_price"] if side == "home" else p["away_price"],
                     "p_market": None if prob is None else round(prob, 5), "books": p["books"]})
    started = bool(start) and pd.Timestamp(start) <= pd.Timestamp.now(tz="UTC")
    close_at = None
    if started:
        pre = [r for r in rows if r["captured_at"] < start and r["kind"] != "early"]
        close_at = pre[-1]["captured_at"] if pre else None
    # Only THIS game's missed windows: the gap log is league-wide, and a
    # window missed for another game's close is not a hole in this line. A
    # close window sits in the half hour before first pitch; an early poll
    # is the day's, for every game still to come.
    first = rows[0]["captured_at"] if rows else None
    lo = (pd.Timestamp(start) - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ") \
        if start else None
    missed = [g for g in gaps if start and (
        (lo <= g["intended_at"] < start and not g["reason"].startswith("early_"))
        or (g["reason"].startswith("early_") and first and first <= g["intended_at"] < start))]
    return {"points": rows, "captures": len(rows), "close_at": close_at, "frozen": started,
            "missed_windows": [{"intended_at": g["intended_at"], "reason": g["reason"]}
                               for g in missed]}


def true_gaps(store, sport: str) -> list[dict]:
    """Logged gaps whose window was in fact never taken (bronze.coverage's
    rule), sorted."""
    try:
        snaps = list(store.snapshots(sport))
        taken = {r["captured_at"] for r in snaps}
        early = {r["captured_at"][:10] for r in snaps if r.get("kind") == "early"}
        out = [g for g in store.gaps(sport)
               if g["intended_at"] not in taken
               and not (g["reason"].startswith("early_") and g["intended_at"][:10] in early)]
        seen, uniq = set(), []
        for g in sorted(out, key=lambda g: g["intended_at"]):
            if (g["intended_at"], g["reason"]) not in seen:
                seen.add((g["intended_at"], g["reason"]))
                uniq.append(g)
        return uniq
    except Exception:
        return []
