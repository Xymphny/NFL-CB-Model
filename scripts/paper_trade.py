#!/usr/bin/env python3
"""Price the games a capture was taken for, and record them as paper trades.

    python3 scripts/paper_trade.py --snapshot data/bronze/odds/<sport>/current-<ts>.json

Called by scripts/capture.py after every capture (--paper), so each game gets
a paper signal priced at the window just before it starts -- the closing
price, which is the sharpest number there is to be graded against. Every
league paper-trades until that grade says otherwise (ADR 0024);
scripts/settle_ledger.py settles these rows and model/grade_market_weight.py
grades them.

WHICH GAMES
Only those starting within WINDOW_HOURS of the capture. The odds endpoint
returns every upcoming game, and pricing a week of NBA on every capture would
write hundreds of thousands of rows a season, most of them far from the close.

WHICH SLATE
Date leagues (CFB, MLB, NHL, NBA) run once per US-Eastern date among those
games. NFL runs by week, found from the league schedule.

ALWAYS --paper. An automated run is never a placed bet.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.matching import local_date  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402
from coverline.execution.settle import VENDOR  # noqa: E402

LEDGER_DIR = ROOT / "data" / "ledger"
WINDOW_HOURS = 3.0
#: From the daily EARLY snapshot (14:00 UTC), the next day of games: each game
#: falls in exactly one day's window, so it gets exactly one early paper trade,
#: whose CLV against the later close is the measurement (capture.EARLY_UTC_HOUR).
EARLY_WINDOW_HOURS = 24.0
#: Nominal. At weight 0 nothing is sized; the figure only has to be valid.
BANKROLL = os.environ.get("PAPER_BANKROLL", "1000")

LEAGUE_OF = {v: k for k, v in VENDOR.items()}


def _runner():
    spec = importlib.util.spec_from_file_location(
        "recommend_slate", ROOT / "scripts" / "recommend_slate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recommend_slate"] = mod
    spec.loader.exec_module(mod)
    return mod


def nfl_week(dates: set[str], schedule: pd.DataFrame | None = None) -> tuple[int, int] | None:
    """(season, week) whose regular-season games fall on these ET dates."""
    if schedule is None:
        from coverline.leagues.nfl.live import load_schedule
        years = sorted({int(d[:4]) for d in dates} | {int(d[:4]) - 1 for d in dates})
        schedule = load_schedule(years)
    s = schedule[(schedule.game_type == "REG") & schedule.gameday.astype(str).isin(dates)]
    weeks = sorted(set(zip(s.season.astype(int), s.week.astype(int))))
    return weeks[-1] if weeks else None


def slates(sport: str, captured_at: str, quotes, window_hours: float = WINDOW_HOURS,
           schedule: pd.DataFrame | None = None) -> tuple[list[list[str]], str]:
    """The runner arguments this capture calls for, and the cutoff used."""
    league = LEAGUE_OF.get(sport)
    t0 = pd.Timestamp(captured_at)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    cutoff = (t0 + timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    starts = {q.commence_time for q in quotes if q.commence_time
              and t0 <= pd.Timestamp(q.commence_time) < pd.Timestamp(cutoff)}
    dates = {local_date(c) for c in starts}
    if not dates or league is None:
        return [], cutoff
    if league == "nfl":
        sw = nfl_week(dates, schedule)
        return ([["--league", "nfl", "--season", str(sw[0]), "--week", str(sw[1])]]
                if sw else []), cutoff
    return [["--league", league, "--date", d] for d in sorted(dates)], cutoff


def paper_trade(snapshot: Path, ledger_dir: Path = LEDGER_DIR,
                window_hours: float | None = None) -> list[tuple[list[str], int]]:
    rec = BronzeStore(snapshot.parents[2]).read_snapshot(snapshot)
    meta = rec["_meta"]
    if window_hours is None:
        window_hours = EARLY_WINDOW_HOURS if meta.get("kind") == "early" else WINDOW_HOURS
    quotes = normalize(rec["payload"], captured_at=meta["captured_at"])
    runs, cutoff = slates(meta["sport"], meta["captured_at"], quotes, window_hours)
    if not runs:
        print(f"[paper] {snapshot.name}: nothing to price "
              f"({LEAGUE_OF.get(meta['sport'], meta['sport'])}, window to {cutoff})")
        return []
    R = _runner()
    out = []
    for args in runs:
        argv = [*args, "--snapshot", str(snapshot), "--events-before", cutoff,
                "--paper", "--commit", "--ledger", str(ledger_dir),
                "--bankroll", BANKROLL]
        print(f"[paper] {' '.join(args)} (games before {cutoff})", flush=True)
        try:
            code = R.main(argv)
        except SystemExit as e:          # argparse refusals
            code = int(e.code or 0)
        out.append((args, code))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--ledger", default=str(LEDGER_DIR))
    ap.add_argument("--window-hours", type=float, default=None,
                    help=f"default {WINDOW_HOURS}h from a close, {EARLY_WINDOW_HOURS}h from an early poll")
    a = ap.parse_args(argv)
    res = paper_trade(Path(a.snapshot), Path(a.ledger), a.window_hours)
    # Exit 2 means "inputs not ready" (a stale file, a missing slate) and is
    # reported, not fatal; anything else non-zero is a failure.
    return 1 if any(code not in (0, 2) for _, code in res) else 0


if __name__ == "__main__":
    raise SystemExit(main())
