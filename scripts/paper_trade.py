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

FROM AN EARLY SNAPSHOT, AS EARLY AS THE INPUTS ALLOW
The daily early poll (14:00 UTC) prices the next 24 hours in the daily
leagues. The weekly football leagues' inputs change once a week, so their
early window runs to the end of the week those inputs are good for -- the
NFL week of the next game (ratings update Tuesday 11:00 UTC), and for CFB
until the next cfb-weekly-job run (Sunday 10:00 UTC). A Tuesday poll then
prices Sunday's games, and CLV measures the week's movement rather than
game-day's. An early poll SKIPS games the ledger already holds, so each game
gets one early trade -- its first -- plus its close.

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
#: Leagues whose early window is the inputs' week, not the next day.
WEEKLY_EARLY = {"nfl", "cfb"}
#: When the CFB ratings refresh (render.yaml cfb-weekly-job: Sunday 10:00 UTC).
CFB_RATINGS_WEEKDAY, CFB_RATINGS_UTC_HOUR = 6, 10

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


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def early_cutoff(league: str, captured_at: str, quotes,
                 schedule: pd.DataFrame | None = None) -> str | None:
    """End of the week the league's inputs are good for, or None for a daily
    league (which keeps EARLY_WINDOW_HOURS)."""
    t0 = _utc(captured_at)
    if league == "cfb":
        days = (CFB_RATINGS_WEEKDAY - t0.weekday()) % 7
        t = (t0 + timedelta(days=days)).normalize() + timedelta(hours=CFB_RATINGS_UTC_HOUR)
        return _iso(t if t > t0 else t + timedelta(days=7))
    if league == "nfl":
        starts = sorted(_utc(q.commence_time) for q in quotes
                        if q.commence_time and _utc(q.commence_time) >= t0)
        if not starts:
            return _iso(t0)
        week = nfl_week({local_date(_iso(starts[0]))}, schedule)
        if week is None:
            return _iso(t0)
        same = [t for t in starts if nfl_week({local_date(_iso(t))}, schedule) == week]
        return _iso(same[-1] + timedelta(seconds=1))
    return None


def slates(sport: str, captured_at: str, quotes, window_hours: float = WINDOW_HOURS,
           schedule: pd.DataFrame | None = None,
           cutoff: str | None = None) -> tuple[list[list[str]], str]:
    """The runner arguments this capture calls for, and the cutoff used.
    `cutoff`, when given, replaces the window."""
    league = LEAGUE_OF.get(sport)
    t0 = _utc(captured_at)
    cutoff = cutoff or _iso(t0 + timedelta(hours=window_hours))
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


def already_priced(ledger_dir: Path, league: str | None) -> set[str]:
    """Vendor event ids this ledger holds any signal for, in `league`."""
    if league is None or not Path(ledger_dir).exists():
        return set()
    from coverline.execution.ledger import BetLedger
    return {s.event_id for s in BetLedger(ledger_dir).signals() if s.league == league}


def paper_trade(snapshot: Path, ledger_dir: Path = LEDGER_DIR,
                window_hours: float | None = None) -> list[tuple[list[str], int]]:
    rec = BronzeStore(snapshot.parents[2]).read_snapshot(snapshot)
    meta = rec["_meta"]
    early = meta.get("kind") == "early"
    quotes = normalize(rec["payload"], captured_at=meta["captured_at"])
    league = LEAGUE_OF.get(meta["sport"])
    cut, skip = None, set()
    if early:
        skip = already_priced(ledger_dir, league)
        quotes = [q for q in quotes if q.event_id not in skip]
        if window_hours is None:
            cut = early_cutoff(league, meta["captured_at"], quotes)
    if window_hours is None:
        window_hours = EARLY_WINDOW_HOURS if early else WINDOW_HOURS
    runs, cutoff = slates(meta["sport"], meta["captured_at"], quotes, window_hours, cutoff=cut)
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
        if skip:
            argv += ["--skip-events", ",".join(sorted(skip))]
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
