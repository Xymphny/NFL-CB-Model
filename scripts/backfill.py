#!/usr/bin/env python3
"""Plan and run the historical CLV backfill.

DRY BY DEFAULT. With no flags this prints what the job would cost and touches
nothing. Spending requires --run, which is deliberate: the full five-league
plan is ~90,000 credits, most of a month on the 100K tier, and a job that size
should not be one typo away from starting.

    python3 scripts/backfill.py                      # cost it, spend nothing
    python3 scripts/backfill.py --season 2024        # a different season
    python3 scripts/backfill.py --sports nfl nba     # a subset
    python3 scripts/backfill.py --run --budget 30000 # spend, capped

RESUMABLE. Re-running skips whatever is already in bronze, so a job can be
taken in slices across several days and a crash costs only the unfetched part.

BEFORE THE FIRST RUN: verify the historical endpoint works, which costs 30
credits and protects this job:
    python3 scripts/shakeout_odds_api.py --historical
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverline.execution import backfill as B  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.odds_client import CreditLedger, OddsAPIClient  # noqa: E402

BRONZE = ROOT / "data" / "bronze"

#: Season windows and the UTC clock times worth sampling, chosen so one
#: request covers a whole kickoff block. Offsets are from the season's start
#: year. Edit deliberately -- every added time multiplies the cost of a league.
LEAGUES = {
    "nfl": dict(sport="americanfootball_nfl", start="09-05", end="01-05",
                times=["17:00", "20:05", "20:25", "00:15"], weekdays=[3, 5, 6, 0],
                end_next_year=True),
    "ncaaf": dict(sport="americanfootball_ncaaf", start="08-24", end="12-07",
                  times=["16:00", "19:30", "23:00", "02:00"], weekdays=[4, 5],
                  end_next_year=False),
    "nba": dict(sport="basketball_nba", start="10-22", end="04-13",
                times=["23:00", "23:30", "00:00", "02:00", "02:30"], weekdays=None,
                end_next_year=True),
    "nhl": dict(sport="icehockey_nhl", start="10-04", end="04-17",
                times=["23:00", "23:30", "00:00", "02:00"], weekdays=None,
                end_next_year=True),
    "mlb": dict(sport="baseball_mlb", start="03-28", end="09-29",
                times=["17:00", "19:00", "23:00", "00:00", "02:00"], weekdays=None,
                end_next_year=False),
}


def build_plan(season: int, leagues: list[str], regions: list[str]) -> B.Plan:
    groups = []
    for name in leagues:
        cfg = LEAGUES[name]
        end_year = season + 1 if cfg["end_next_year"] else season
        groups.append(B.daily_snapshots(
            sport=cfg["sport"],
            start=f"{season}-{cfg['start']}",
            end=f"{end_year}-{cfg['end']}",
            times_of_day=cfg["times"],
            weekdays=cfg["weekdays"],
            regions=regions,
        ))
    return B.plan(*groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--sports", nargs="+", default=list(LEAGUES),
                    choices=list(LEAGUES))
    ap.add_argument("--regions", nargs="+", default=["eu"])
    ap.add_argument("--run", action="store_true", help="actually spend credits")
    ap.add_argument("--budget", type=int, default=30_000)
    args = ap.parse_args()

    store = BronzeStore(BRONZE)
    full = build_plan(args.season, args.sports, args.regions)
    todo = B.remaining(full, store)

    print(f"Season {args.season}, regions {args.regions}\n")
    print("FULL PLAN")
    print(full.describe())
    done = len(full) - len(todo)
    if done:
        print(f"\n  {done} snapshots already in bronze")
    print("\nOUTSTANDING")
    print(todo.describe())

    monthly = 100_000
    print(f"\n  outstanding is {todo.total_cost / monthly:.0%} of a "
          f"{monthly:,}-credit month")
    if todo.total_cost > 0.75 * monthly:
        print("  NOTE: this leaves little headroom for live capture in the same\n"
              "        month. Consider splitting across two months with --sports,\n"
              "        or narrowing --season. The job is resumable either way.")

    if not args.run:
        print("\nDry run. Nothing was fetched and no credits were spent.")
        print("Add --run to execute.")
        return 0

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("\nODDS_API_KEY is not set. Refusing to run.")
        return 2
    if todo.total_cost > args.budget:
        print(f"\nOutstanding cost {todo.total_cost:,} exceeds --budget "
              f"{args.budget:,}.\nThe run will stop when the budget is reached "
              "and gap the remainder, which is safe and resumable.")

    client = OddsAPIClient(key, CreditLedger(budget=args.budget))
    print(f"\nRunning with a {args.budget:,}-credit cap...\n")
    res = B.run(todo, client, store)
    print(res.summary())
    if res.stopped_early:
        print("Re-run the same command with a fresh budget to continue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
