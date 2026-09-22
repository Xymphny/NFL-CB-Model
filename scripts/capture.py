#!/usr/bin/env python3
"""Plan and run the close-capture that CLV is measured against.

WHAT THIS DOES
Asks the API which games are coming (FREE -- the vendor's guide says the
events endpoint "does not count against the usage quota"), collapses their
kickoff times into the fewest polls that cover them, and captures each one
shortly before it starts. Games within a few minutes of each other share a
poll, so an NBA night with twelve games at five tip times costs five requests
rather than twelve.

DRY BY DEFAULT. It prints the plan and the credits it would spend and exits.
`--run` is the only thing that issues a paid request, which is the same
contract scripts/backfill.py has, and for the same reason: the monthly quota
is shared with the backfill and running out in February is a planning failure
rather than an error to handle.

WHAT IT REFUSES TO DO
Capture an overdue window. A poll taken after kickoff returns an IN-PLAY
price, which is a different instrument from a close, and quietly filing one
under a pre-game timestamp corrupts every CLV figure computed from it
afterwards. Those windows are gapped with a reason instead.

USAGE
    export ODDS_API_KEY=...
    python3 scripts/capture.py                 # plan only
    python3 scripts/capture.py --run --budget 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.capture import (  # noqa: E402
    CaptureWindow, estimate_monthly, run, windows_for_slate,
)
from coverline.execution.odds_client import (  # noqa: E402
    CreditLedger, OddsAPIClient,
)

#: The vendor's sport keys, and the regions each league's prices come from.
#: Three markets per poll: h2h, spreads, totals. Cost is markets x regions, so
#: one region is the difference between 3 credits a poll and 6.
SPORTS = {
    "americanfootball_nfl": ("eu",),
    "americanfootball_ncaaf": ("eu",),
    "basketball_nba": ("eu",),
    "icehockey_nhl": ("eu",),
    "baseball_mlb": ("eu",),
}

#: Windows a league typically needs in a week, for the monthly estimate. These
#: are counts of distinct kickoff CLUSTERS, not games.
TYPICAL_WEEKLY_WINDOWS = {
    "americanfootball_nfl": 6,
    "americanfootball_ncaaf": 8,
    "basketball_nba": 30,
    "icehockey_nhl": 20,
    "baseball_mlb": 35,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


PLAN_CACHE = ROOT / "data" / "capture_plan.json"


def _cached_plan(now: str, max_age_minutes: int) -> list[dict] | None:
    """The last plan, if it is still fresh.

    WHY CACHE SOMETHING THAT IS FREE. The events endpoint costs no credits,
    but a cron at this cadence would still issue five requests every run --
    about 480 a day against an endpoint somebody else pays to serve. Free to
    us is not free, and a plan does not change between two polls fifteen
    minutes apart.
    """
    if not PLAN_CACHE.exists():
        return None
    try:
        blob = json.loads(PLAN_CACHE.read_text())
        built = datetime.fromisoformat(blob["built_at"].replace("Z", "+00:00"))
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    age = datetime.fromisoformat(now.replace("Z", "+00:00")) - built
    if age > timedelta(minutes=max_age_minutes) or age < timedelta(0):
        return None
    return blob["windows"]


def plan(client: OddsAPIClient, *, horizon_hours: int, now: str,
         lead_minutes: int, cluster_minutes: int,
         cache_minutes: int = 0) -> list[CaptureWindow]:
    """Every window due inside the horizon. Costs nothing to compute."""
    if cache_minutes:
        cached = _cached_plan(now, cache_minutes)
        if cached is not None:
            return [CaptureWindow(sport=w["sport"], at=w["at"],
                                  markets=tuple(w["markets"]),
                                  regions=tuple(w["regions"]))
                    for w in cached]
    cutoff = datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(
        hours=horizon_hours)
    out: list[CaptureWindow] = []
    for sport, regions in SPORTS.items():
        resp = client.events(sport=sport, captured_at=now)
        starts = [e["commence_time"] for e in resp.payload
                  if e.get("commence_time")
                  and datetime.fromisoformat(
                      e["commence_time"].replace("Z", "+00:00")) <= cutoff]
        if not starts:
            continue
        out += windows_for_slate(sport=sport, commence_times=starts,
                                 lead_minutes=lead_minutes,
                                 cluster_minutes=cluster_minutes,
                                 regions=regions)
    out = sorted(out, key=lambda w: w.at)
    if cache_minutes:
        PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PLAN_CACHE.write_text(json.dumps({
            "built_at": now,
            "horizon_hours": horizon_hours,
            "windows": [{"sport": w.sport, "at": w.at,
                         "markets": list(w.markets), "regions": list(w.regions)}
                        for w in out],
        }, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="actually spend credits")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--horizon-hours", type=int, default=6)
    ap.add_argument("--lead-minutes", type=int, default=5)
    ap.add_argument("--cluster-minutes", type=int, default=20)
    ap.add_argument("--tolerance-minutes", type=int, default=10)
    ap.add_argument("--plan-cache-minutes", type=int, default=60,
                    help="reuse the last slate plan for this long; 0 disables")
    a = ap.parse_args(argv)

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY is not set. It is never passed on the command "
              "line; export it.")
        return 2

    now = _now()
    ledger = CreditLedger(budget=a.budget)
    client = OddsAPIClient(api_key=key, ledger=ledger)

    windows = plan(client, horizon_hours=a.horizon_hours, now=now,
                   lead_minutes=a.lead_minutes,
                   cluster_minutes=a.cluster_minutes,
                   cache_minutes=a.plan_cache_minutes)
    cost = sum(w.cost for w in windows)
    print(f"{now}: {len(windows)} windows in the next {a.horizon_hours}h, "
          f"{cost} credits (events lookups were free)")
    for w in windows:
        print(f"  {w.at}  {w.sport:24} {w.cost:>2} credits")

    monthly = estimate_monthly(TYPICAL_WEEKLY_WINDOWS, markets=3, regions=1)
    print("\ntypical month at this shape, if every league is in season:")
    for sport, credits in sorted(monthly.items(), key=lambda kv: -kv[1]):
        if sport != "TOTAL":
            print(f"  {sport:24} {credits:>7,}")
    print(f"  {'TOTAL':24} {monthly['TOTAL']:>7,}  "
          f"({monthly['TOTAL'] / 100_000:.0%} of a 100,000-credit month)")
    print("  NOTE: the five leagues do not all overlap -- NFL and CFB stop in "
          "January,\n        MLB starts in March -- so this is the worst case, "
          "not the average.\n        The backfill is 89,760 credits and cannot "
          "share a month with it.")

    if not a.run:
        print("\nDry run. Nothing was fetched and no credits were spent.")
        print("Add --run to execute.")
        return 0

    if cost > a.budget:
        print(f"\nplan costs {cost} credits against a budget of {a.budget}; "
              "the run will capture what fits and GAP the rest with a reason.")

    store = BronzeStore(ROOT / "data" / "bronze")
    res = run(windows, client, store, now=now,
              tolerance_minutes=a.tolerance_minutes)
    print(f"\n{res.summary()}")
    for key_, reason in res.gapped:
        print(f"  gapped {key_}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
