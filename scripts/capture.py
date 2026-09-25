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

WHERE THE SNAPSHOTS LIVE
In the repository checkout, under data/bronze, committed back after any run
that captured something. That is not a preference -- on Render a cron run
starts from a FRESH CHECKOUT and its filesystem is discarded when the process
exits, so the checkout is the only state that survives. A snapshot written
anywhere else is a snapshot that does not exist, and a CLV series with silent
holes in it is the exact failure capture.py's own docstring says it exists to
prevent.

The cost is bearable and was measured rather than assumed. A real odds
snapshot in this repository averages 26 KB; 99 windows a week with every
league in season is about 5,100 snapshots and 139 MB a year, and realistically
less because the seasons only partly overlap. Pushes are per RUN THAT
CAPTURED, not per run: at a fifteen-minute cadence that is roughly 14 a day,
not 96. deploy/git_utils.py already does fetch-rebase-retry with a stash,
hardened by a CFB snapshot that was lost to exactly this race on 2026-09-05.

IT CANNOT SPEND UNTIL IT IS TOLD TO
CAPTURE_ENABLED must be exactly "1". Unset or anything else and the job exits
before an API client is constructed, so it makes no request of any kind and
costs nothing -- which is what lets the schedule exist before the subscription
does. --run is still required on top of that, the same contract
scripts/backfill.py has.

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
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.capture import (  # noqa: E402
    EARLY_TOLERANCE_MINUTES, EARLY_UTC_HOUR, CaptureWindow, estimate_monthly, run,
    windows_for_slate,
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


#: The Odds API plan in force: the $30 tier, bought 2026-09-24 while only the
#: football leagues and the end of MLB were in season. Change this with the plan.
MONTHLY_CREDITS = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


PLAN_CACHE = ROOT / "data" / "capture_plan.json"

#: The store lives in the checkout because on Render nothing else survives.
BRONZE_ROOT = ROOT / "data" / "bronze"


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


def early_windows(sport: str, starts: list[datetime], now: datetime,
                  horizon_hours: int, regions) -> list[CaptureWindow]:
    """Today's and tomorrow's 14:00 UTC early poll, where one is in reach and
    the sport has a game in the 24 hours after it (out of season: nothing).

    In reach means from EARLY_TOLERANCE_MINUTES before now -- a run that
    starts late in the morning still takes the day's early price -- to the
    horizon, so the plan cache already holds it."""
    out = []
    for days in (0, 1):
        at = (now + timedelta(days=days)).replace(hour=EARLY_UTC_HOUR, minute=0,
                                                  second=0, microsecond=0)
        if not (now - timedelta(minutes=EARLY_TOLERANCE_MINUTES) <= at
                <= now + timedelta(hours=horizon_hours)):
            continue
        if any(at <= s < at + timedelta(hours=24) for s in starts):
            out.append(CaptureWindow(sport=sport, at=at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     regions=tuple(regions), reason="early"))
    return out


def plan(client: OddsAPIClient, *, horizon_hours: int, now: str,
         lead_minutes: int, cluster_minutes: int,
         cache_minutes: int = 0) -> list[CaptureWindow]:
    """Every window due inside the horizon. Costs nothing to compute."""
    if cache_minutes:
        cached = _cached_plan(now, cache_minutes)
        if cached is not None:
            return [CaptureWindow(sport=w["sport"], at=w["at"],
                                  markets=tuple(w["markets"]),
                                  regions=tuple(w["regions"]),
                                  reason=w.get("reason", "close"))
                    for w in cached]
    cutoff = datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(
        hours=horizon_hours)
    out: list[CaptureWindow] = []
    t_now = datetime.fromisoformat(now.replace("Z", "+00:00"))
    for sport, regions in SPORTS.items():
        resp = client.events(sport=sport, captured_at=now)
        every = [datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
                 for e in resp.payload if e.get("commence_time")]
        starts = [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in every if d <= cutoff]
        if starts:
            out += windows_for_slate(sport=sport, commence_times=starts,
                                     lead_minutes=lead_minutes,
                                     cluster_minutes=cluster_minutes,
                                     regions=regions)
        out += early_windows(sport, every, t_now, horizon_hours, regions)
    out = sorted(out, key=lambda w: w.at)
    if cache_minutes:
        PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PLAN_CACHE.write_text(json.dumps({
            "built_at": now,
            "horizon_hours": horizon_hours,
            "windows": [{"sport": w.sport, "at": w.at,
                         "markets": list(w.markets), "regions": list(w.regions),
                         "reason": w.reason}
                        for w in out],
        }, indent=2) + "\n")
    return out


def _isolated(argv: list[str]) -> int:
    """Run one step as `python <argv>` from the repo root; its return code.
    Negative means the child was killed by that signal (an OOM kill is -9)."""
    return subprocess.run([sys.executable, *argv], cwd=ROOT).returncode


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
    ap.add_argument("--paper", action="store_true",
                    help="after each capture, price the games in its window as "
                         "paper trades (scripts/paper_trade.py)")
    ap.add_argument("--persist", action="store_true",
                    help="commit captured snapshots back to the repository, "
                         "which on Render is the only state that survives")
    a = ap.parse_args(argv)

    # THE GATE, checked before anything can issue a request. An unset or
    # non-"1" value exits here, so the job can be scheduled before the
    # subscription exists without costing a credit.
    if os.environ.get("CAPTURE_ENABLED") != "1":
        print("CAPTURE_ENABLED is not '1'. Exiting before any request is "
              "made, so this run costs nothing. Set it deliberately when the "
              "subscription is live.")
        return 0

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

    # Plus the daily early poll: seven a week per league in season.
    monthly = estimate_monthly({k: v + 7 for k, v in TYPICAL_WEEKLY_WINDOWS.items()},
                               markets=3, regions=1)
    print("\ntypical month at this shape, if every league is in season:")
    for sport, credits in sorted(monthly.items(), key=lambda kv: -kv[1]):
        if sport != "TOTAL":
            print(f"  {sport:24} {credits:>7,}")
    print(f"  {'TOTAL':24} {monthly['TOTAL']:>7,}  "
          f"({monthly['TOTAL'] / MONTHLY_CREDITS:.0%} of the "
          f"{MONTHLY_CREDITS:,}-credit plan)")
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

    store = BronzeStore(BRONZE_ROOT)
    res = run(windows, client, store, now=now,
              tolerance_minutes=a.tolerance_minutes)
    print(f"\n{res.summary()}")
    for key_, reason in res.gapped:
        print(f"  gapped {key_}: {reason}")

    # PAPER AND EXPORT RUN IN CHILD PROCESSES. The board and record export
    # alone peaks at ~370 MB against the cron's 512 MB, after paper trading
    # has loaded every model in the same interpreter. An out-of-memory kill is
    # not an exception: in-process it killed the job before the commit below,
    # and on Render's ephemeral disk the snapshot -- the one thing that cannot
    # be fetched again -- went with it. A child can only take itself down.
    papered, killed = 0, False
    if a.paper and res.captured:
        for key_ in res.captured:
            rc = _isolated(["scripts/paper_trade.py", "--snapshot", str(res.paths[key_])])
            if rc < 0:
                killed = True
                print(f"[paper] {key_} killed by signal {-rc}; ledger not committed")
            elif rc == 0:
                papered += 1
            else:
                print(f"[paper] {key_} exited {rc}")

    exported = False
    if a.paper and res.captured:
        # Only a CLEAN export is published: a child killed mid-write can
        # leave truncated JSON, and the site would serve it.
        exported = all(_isolated([f"scripts/{m}.py"]) == 0
                       for m in ("export_board", "export_record"))
        if not exported:
            print("[export] failed; data/site not committed")

    if a.persist and (res.captured or res.gapped):
        # Only when something changed. Most runs have no window due, and a
        # push per run would be 96 a day against a branch three other crons
        # and a human already share.
        sys.path.insert(0, str(ROOT))
        from deploy.git_utils import git_commit_and_push

        git_commit_and_push(
            [str(BRONZE_ROOT.relative_to(ROOT))]
            + ([] if killed else ["data/ledger"])
            + (["data/site"] if exported else []),
            f"capture {now}: {len(res.captured)} snapshots, "
            f"{len(res.gapped)} gapped, {res.credits_spent} credits"
            + (f", {papered} paper snapshot(s)" if papered else ""),
        )
    elif a.persist:
        print("nothing captured or gapped; no commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
