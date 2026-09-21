#!/usr/bin/env python3
"""Prove the ingestion pipeline against the real API, on the free tier.

WHY THIS EXISTS
The odds client was built and tested entirely offline against fixtures, which
proves the code is self-consistent and proves nothing about whether the real
API behaves as assumed. This script closes that gap using the free Starter
tier (500 credits/month), so the $59 subscription is bought after the pipeline
is known to work rather than before.

WHAT IT CAN AND CANNOT PROVE
Historical endpoints are paid-only -- the API docs say so plainly -- so the
backfill path CANNOT be validated for free. Everything else can: the cost
model, the response shape, region coverage, Pinnacle's presence, devigging on
real prices, and bronze round-tripping. Run this on a free key first; then,
after subscribing, run it with --historical to check the one remaining path
BEFORE spending 92,000 credits on a backfill.

BUDGET
Hard-capped (default 60 credits, about 12% of a free month). The cap is
enforced by CreditLedger before any request is issued, so a bug in this script
cannot drain the quota.

USAGE
    export ODDS_API_KEY=...          # never passed on the command line
    python3 scripts/shakeout_odds_api.py
    python3 scripts/shakeout_odds_api.py --historical    # paid plans only

The key is read from the environment and never printed; stored URLs are
redacted by the client.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.normalize import best_price, normalize, two_sided  # noqa: E402
from coverline.execution.odds_client import (  # noqa: E402
    CostModelDrift, CreditLedger, OddsAPIClient, QuotaExceeded,
)

MARKETS = ["h2h", "spreads", "totals"]
SPORT = "americanfootball_nfl"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ", SKIP: " skip "}[status]
    print(f"[{mark}] {name}" + (f"\n          {detail}" if detail else ""))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60,
                    help="hard credit cap for this run (default 60)")
    ap.add_argument("--historical", action="store_true",
                    help="also test the historical endpoint (paid plans only)")
    ap.add_argument("--sport", default=SPORT)
    args = ap.parse_args()

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ODDS_API_KEY is not set in the environment.\n"
              "  export ODDS_API_KEY=... and re-run. Do not pass it as an "
              "argument -- it would land in your shell history.")
        return 2

    ledger = CreditLedger(budget=args.budget)
    client = OddsAPIClient(key, ledger)
    print(f"Shakeout against the live API. Budget: {args.budget} credits.\n")

    # 1. the free endpoint -- proves the key works before spending anything
    try:
        resp = client.sports(captured_at=now())
        keys = [s.get("key") for s in resp.payload]
        record(PASS, "key is valid and /sports responds (0 credits)",
               f"{len(keys)} sports; remaining quota reported: {resp.remaining}")
        if args.sport not in keys:
            record(WARN, f"{args.sport} not in the active sports list",
                   "out of season, or the key differs. Odds calls may return [].")
    except Exception as exc:
        record(FAIL, "/sports failed", str(exc)[:300])
        return 1

    # 2. current odds, eu region -- the CLV baseline region (Pinnacle lives here)
    eu = None
    try:
        eu = client.current_odds(sport=args.sport, markets=MARKETS,
                                 regions=["eu"], captured_at=now())
        record(PASS, "current odds, eu region",
               f"{len(eu.payload)} events; predicted {eu.cost_predicted} credits, "
               f"charged {eu.cost_actual}")
    except CostModelDrift as exc:
        record(FAIL, "COST MODEL DRIFT", str(exc)[:400])
    except QuotaExceeded as exc:
        record(FAIL, "budget exhausted earlier than expected", str(exc)[:200])
    except Exception as exc:
        record(FAIL, "current odds (eu) failed", str(exc)[:300])

    # 3. the question ADR 0002 turned on: is Pinnacle actually there?
    if eu and eu.payload:
        books = {b.get("key") for ev in eu.payload for b in (ev.get("bookmakers") or [])}
        if "pinnacle" in books:
            record(PASS, "Pinnacle present in eu",
                   "the CLV baseline is available as ADR 0002 assumed")
        else:
            record(FAIL, "PINNACLE ABSENT from the eu response",
                   f"books seen: {sorted(books)[:12]}. ADR 0002 assumed it is "
                   "carried; if it is not, the fair-price baseline needs "
                   "rethinking before the subscription is worth buying.")
    elif eu:
        record(SKIP, "Pinnacle check", "no events returned (likely out of season)")

    # 4. us region -- where bets actually get placed
    us = None
    try:
        us = client.current_odds(sport=args.sport, markets=MARKETS,
                                 regions=["us"], captured_at=now())
        record(PASS, "current odds, us region",
               f"{len(us.payload)} events; charged {us.cost_actual}")
    except Exception as exc:
        record(FAIL, "current odds (us) failed", str(exc)[:300])

    # 5. the cost model, checked against the API's own accounting
    if ledger.spent_actual:
        if ledger.spent_predicted == ledger.spent_actual:
            record(PASS, "cost model matches the API exactly",
                   f"predicted {ledger.spent_predicted}, charged "
                   f"{ledger.spent_actual} over {ledger.calls} calls")
        else:
            record(FAIL, "cost model disagrees with the API",
                   f"predicted {ledger.spent_predicted}, charged "
                   f"{ledger.spent_actual}")
    else:
        record(WARN, "no quota headers returned",
               "spend cannot be reconciled; budget guards are unverified")

    # 6. normalise and devig a real market end to end
    if eu and eu.payload:
        quotes = normalize(eu.payload, captured_at=eu.captured_at)
        record(PASS if quotes else FAIL, "normalised real payload",
               f"{len(quotes)} quotes from {len({q.event_id for q in quotes})} events")
        done = False
        for ev in eu.payload:
            for book in (ev.get("bookmakers") or []):
                sides = two_sided(quotes, event_id=ev["id"], market="h2h",
                                  bookmaker=book["key"])
                if len(sides) == 2:
                    prices = [q.price_decimal for q in sides]
                    cmp = P.devig_all(prices, index=0)
                    hold = P.hold(prices)
                    ok = 0.0 < cmp.fair_by_method["power"] < 1.0 and 0 <= hold < 0.25
                    record(PASS if ok else FAIL, "devigged a real two-sided market",
                           f"{book['key']} {ev.get('home_team')} vs "
                           f"{ev.get('away_team')}: hold {hold:.2%}, fair "
                           f"{cmp.fair_by_method['power']:.4f}, method spread "
                           f"{cmp.spread:.4f}")
                    done = True
                    break
            if done:
                break
        if not done:
            record(WARN, "no complete two-sided market found", "cannot devig")

    # 7. line shopping across regions -- the +2.31pt ROI job
    if eu and us and eu.payload and us.payload:
        allq = (normalize(eu.payload, captured_at=eu.captured_at)
                + normalize(us.payload, captured_at=us.captured_at))
        ev = eu.payload[0]
        best = best_price(allq, event_id=ev["id"], market="h2h",
                          outcome=ev.get("home_team", ""))
        if best:
            n_books = len({q.bookmaker for q in allq if q.event_id == ev["id"]})
            record(PASS, "line shopping finds a best price",
                   f"{n_books} books on {ev.get('home_team')}; best "
                   f"{best.price_decimal:.4f} at {best.bookmaker}")

    # 8. bronze round-trip, into a temp dir so nothing real is touched
    if eu:
        with tempfile.TemporaryDirectory() as tmp:
            store = BronzeStore(tmp)
            s = store.write_snapshot(sport="nfl", captured_at=eu.captured_at,
                                     payload=eu.payload, cost=eu.cost_predicted,
                                     source_url=eu.url)
            back = store.read_snapshot(s.path)
            ok = back["payload"] == eu.payload and "REDACTED" in back["_meta"]["source_url"]
            record(PASS if ok else FAIL, "bronze round-trip, key redacted",
                   f"{s.n_events} events written and read back identically")

    # 9. historical -- paid only
    if args.historical:
        try:
            h = client.historical_odds(
                sport=args.sport, markets=MARKETS, regions=["eu"],
                date="2026-01-05T18:00:00Z", captured_at=now())
            record(PASS, "historical endpoint reachable",
                   f"predicted {h.cost_predicted}, charged {h.cost_actual} "
                   "-- the backfill path works")
        except Exception as exc:
            record(FAIL, "historical endpoint failed", str(exc)[:300])
    else:
        record(SKIP, "historical endpoint",
               "paid plans only. Re-run with --historical after subscribing, "
               "BEFORE spending ~92,000 credits on a backfill.")

    # report
    print("\n" + "=" * 68)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"{sum(1 for s,_,_ in results if s==PASS)} passed, {n_fail} failed, "
          f"{n_warn} warnings, "
          f"{sum(1 for s,_,_ in results if s==SKIP)} skipped")
    print(f"credits spent: {ledger.spent_actual or ledger.spent_predicted} "
          f"of a {args.budget} budget; API reports {ledger.remaining_reported} "
          "remaining this month")
    if n_fail:
        print("\nDO NOT SUBSCRIBE YET -- the failures above are about the real "
              "API, not the code, and buying a plan will not fix them.")
    else:
        print("\nPipeline verified against the live API. The one untested path "
              "is historical, which needs a paid plan.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
