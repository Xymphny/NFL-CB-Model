#!/usr/bin/env python3
"""Price a slate end to end: model -> market -> stake -> ledger.

This is the operator-facing command. Everything else in the new core is a
component; this is the thing you actually run.

    python3 scripts/recommend_slate.py --league nfl --week 2
    python3 scripts/recommend_slate.py --league nfl --week 2 --snapshot <file>
    python3 scripts/recommend_slate.py --league nfl --week 2 --commit

DRY BY DEFAULT. Nothing is written to the ledger without --commit, because
the ledger is append-only and a slate run that turns out to be misconfigured
cannot be taken back out of it.

IT USES THE ROBUST SHRINKAGE WEIGHT, NOT THE POOLED ONE
core.evidence currently reports a pooled weight near 0.89 and a robust weight
near 0.52, and the whole gap rests on a single attempt. The robust figure is
the one that survives losing any single observation, and sizing real money
from a number resting on one experiment is the mistake the shrinkage machinery
exists to avoid. --pooled overrides this and prints a warning.

IT DOES NOT PLACE BETS. It prints recommendations and, with --commit, records
them. Placing is a human action and the fill gets recorded separately.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from coverline.core import evidence as E  # noqa: E402
from coverline.execution import recommend as Rc  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402
from coverline.leagues.nfl.events import match_events, unmatched  # noqa: E402

LEDGER_DIR = ROOT / "data" / "ledger"
BRONZE_DIR = ROOT / "data" / "bronze"


def load_nfl(season: int, week: int):
    from coverline.leagues.nfl.live import RatingsSnapshotSource, load_schedule
    from coverline.leagues.nfl.model import NFLModel
    fixture = ROOT / "tests" / "fixtures" / f"nfl_{season}_week{week:02d}_schedule.csv"
    sched = pd.read_csv(fixture) if fixture.exists() else load_schedule([season])
    src = RatingsSnapshotSource.for_week(season, week, sched)
    return NFLModel(src), src


def load_cfb(season: int, week: int):
    from coverline.leagues.cfb.model import CFBModel
    from coverline.leagues.cfb.sources import CachedWalkForwardSource
    src = CachedWalkForwardSource.load()
    return CFBModel(src), src


LOADERS = {"nfl": load_nfl, "cfb": load_cfb}


def latest_snapshot(store: BronzeStore, sport: str) -> Path | None:
    rows = [r for r in store.snapshots(sport)]
    if not rows:
        return None
    return store.root / rows[-1]["path"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=sorted(LOADERS))
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--snapshot", help="a bronze odds snapshot; default is the newest")
    ap.add_argument("--market", default="spreads")
    ap.add_argument("--bankroll", type=float, required=True)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--max-fraction", type=float, default=0.02)
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--pooled", action="store_true",
                    help="use the pooled shrinkage weight instead of the robust one")
    ap.add_argument("--commit", action="store_true",
                    help="write the signals to the append-only ledger")
    args = ap.parse_args()

    log = E.load_attempts()
    weight = log.weight() if args.pooled else log.robust_weight()
    print(f"attempt log: {log.summary()}")
    if args.pooled:
        print(f"  USING POOLED WEIGHT {weight:.4f}. It rests on one attempt -- "
              f"the robust figure is {log.robust_weight():.4f}.")
    else:
        print(f"  using robust weight {weight:.4f} "
              f"(pooled would be {log.weight():.4f})")
    if weight <= 0.0:
        print("\nThe attempt log says nothing in the pipeline is distinguishable "
              "from noise. Nothing will be staked.")

    model, source = LOADERS[args.league](args.season, args.week)
    print(f"\n{args.league.upper()} {args.season} week {args.week}: "
          f"{model.__class__.__name__}, markets {list(model.primary_markets)}")

    store = BronzeStore(BRONZE_DIR)
    snap_path = Path(args.snapshot) if args.snapshot else latest_snapshot(
        store, args.league)
    if snap_path is None or not Path(snap_path).exists():
        print(f"\nNo odds snapshot for {args.league}. Capture one first:")
        print("  see src/coverline/execution/capture.py, or pass --snapshot")
        return 2

    payload = store.read_snapshot(snap_path)["payload"]
    quotes = normalize(payload, captured_at=str(snap_path))
    events = sorted({q.event_id for q in quotes})
    print(f"snapshot {Path(snap_path).name}: {len(events)} events, "
          f"{len(quotes)} quotes")

    # Resolve the book's event ids to model game ids. Without this the
    # runner has a model and a market and no way to know they describe the
    # same game.
    known = set(source.game_ids()) if hasattr(source, "game_ids") else None
    matches = match_events(quotes, season=args.season, week=args.week,
                           known_game_ids=known)
    skipped = unmatched(quotes, matches)
    print(f"matched {len(matches)} events to model games"
          + (f"; {len(skipped)} unmatched" if skipped else ""))
    if skipped:
        # Named, not counted. A slate that silently prices 6 of 16 games is
        # the shape of a problem that goes unnoticed for weeks.
        for e in skipped[:8]:
            print(f"    unmatched: {e}")

    ledger = BetLedger(LEDGER_DIR) if args.commit else None
    placed = declined = withheld = 0

    for m in matches:
        ev = m.event_id
        try:
            dist = model.predict(m.game_id, "now")
        except Exception as exc:
            print(f"  {m.game_id}: no prediction ({type(exc).__name__}) -- skipped")
            continue
        sigs = Rc.recommend(dist=dist, quotes=quotes, event_id=ev,
                            market=args.market, league=args.league,
                            bankroll=args.bankroll, shrinkage=weight,
                            ledger=ledger, kelly_multiple=args.kelly,
                            max_bankroll_fraction=args.max_fraction,
                            min_edge=args.min_edge)
        if not sigs:
            withheld += 1
            continue
        for s in sigs:
            if s.placed:
                placed += 1
                print(f"  BET  {s.selection:6} {s.line:+.1f} @ {s.book:12} "
                      f"stake {s.stake:>9,.2f}  edge {s.edge_used:+.2%}")
            else:
                declined += 1

    print(f"\n{placed} recommended, {declined} declined, "
          f"{withheld} events withheld (no priceable market)")
    if ledger is not None:
        print(f"written to {LEDGER_DIR}")
        print(f"conversion so far: {ledger.conversion()}")
    else:
        print("Dry run -- nothing written. Add --commit to record these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
