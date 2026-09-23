#!/usr/bin/env python3
"""Close and settle every recorded signal, then (optionally) regrade.

    python3 scripts/settle_ledger.py                 # all leagues
    python3 scripts/settle_ledger.py --league nba
    python3 scripts/settle_ledger.py --grade         # and rewrite market_weights.json

THE LOOP
  recommend_slate.py --commit   writes a signal per side per book, placed or paper
  capture job                   stores the market up to each game's start
  this, step 1 (grade)          attaches the last pre-start snapshot as the close
  this, step 2 (settle)         attaches the final score as an outcome
  this, --grade                 re-estimates each league's weight against the
                                market, and its tier bands (ADR 0025), from the
                                ledger once it has 150 games

Every league paper-trades until step 3 says otherwise (ADR 0024). This is the
only path by which that changes, which is why it exists before there is
anything in the ledger to settle.

Appends only. Running it twice closes and settles nothing twice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.execution import grade as G  # noqa: E402
from coverline.execution import settle as S  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402

LEDGER_DIR = ROOT / "data" / "ledger"
BRONZE_DIR = ROOT / "data" / "bronze"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=sorted(S.VENDOR), action="append")
    ap.add_argument("--ledger", default=str(LEDGER_DIR))
    ap.add_argument("--bronze", default=str(BRONZE_DIR))
    ap.add_argument("--grade", action="store_true",
                    help="re-estimate the market weights afterwards")
    a = ap.parse_args(argv)

    ledger = BetLedger(Path(a.ledger))
    store = BronzeStore(Path(a.bronze))
    failed = []
    for league in a.league or sorted(S.VENDOR):
        closes = G.grade(ledger, store, sport=S.VENDOR[league])
        print(f"{league}: closes -- {closes.summary()}"
              + (f" {closes.reasons()}" if closes.ungraded else ""))
        if league not in S.FINALS:
            print(f"{league}: outcomes -- no finals source (see settle.py)")
            continue
        try:
            finals = S.FINALS[league]()
        except Exception as exc:        # one league's feed must not stop the rest
            failed.append(league)
            print(f"{league}: outcomes -- finals unavailable: {type(exc).__name__}: {exc}")
            continue
        print(f"{league}: outcomes -- {S.settle(ledger, finals, league).summary()}")

    for label, paper in (("placed", False), ("paper", True)):
        c = G.clv_summary(ledger, paper=paper)
        print(f"CLV {label}: {c['graded_valid']} graded, mean "
              f"{c['mean_clv_probability_points']} prob points, "
              f"{c['mean_line_points']} line points")

    if a.grade:
        from model import derive_tier_thresholds, grade_market_weight
        grade_market_weight.main(ledger_dir=Path(a.ledger))
        derive_tier_thresholds.main(ledger_dir=Path(a.ledger))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
