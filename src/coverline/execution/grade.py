"""Attach captured closes to recorded signals, and compute CLV.

THE LOOP THIS CLOSES
Everything upstream produces one half of a comparison. The ledger holds what
was recommended and what was bet; bronze holds what the market closed at. Until
something joins them, CLV -- the metric this whole evaluation approach rests on
-- is computable in principle and computed never.

WHICH SNAPSHOT IS "THE CLOSE"
The last one captured strictly BEFORE the game starts. Not the latest
available: a snapshot taken after kickoff carries an in-play price, which is a
different instrument, and using it would quietly inflate or deflate every CLV
figure depending on which way the game went. `close_for` refuses to use one.

WHAT IT WILL NOT DO
It will not invent a close. A signal whose event never appeared in a captured
snapshot stays ungraded and is REPORTED as ungraded, because an ungraded bet
and a break-even one are different facts and a summary that merges them is
worse than a shorter one.

It will not re-grade. Closes are append-only like everything else in the
ledger: a second close for one signal would mean two versions of what the
market did, and picking one silently is how a record stops being a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from coverline.core import pricing as P
from coverline.execution.bronze import BronzeStore
from coverline.execution.ledger import BetLedger, Close, Signal
from coverline.execution.normalize import Quote, normalize, two_sided

#: Markets with no market-making book behind them. CLV against these is, in a
#: practitioner's words, at best an informed guess -- so it is recorded as
#: invalid rather than as a number that averages in with the real ones.
NO_SHARP_CLOSE = ("player_", "props", "_prop")


def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def has_sharp_close(market: str) -> bool:
    return not any(tag in market.lower() for tag in NO_SHARP_CLOSE)


@dataclass(frozen=True)
class CloseCandidate:
    captured_at: str
    quotes: tuple[Quote, ...]


def snapshots_for(store: BronzeStore, sport: str) -> list[CloseCandidate]:
    """Every captured snapshot for a sport, oldest first."""
    out: list[CloseCandidate] = []
    for row in store.snapshots(sport):
        path = store.root / row["path"]
        if not path.exists():
            continue
        rec = store.read_snapshot(path)
        out.append(CloseCandidate(
            captured_at=rec["_meta"]["captured_at"],
            quotes=tuple(normalize(rec["payload"],
                                   captured_at=rec["_meta"]["captured_at"])),
        ))
    return sorted(out, key=lambda c: c.captured_at)


def close_for(candidates: Sequence[CloseCandidate], *, event_id: str,
              commence_time: str | None) -> CloseCandidate | None:
    """The last snapshot strictly before kickoff that contains the event.

    Returns None rather than falling back to a later one. A post-kickoff
    snapshot is an in-play price, and using it would move every CLV figure in
    whichever direction the game happened to go -- a bias that looks like
    skill.
    """
    cutoff = _parse(commence_time) if commence_time else None
    best: CloseCandidate | None = None
    for c in candidates:
        if cutoff is not None and _parse(c.captured_at) >= cutoff:
            continue
        if any(q.event_id == event_id for q in c.quotes):
            best = c
    return best


@dataclass
class GradeReport:
    graded: list[str]
    ungraded: list[tuple[str, str]]
    already_had: list[str]

    def summary(self) -> str:
        return (f"{len(self.graded)} graded, {len(self.already_had)} already "
                f"closed, {len(self.ungraded)} ungraded")

    def reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, why in self.ungraded:
            out[why] = out.get(why, 0) + 1
        return out


def grade(ledger: BetLedger, store: BronzeStore, *, sport: str,
          commence_times: dict[str, str] | None = None,
          devig_method: P.Method = "power",
          include_paper: bool = True) -> GradeReport:
    """Attach a close to every priced signal that does not have one.

    Placed bets and paper trades alike. The kickoff comes from
    `commence_times` when given, else from the signal itself.
    """
    commence_times = commence_times or {}
    candidates = snapshots_for(store, sport)
    closed = {c.signal_id for c in ledger.closes()}
    report = GradeReport(graded=[], ungraded=[], already_had=[])

    for sig in ledger.signals():
        # PAPER TRADES ARE GRADED TOO. Every league paper-trades until a grade
        # against the market clears (ADR 0024), and that grade is built from
        # exactly these closes. An unplaced row with no book is an older row
        # or a non-candidate (e.g. line_moved) and has no market to close.
        if not sig.placed and (not include_paper or sig.book is None):
            continue
        if sig.signal_id in closed:
            report.already_had.append(sig.signal_id)
            continue

        cand = close_for(candidates, event_id=sig.event_id,
                         commence_time=commence_times.get(sig.event_id,
                                                          sig.commence_time))
        if cand is None:
            report.ungraded.append((sig.signal_id, "no_pregame_snapshot"))
            continue

        sides = two_sided(cand.quotes, event_id=sig.event_id,
                          market=sig.market, bookmaker=sig.book or "")
        if not sides:
            report.ungraded.append((sig.signal_id, "market_not_in_snapshot"))
            continue

        idx = next((i for i, q in enumerate(sides) if q.outcome == sig.selection),
                   None)
        if idx is None:
            report.ungraded.append((sig.signal_id, "selection_not_in_snapshot"))
            continue

        ledger.record_close(Close(
            signal_id=sig.signal_id, at=cand.captured_at,
            close_decimals=[q.price_decimal for q in sides],
            outcome_index=idx, close_line=sides[idx].point,
            market_has_sharp_close=has_sharp_close(sig.market),
            devig_method=devig_method,
        ))
        report.graded.append(sig.signal_id)

    return report


def clv_summary(ledger: BetLedger, paper: bool = False) -> dict[str, Any]:
    """CLV across graded bets, with the invalid ones kept separate.

    Prop closes are not sharp forecasts, so averaging them in with real ones
    would produce a number that is neither. They are counted, not merged.

    `paper=True` summarises the UNPLACED priced signals instead -- what the
    model would have earned against the close had it bet. Never merged with
    the placed figure: one is a record of money, the other a measurement.
    """
    valid, invalid, ungraded = [], 0, 0
    naive_gap: list[float] = []
    points: list[float] = []
    line_moves: list[float] = []

    closes = {c.signal_id: c for c in ledger.closes()}
    for sig in ledger.signals():
        if sig.placed == paper or sig.price_decimal is None:
            continue
        # Paper rows record BOTH sides of every market at every book, and
        # their CLVs cancel to about zero by construction. The paper figure is
        # the side the model preferred: positive claimed edge.
        if paper and sig.edge_claimed <= 0:
            continue
        clv = ledger.clv_of(sig, closes.get(sig.signal_id))
        if clv is None:
            ungraded += 1
            continue
        if not clv.valid:
            invalid += 1
            continue
        valid.append(clv.ev_pct)
        points.append(clv.prob_points)
        if clv.line_points is not None:
            line_moves.append(clv.line_points)
        naive_gap.append(clv.devig_overstatement)

    n = len(valid)
    return {
        "graded_valid": n,
        "graded_invalid_no_sharp_close": invalid,
        "ungraded": ungraded,
        "mean_clv_devigged": round(sum(valid) / n, 5) if n else None,
        "mean_clv_probability_points": (round(sum(points) / n, 5)
                                        if n else None),
        "mean_overstatement_if_naive": (round(sum(naive_gap) / n, 5)
                                        if n else None),
        # Spread and total CLV mostly arrives as a moved LINE at an unchanged
        # price, which prob points at the bet's own outcome cannot see.
        "mean_line_points": (round(sum(line_moves) / len(line_moves), 4)
                             if line_moves else None),
        "note": ("mean_clv_devigged is EV against the fair close. The naive "
                 "figure -- against the posted close -- is higher by the hold, "
                 "and is what a CLV number means when nobody says which one "
                 "they computed. NO MEAN IN AMERICAN CENTS IS REPORTED: that "
                 "scale is discontinuous at even money and non-linear "
                 "elsewhere, so 2.4 probability points can read as 210 cents "
                 "while 1.1 points reads as 5. mean_clv_probability_points is "
                 "the aggregate that means something."),
    }
