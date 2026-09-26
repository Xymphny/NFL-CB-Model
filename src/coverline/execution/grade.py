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
from datetime import datetime, timedelta, timezone
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


def _utcnow() -> datetime:
    """The clock grade() closes against; a seam for tests."""
    return datetime.now(timezone.utc)


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
        # The daily EARLY poll is a price to be graded, never a close: if a
        # close capture were missed, falling back to the early snapshot would
        # grade an early trade against itself and report zero CLV as a result.
        if row.get("kind") == "early":
            continue
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
          include_paper: bool = True, now: str | None = None) -> GradeReport:
    """Attach a close to every priced signal that does not have one.

    Placed bets and paper trades alike. The kickoff comes from
    `commence_times` when given, else from the signal itself.

    ONLY ONCE THE GAME HAS STARTED. Every snapshot carries every upcoming
    game, so before kickoff "the last snapshot before kickoff" is merely the
    last snapshot SO FAR: Thursday night's CFB close window holds Saturday's
    games, and a Friday settle run would have fixed it as their close --
    append-only, so for good. A game not yet started is left for a later run.
    A signal with no kickoff at all is ungraded rather than closed on the
    newest snapshot, which could be an in-play price.
    """
    commence_times = commence_times or {}
    t_now = _parse(now) if now else _utcnow()
    from coverline.execution.settle import VENDOR
    mine = {sport} | {lg for lg, v in VENDOR.items() if v == sport}
    candidates = snapshots_for(store, sport)
    closed = {c.signal_id for c in ledger.closes()}
    report = GradeReport(graded=[], ungraded=[], already_had=[])

    for sig in ledger.signals():
        if sig.league not in mine:
            continue                      # another league's pass closes it
        # PAPER TRADES ARE GRADED TOO. Every league paper-trades until a grade
        # against the market clears (ADR 0024), and that grade is built from
        # exactly these closes. An unplaced row with no book is an older row
        # or a non-candidate (e.g. line_moved) and has no market to close.
        if not sig.placed and (not include_paper or sig.book is None):
            continue
        if sig.signal_id in closed:
            report.already_had.append(sig.signal_id)
            continue

        kickoff = commence_times.get(sig.event_id, sig.commence_time)
        if not kickoff:
            report.ungraded.append((sig.signal_id, "no_kickoff_time"))
            continue
        if _parse(kickoff) > t_now:
            report.ungraded.append((sig.signal_id, "not_started"))
            continue
        cand = close_for(candidates, event_id=sig.event_id, commence_time=kickoff)
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


def line_aware_clv(sig: Signal, close: Close,
                   clv: P.ClosingLineValue) -> tuple[float, float, bool, float]:
    """(prob_points, ev_pct, line_adjusted, market_move) for one signal.

    When the line moved, the close's fair probability is moved to the bet's
    line along the league's margin distribution (execution/line_clv.py);
    otherwise the price-only figures stand, which are right when it did not.

    prob_points is AFTER THE VIG: fair close minus the bet's implied price, so
    a pick whose market never moved scores minus the hold (about -2.4 at
    -110). market_move is the part that measures the model: the same book's
    devigged close, at the bet's line, minus its devigged price when the
    trade was made (Signal.p_market). Zero for no movement, positive when the
    market came to the pick.
    """
    from coverline.execution.line_clv import fair_at_bet_line
    fair_close = float(P.devig(close.close_decimals, close.devig_method)[close.outcome_index])
    fair = fair_at_bet_line(league=sig.league, side=sig.side, fair_close=fair_close,
                            close_line=close.close_line, bet_line=sig.line)
    if fair is None:
        same_line = close.close_line is None or sig.line is None or \
            float(close.close_line) == float(sig.line)
        move = (fair_close - sig.p_market) if same_line else float("nan")
        return clv.prob_points, clv.ev_pct, False, move
    return (fair - float(P.decimal_to_implied(sig.price_decimal)),
            P.expected_value(fair, sig.price_decimal), True, fair - sig.p_market)


#: A paper trade priced this close to its close IS the close, so its CLV is
#: zero by construction. The capture job paper-trades from both the daily
#: early poll and the closing poll; only the first measures anything.
MIN_LEAD_MINUTES = 30


def priced_before_close(sig, close, minutes: int = MIN_LEAD_MINUTES) -> bool:
    """True when the signal was priced at least `minutes` before its close."""
    if close is None or not getattr(close, "at", None) or not sig.at:
        return False
    return _parse(close.at) - _parse(sig.at) >= timedelta(minutes=minutes)


def clv_summary(ledger: BetLedger, paper: bool = False) -> dict[str, Any]:
    """CLV across graded bets, with the invalid ones kept separate.

    Prop closes are not sharp forecasts, so averaging them in with real ones
    would produce a number that is neither. They are counted, not merged.

    `paper=True` summarises the UNPLACED priced signals instead -- what the
    model would have earned against the close had it bet. Never merged with
    the placed figure: one is a record of money, the other a measurement.
    """
    valid, invalid, ungraded, adjusted = [], 0, 0, 0
    moves: list[float] = []
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
        if paper and not priced_before_close(sig, closes.get(sig.signal_id)):
            continue                      # priced at the close: CLV 0 by construction
        clv = ledger.clv_of(sig, closes.get(sig.signal_id))
        if clv is None:
            ungraded += 1
            continue
        if not clv.valid:
            invalid += 1
            continue
        pts, ev, moved, move = line_aware_clv(sig, closes[sig.signal_id], clv)
        adjusted += moved
        if move == move:                          # NaN: a moved line with no family
            moves.append(move)
        valid.append(ev)
        points.append(pts)
        if clv.line_points is not None:
            line_moves.append(clv.line_points)
        naive_gap.append(clv.devig_overstatement)

    n = len(valid)
    return {
        "graded_valid": n,
        "graded_invalid_no_sharp_close": invalid,
        "ungraded": ungraded,
        "mean_clv_devigged": round(sum(valid) / n, 5) if n else None,
        # LINE-AWARE: a moved line is valued along the league's margin
        # distribution, so a spread that beat the close by a point is no
        # longer reported as minus the vig (execution/line_clv.py).
        "mean_clv_probability_points": (round(sum(points) / n, 5)
                                        if n else None),
        "line_adjusted": adjusted,
        # The vig-free part: did the market move to the pick? Zero means the
        # close agreed with the price the trade was made at.
        "mean_market_move_points": (round(sum(moves) / len(moves), 5) if moves else None),
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
