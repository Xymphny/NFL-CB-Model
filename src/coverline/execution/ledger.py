"""The bet ledger: what was recommended, what was placed, and what it cost.

WHY A SIGNAL THAT WAS NOT BET IS A RECORD WORTH KEEPING
A ledger of placed bets can tell you how the bets you made performed. It
cannot tell you what your execution cost, because the bets you did not get --
the line moved, the book limited you, you were asleep -- leave no trace. The
difference between the model's recommendations and the positions actually held
IS the execution layer's performance, and it is invisible unless the
unconverted signals are written down with a reason.

So every recommendation is recorded, whether or not it became a bet. That is
the single design decision this module exists around.

APPEND-ONLY, BECAUSE A PLACED BET IS A FACT
Records are appended and never rewritten. A bet's later life -- the closing
price, the settlement -- arrives as separate rows keyed to it, not as edits.
Rewriting history in a betting ledger is how a losing week becomes a
break-even one without anybody deciding to lie.

REFUSES TO RECORD WHAT IT CANNOT LATER GRADE
A bet without the fields CLV needs is a bet that can never be evaluated, and
discovering that months later is worse than being stopped now. `place()`
requires the line, the price, the book, the stake, and the model probability
it was staked on. It also requires the SHRUNK probability actually bet on and
the weight used, because a log that records only the raw model number
describes a bet that was never placed (see core/staking.py).

WHAT IS DELIBERATELY NOT STORED
No account identifiers, no credentials, no book logins. Book NAMES yes,
account references never. This file sits in a git repo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from coverline.core import pricing as P

Disposition = Literal["placed", "not_placed"]

#: Why a recommendation did not become a bet. A closed set, because "other"
#: with free text is how the most common reason becomes unqueryable.
NOT_PLACED_REASONS = frozenset({
    "line_moved",          # the price was gone by the time it was actioned
    "below_threshold",     # edge did not clear the floor after shrinking
    "limited",             # the book would not take it, or not at size
    "missed",              # nobody was there; the window passed
    "bankroll",            # sizing said yes, available funds said no
    "declined",            # a deliberate pass
    "no_price",            # no book offered the market
})

SETTLEMENTS = frozenset({"win", "loss", "push", "void"})
SIDES = frozenset({"home", "away", "over", "under"})
OUTCOMES = frozenset({"win", "loss", "push"})


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Signal:
    """A recommendation, placed or not.

    The model fields are required even when the bet was not placed: a signal
    that cannot be priced in hindsight cannot contribute to the execution-cost
    measurement it exists for.
    """

    signal_id: str
    at: str
    league: str
    event_id: str
    market: str
    selection: str
    line: float | None

    # what the model believed
    p_model: float
    p_market: float
    p_used: float
    shrinkage: float
    edge_claimed: float
    edge_used: float

    disposition: Disposition
    #: P(the bet voids) under the model at the time: the push on a spread,
    #: the tie on a moneyline. Recorded because it is the quantity that
    #: caused the largest pricing error this project has found -- MLB
    #: moneylines were out by 5.9 points because 10.8% of the mass sat on a
    #: tied margin and nothing conditioned it out -- and because a graded bet
    #: whose record does not carry it cannot be re-checked for that error.
    #:
    #: Optional with a None default so rows written before this field existed
    #: still load. None means "not recorded", which is different from 0.0.
    push_probability: float | None = None
    not_placed_reason: str | None = None

    # which game and which side -- what settlement needs. Optional so rows
    # written before they existed still load; a row without them can be
    # closed but never settled, and settle_outcomes reports it as such.
    game_id: str | None = None
    commence_time: str | None = None
    #: "home", "away", "over" or "under". The selection is a feed name, and
    #: working out which side a name was after the fact is exactly how the
    #: away side was once mispriced (ADR 0015).
    side: str | None = None

    # execution, present only when placed
    book: str | None = None
    price_decimal: float | None = None
    stake: float | None = None
    bankroll_at_placement: float | None = None
    best_available_decimal: float | None = None
    books_surveyed: int | None = None

    notes: str = ""

    @property
    def placed(self) -> bool:
        return self.disposition == "placed"

    @property
    def price_slippage_cents(self) -> float | None:
        """American cents left on the table against the best price seen.

        The direct measure of shopping quality. None when nothing was
        surveyed, which is itself worth knowing.
        """
        if self.price_decimal is None or self.best_available_decimal is None:
            return None
        return (P.decimal_to_american(self.best_available_decimal)
                - P.decimal_to_american(self.price_decimal))


@dataclass(frozen=True)
class Close:
    """The closing market for a signal, recorded after the fact."""

    signal_id: str
    at: str
    close_decimals: list[float]
    outcome_index: int
    close_line: float | None
    market_has_sharp_close: bool = True
    devig_method: str = "power"


@dataclass(frozen=True)
class Outcome:
    """How a signal's bet resolved, placed or not.

    Separate from Settlement on purpose: a settlement is money (it carries a
    PnL and exists only for placed bets), an outcome is a fact about the game
    and exists for every signal that can be tied to one. Paper trading is
    graded on outcomes; the bankroll on settlements.
    """

    signal_id: str
    at: str
    result: str               # win | loss | push
    home_score: int
    away_score: int


@dataclass(frozen=True)
class Settlement:
    signal_id: str
    at: str
    result: str
    pnl: float


@dataclass
class BetLedger:
    """Append-only JSONL, one file per record type."""

    root: Path
    #: signal_id sets per file, keyed by path, each with the file's (size,
    #: mtime) when it was built. Uniqueness checks used to re-read the whole
    #: file on every append -- quadratic, and a season of paper trades writes
    #: tens of thousands of rows. A file changed by anyone else (a different
    #: process, a git pull) no longer matches its signature and is re-read.
    _index: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @staticmethod
    def _sig(path: Path) -> tuple[int, int] | None:
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _ids(self, path: Path) -> set[str]:
        sig = self._sig(path)
        hit = self._index.get(path)
        if hit is None or hit[0] != sig:
            ids = {r["signal_id"] for r in self._read(path)}
            self._index[path] = (sig, ids)
            return ids
        return hit[1]

    def _append_indexed(self, path: Path, row: dict) -> None:
        ids = self._ids(path)
        self._append(path, row)
        ids.add(row["signal_id"])
        self._index[path] = (self._sig(path), ids)

    # -- paths ------------------------------------------------------------

    @property
    def signals_path(self) -> Path:
        return self.root / "signals.jsonl"

    @property
    def closes_path(self) -> Path:
        return self.root / "closes.jsonl"

    @property
    def outcomes_path(self) -> Path:
        return self.root / "outcomes.jsonl"

    @property
    def settlements_path(self) -> Path:
        return self.root / "settlements.jsonl"

    # -- writing ----------------------------------------------------------

    def record(self, signal: Signal) -> Signal:
        self._validate(signal)
        if signal.signal_id in self._ids(self.signals_path):
            raise ValueError(
                f"signal {signal.signal_id!r} is already recorded. The ledger "
                "is append-only: a correction is a new row, not an edit."
            )
        self._append_indexed(self.signals_path, asdict(signal))
        return signal

    def record_close(self, close: Close) -> None:
        if close.signal_id not in self._ids(self.signals_path):
            raise KeyError(f"no signal {close.signal_id!r} to attach a close to")
        if len(close.close_decimals) < 2:
            raise ValueError(
                "a close needs the full market: the margin cannot be removed "
                "from one side, and CLV against a vigged price is not CLV"
            )
        self._append(self.closes_path, asdict(close))

    def record_outcome(self, o: Outcome) -> None:
        if o.result not in OUTCOMES:
            raise ValueError(f"result must be one of {sorted(OUTCOMES)}")
        if o.signal_id not in self._ids(self.signals_path):
            raise KeyError(f"no signal {o.signal_id!r} to record an outcome for")
        if o.signal_id in self._ids(self.outcomes_path):
            raise ValueError(
                f"{o.signal_id!r} already has an outcome. A game resolves once; "
                "a second row would be two versions of one result.")
        self._append_indexed(self.outcomes_path, asdict(o))

    def record_settlement(self, s: Settlement) -> None:
        if s.result not in SETTLEMENTS:
            raise ValueError(f"result must be one of {sorted(SETTLEMENTS)}")
        known = {x.signal_id for x in self.signals() if x.placed}
        if s.signal_id not in known:
            raise KeyError(
                f"{s.signal_id!r} is not a PLACED signal; only placed bets settle"
            )
        self._append(self.settlements_path, asdict(s))

    # -- reading ----------------------------------------------------------

    def signals(self) -> list[Signal]:
        return [Signal(**r) for r in self._read(self.signals_path)]

    def closes(self) -> list[Close]:
        return [Close(**r) for r in self._read(self.closes_path)]

    def outcomes(self) -> list[Outcome]:
        return [Outcome(**r) for r in self._read(self.outcomes_path)]

    def settlements(self) -> list[Settlement]:
        return [Settlement(**r) for r in self._read(self.settlements_path)]

    def clv(self, signal_id: str) -> P.ClosingLineValue | None:
        """Devigged CLV for one signal, or None if no close is recorded yet."""
        sig = next((s for s in self.signals() if s.signal_id == signal_id), None)
        close = next((c for c in self.closes() if c.signal_id == signal_id), None)
        return self.clv_of(sig, close)

    @staticmethod
    def clv_of(sig: "Signal | None", close: "Close | None") -> P.ClosingLineValue | None:
        """CLV from a signal and its close, for callers holding both -- a
        summary over thousands of rows must not rescan the files per row."""
        if sig is None or close is None or sig.price_decimal is None:
            return None
        return P.closing_line_value(
            bet_decimal=sig.price_decimal,
            close_decimals=close.close_decimals,
            outcome_index=close.outcome_index,
            bet_line=sig.line, close_line=close.close_line,
            method=close.devig_method,
            market_has_sharp_close=close.market_has_sharp_close,
        )

    # -- diagnostics ------------------------------------------------------

    def conversion(self) -> dict[str, Any]:
        """How many recommendations became bets, and why the rest did not.

        This is the execution layer's scoreboard. A high edge that never
        converts is worth exactly nothing, and only this view shows it.
        """
        sigs = self.signals()
        if not sigs:
            return {"signals": 0, "placed": 0, "rate": None, "reasons": {}}
        reasons: dict[str, int] = {}
        for s in sigs:
            if not s.placed and s.not_placed_reason:
                reasons[s.not_placed_reason] = reasons.get(s.not_placed_reason, 0) + 1
        placed = sum(1 for s in sigs if s.placed)
        return {"signals": len(sigs), "placed": placed,
                "rate": round(placed / len(sigs), 4), "reasons": reasons}

    def execution_cost(self) -> dict[str, Any]:
        """Cents given up against the best price seen, across placed bets."""
        vals = [s.price_slippage_cents for s in self.signals() if s.placed]
        measured = [v for v in vals if v is not None]
        return {
            "placed": len(vals),
            "measurable": len(measured),
            "unmeasured": len(vals) - len(measured),
            "mean_cents_given_up": (round(sum(measured) / len(measured), 3)
                                    if measured else None),
        }

    # -- internals --------------------------------------------------------

    @staticmethod
    def _validate(s: Signal) -> None:
        for name, p in (("p_model", s.p_model), ("p_market", s.p_market),
                        ("p_used", s.p_used)):
            if not 0.0 < p < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if not 0.0 <= s.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if s.side is not None and s.side not in SIDES:
            raise ValueError(f"side must be one of {sorted(SIDES)}; got {s.side!r}")

        if s.disposition == "placed":
            missing = [n for n, v in (("book", s.book),
                                      ("price_decimal", s.price_decimal),
                                      ("stake", s.stake)) if v is None]
            if missing:
                raise ValueError(
                    f"a placed bet needs {missing} or it can never be graded"
                )
            if s.price_decimal is not None and s.price_decimal <= 1.0:
                raise ValueError("decimal price must exceed 1.0")
            if s.stake is not None and s.stake <= 0:
                raise ValueError("a placed bet needs a positive stake")
        elif s.disposition == "not_placed":
            if s.not_placed_reason not in NOT_PLACED_REASONS:
                raise ValueError(
                    f"not_placed_reason must be one of "
                    f"{sorted(NOT_PLACED_REASONS)}; got {s.not_placed_reason!r}. "
                    "Free text is how the commonest reason becomes unqueryable."
                )
            if s.stake is not None:
                raise ValueError("an unplaced signal cannot carry a stake")
        else:
            raise ValueError(f"unknown disposition {s.disposition!r}")

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    @staticmethod
    def _read(path: Path) -> Iterator[dict]:
        if not Path(path).exists():
            return iter(())
        return (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())
