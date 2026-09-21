"""The attempt log, and the weight computed from it.

This module replaces a judgement call with arithmetic. The old ship/no-ship
significance gate has been retired (see core/staking.py for why it had no
power at these sample sizes); candidates now ship at a WEIGHT, and the weight
is b = 1 - 1/E[t^2] across every candidate ever evaluated -- failures included.

THE FAILURES ARE THE MECHANISM, NOT A COURTESY
Computing the weight from winners only inflates E[t^2] and returns a weight
that ships noise at full size, which is precisely the selection effect the
shrinkage exists to undo. So the loader refuses to compute a weight from a
filtered log, and ``drop_negative_for_demonstration`` exists only so the tests
can show how large the distortion is.

ONE ESTIMAND, STRICTLY
Only held-out paired gains against the incumbent are eligible. Training-side
coefficient t-statistics answer a different question -- "is this variable
associated with the target" rather than "does this change help on data it has
never seen" -- and this repo contains a clean demonstration of the gap: the
QB-continuity hypothesis measured t=+2.63 on train and t=-0.17 held out, same
idea, same week. Admitting train-side statistics would have produced a weight
above 0.9 from numbers that predicted nothing.

WHY THE RESULT IS A CEILING AND NOT AN ESTIMATE
The current log was reconstructed from committed artifacts because no attempt
log was kept at the time; the project recorded findings, not attempts. Whatever
was tried and never written up is missing, and what goes unwritten skews toward
nulls -- nobody writes up a failure they were not already invested in. Missing
failures bias E[t^2] up, so they bias b up. ``AttemptLog.is_ceiling`` is True
while any recovered attempt remains in the log, and callers are expected to
treat the number as an upper bound on what is defensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

#: Repo root, found by walking up from this file: src/coverline/core/ -> root.
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG = _ROOT / "evidence" / "attempts.yaml"

ELIGIBLE_ESTIMAND = "held_out_paired_gain"

VALID_DECISIONS = {
    "shipped",
    "shipped_but_contradicted",
    "not_shipped",
    "rejected",
}

REQUIRED_FIELDS = {
    "id", "date", "hypothesis", "estimand", "metric",
    "estimate", "standard_error", "t", "decision", "recovered", "source",
}


@dataclass(frozen=True)
class Attempt:
    id: str
    t: float
    estimate: float
    standard_error: float
    metric: str
    decision: str
    recovered: bool
    source: str

    @property
    def t_implied(self) -> float:
        """t recomputed from estimate and SE, so a transcription error shows."""
        return self.estimate / self.standard_error


@dataclass(frozen=True)
class AttemptLog:
    attempts: tuple[Attempt, ...]
    excluded_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.attempts)

    @property
    def t_statistics(self) -> tuple[float, ...]:
        return tuple(a.t for a in self.attempts)

    @property
    def mean_t_squared(self) -> float:
        return float(np.mean(np.asarray(self.t_statistics) ** 2))

    @property
    def rms_t(self) -> float:
        return float(np.sqrt(self.mean_t_squared))

    @property
    def is_ceiling(self) -> bool:
        """True while any attempt was recovered rather than logged at the time.

        A recovered log is missing whatever was never written up, and what goes
        unwritten skews toward nulls, so the weight it produces is an upper
        bound.
        """
        return any(a.recovered for a in self.attempts)

    @property
    def n_recovered(self) -> int:
        return sum(1 for a in self.attempts if a.recovered)

    @property
    def n_negative(self) -> int:
        """Attempts that came back at or below zero. If this is zero, suspect
        the log rather than the pipeline."""
        return sum(1 for a in self.attempts if a.t <= 0)

    def weight(self) -> float:
        """b = 1 - 1/E[t^2], floored at 0.

        A value of 0 is a real answer, not a failure: it means the candidates
        collectively carry no more signal than noise would, and nothing in the
        pipeline should ship at any weight until that changes.
        """
        if not self.attempts:
            raise ValueError(
                "cannot compute a weight from an empty attempt log; the whole "
                "point is that it contains every candidate, including failures"
            )
        m = self.mean_t_squared
        return 0.0 if m <= 1.0 else 1.0 - 1.0 / m

    def leave_one_out_weights(self) -> dict[str, float]:
        """The weight recomputed with each attempt removed in turn.

        A weight that moves a lot when one row is dropped is a weight built on
        that row, and sizing real money from it is sizing from one experiment.
        """
        out: dict[str, float] = {}
        for a in self.attempts:
            rest = tuple(x for x in self.attempts if x.id != a.id)
            if not rest:
                continue
            m = float(np.mean(np.asarray([r.t for r in rest]) ** 2))
            out[a.id] = 0.0 if m <= 1.0 else 1.0 - 1.0 / m
        return out

    @property
    def is_dominated_by_one(self) -> bool:
        """True when removing a single attempt moves the weight by >0.10.

        Not a failure -- an outsized effect can be real, and this project's
        largest is. It is a statement that the pooled weight is resting on one
        observation, which the caller should know before staking on it.
        """
        loo = self.leave_one_out_weights()
        if not loo:
            return False
        return max(abs(w - self.weight()) for w in loo.values()) > 0.10

    def robust_weight(self) -> float:
        """The most conservative weight consistent with dropping any one row.

        Prefer this over weight() for sizing. The pooled weight is the best
        point estimate; this one survives the loss of the single most
        influential observation, which matters when n is small enough that one
        experiment can move the answer by 30 points -- as it currently can.
        """
        loo = self.leave_one_out_weights()
        return min([self.weight(), *loo.values()]) if loo else self.weight()

    def summary(self) -> str:
        w = self.weight()
        kind = "CEILING (log contains recovered attempts)" if self.is_ceiling else "estimate"
        line = (
            f"{len(self)} attempts ({self.n_negative} at or below zero, "
            f"{self.n_recovered} recovered), RMS t = {self.rms_t:.3f}, "
            f"E[t^2] = {self.mean_t_squared:.4f}, weight b = {w:.4f} -- {kind}"
        )
        if self.is_dominated_by_one:
            line += (f"; DOMINATED BY ONE ATTEMPT -- robust weight "
                     f"{self.robust_weight():.4f}")
        return line


def load_attempts(path: Path | str = DEFAULT_LOG) -> AttemptLog:
    """Read and validate the log. Raises rather than guessing on anything odd."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no attempt log at {path}. Without it there is no defensible "
            "shrinkage weight, and sizing must not proceed on a guessed one."
        )
    doc = yaml.safe_load(path.read_text())
    rows = doc.get("attempts") or []

    attempts: list[Attempt] = []
    for row in rows:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"attempt {row.get('id', '?')!r} missing {sorted(missing)}")
        if row["estimand"] != ELIGIBLE_ESTIMAND:
            raise ValueError(
                f"attempt {row['id']!r} has estimand {row['estimand']!r}; only "
                f"{ELIGIBLE_ESTIMAND!r} is eligible. Record it under 'excluded' "
                "with a reason instead of mixing estimands."
            )
        if row["decision"] not in VALID_DECISIONS:
            raise ValueError(
                f"attempt {row['id']!r} decision {row['decision']!r} not in "
                f"{sorted(VALID_DECISIONS)}"
            )
        if row["standard_error"] <= 0:
            raise ValueError(f"attempt {row['id']!r} has a non-positive standard error")

        a = Attempt(
            id=row["id"], t=float(row["t"]), estimate=float(row["estimate"]),
            standard_error=float(row["standard_error"]), metric=row["metric"],
            decision=row["decision"], recovered=bool(row["recovered"]),
            source=row["source"],
        )
        # Catch a transcribed t that does not match its own estimate and SE.
        if abs(a.t - a.t_implied) > 0.02:
            raise ValueError(
                f"attempt {a.id!r}: recorded t={a.t} but estimate/SE implies "
                f"{a.t_implied:.4f}. One of the three numbers is wrong."
            )
        attempts.append(a)

    ids = [a.id for a in attempts]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate attempt ids: {sorted(dupes)}")

    excluded = tuple(e["id"] for e in (doc.get("excluded") or []))
    return AttemptLog(attempts=tuple(attempts), excluded_ids=excluded)


def current_weight(path: Path | str = DEFAULT_LOG) -> float:
    """The weight to pass to core.staking.size_bet.

    Deliberately a function call rather than a module constant. A constant
    would be importable without reading the log, and the whole discipline is
    that the weight is derived, dated and auditable rather than chosen.
    """
    return load_attempts(path).weight()


def drop_negative_for_demonstration(log: AttemptLog) -> AttemptLog:
    """Return the log with non-positive attempts removed.

    EXISTS ONLY TO BE MEASURED AGAINST. Never call this in production code; it
    is the cherry-picked log whose distortion the tests quantify.
    """
    return AttemptLog(
        attempts=tuple(a for a in log.attempts if a.t > 0),
        excluded_ids=log.excluded_ids,
    )
