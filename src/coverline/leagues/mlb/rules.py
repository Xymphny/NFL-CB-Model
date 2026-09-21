"""The two rules that turn nine innings of scoring into an MLB final score.

WHY THIS FILE EXISTS
ADR 0017 withheld the MLB moneyline on a measured 2.5 point bias and named the
repair. This is the repair. Two league rules produce that bias and neither is
expressible by a symmetric pair of count distributions:

  1. THE HOME TEAM STOPS BATTING WHEN IT LEADS. It never gets a bottom of the
     ninth while ahead. Measured across 12,146 games, it bats in the ninth in
     only 54.8% of them, and in ZERO games did it bat while leading after the
     top of the ninth. The rule is deterministic, not statistical.

  2. EXTRA INNINGS RESOLVE EVERY GAME, and the home team bats last in each of
     them. Tied after the top of the ninth it wins 62.8% of the time, on
     1,149 games. That single number is what a symmetric model cannot say and
     what conditioning a tie out proportionally keeps getting wrong.

THE STATE THE RULE READS
Not the final score, and not a nine-inning score: the margin after the TOP of
the ninth, home runs through eight minus away runs through nine. Everything
downstream of that point is the rules layer; everything before it is scoring.

THE TRAP THIS CLASS EXISTS TO AVOID
`exp_home` in the shipped model is fitted to OBSERVED home runs, which are
truncated in 45% of games. Feeding those rates into a layer that then applies
the truncation again counts it twice, and the result looks BETTER calibrated
than it is -- the worst failure mode available, because it hides itself. So
this class takes SCALE FACTORS, measured separately, that convert observed
rates into the eight-and-a-half-inning quantities the rule actually acts on.
The away scale is not 1.0 either: extra innings give the away team a tenth,
so observed away runs overstate its nine-inning rate by about 3%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import stats

#: Runs per side considered when composing. MLB teams have scored more, never
#: with mass worth carrying at these rates.
MAX_RUNS = 24

#: States at or beyond this are treated as the edge row. A lead of five after
#: the top of the ninth ends the game exactly as a lead of one does, and a
#: deficit that large is not coming back often enough to estimate finely.
MAX_STATE = 4


@dataclass(frozen=True)
class NinthInningLayer:
    """What happens after the top of the ninth, conditional on the state.

    `table` maps a state -- home runs through eight minus away runs through
    nine -- to a distribution over (home runs gained, away runs gained). The
    rows at +1 and beyond are (0, 0) with probability one and are not fitted;
    the league fixes them.

    Framing it as GAINS rather than as a final margin is what keeps this a
    joint distribution rather than a margin correction. A total priced off a
    margin-only fix would be quietly wrong.
    """

    table: Mapping[int, Mapping[tuple[int, int], float]]
    max_state: int = MAX_STATE

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("the ninth-inning layer has no measured table")
        for state, row in self.table.items():
            total = sum(row.values())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"ninth-inning row for state {state} sums to {total}, not 1"
                )
            if state >= 1 and row != {(0, 0): 1.0}:
                raise ValueError(
                    f"state {state} must end the game deterministically; the "
                    "home team does not bat while leading and a row that says "
                    "otherwise is describing a game that cannot happen"
                )
            # NO CELL MAY PRODUCE A TIE. Extra innings run until somebody
            # wins, so a final margin of zero is not merely unlikely, it is
            # impossible -- and a table that permits one will leak that mass
            # into every moneyline priced from it. Caught a hand-built table
            # the first time it ran.
            ties = [(i, j) for (i, j) in row if state + i - j == 0]
            if ties:
                raise ValueError(
                    f"state {state} has cells {ties} that finish level. Extra "
                    "innings resolve every game, so no row may end at a "
                    "margin of zero."
                )

    def row(self, state: int) -> Mapping[tuple[int, int], float]:
        """The gain distribution for a state, clipped to the measured range.

        CLIPPING CANNOT BE NAIVE, and the first version was. A deficit of
        five borrows the row measured at four, whose cells were validated
        against four -- so a cell that took a four-run deficit to a one-run
        WIN takes a five-run deficit to a TIE, which extra innings forbid.
        It leaked 0.0004 of mass onto an impossible outcome, which is small
        and is exactly the kind of small that the whole layer exists to
        remove.

        So a clipped row drops the cells that would finish level and
        renormalises. That is an assumption -- that a six-run deficit behaves
        like a four-run one minus the outcomes it cannot reach -- and it is
        applied only where the data runs out.
        """
        s = int(np.clip(state, -self.max_state, self.max_state))
        if s not in self.table:
            return {(0, 0): 1.0}
        row = self.table[s]
        if s == state:
            return row
        legal = {k: v for k, v in row.items() if state + k[0] - k[1] != 0}
        z = sum(legal.values())
        if z <= 0:
            return {(0, 0): 1.0}
        return {k: v / z for k, v in legal.items()}

    def rows(self) -> list[Mapping[tuple[int, int], float]]:
        return list(self.table.values())


@dataclass(frozen=True)
class MLBFinalScoreDistribution:
    """Eight-and-a-half innings of scoring, pushed through the ninth.

    Satisfies core.interfaces.ScoreDistribution. `mu_home` and `mu_away` are
    the SHIPPED observed-run rates; the scales convert them to the quantities
    the rule reads, and they are constructor arguments rather than constants
    so that a refit changes them in one place.
    """

    mu_home: float
    mu_away: float
    r_home: float
    r_away: float
    layer: NinthInningLayer
    home_eight_scale: float = 0.9284
    away_nine_scale: float = 0.9688
    _joint: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mu_home <= 0 or self.mu_away <= 0:
            raise ValueError("expected runs must be positive")
        if not 0.5 < self.home_eight_scale < 1.0:
            raise ValueError(
                "the home eight-inning scale must shrink the observed rate "
                "and not by more than half; a value of 1.0 means the "
                "truncation is about to be counted twice"
            )
        if not 0.5 < self.away_nine_scale <= 1.0:
            raise ValueError("the away nine-inning scale must be in (0.5, 1]")
        object.__setattr__(self, "_joint", None)

    # -- composition ------------------------------------------------------

    def _nb(self, mu: float, r: float, n: int) -> np.ndarray:
        return stats.nbinom.pmf(np.arange(n + 1), r, r / (r + mu))

    def joint(self) -> np.ndarray:
        """P(final home runs, final away runs)."""
        if self._joint is not None:
            return self._joint

        n = MAX_RUNS
        pad = max(max(max(i, j) for i, j in row) for row in self.layer.rows()) + 1
        ph8 = self._nb(self.mu_home * self.home_eight_scale, self.r_home, n)
        pa9 = self._nb(self.mu_away * self.away_nine_scale, self.r_away, n)
        out = np.zeros((n + 1 + pad, n + 1 + pad))

        for h8 in range(n + 1):
            if ph8[h8] < 1e-12:
                continue
            for a9 in range(n + 1):
                p0 = ph8[h8] * pa9[a9]
                if p0 < 1e-12:
                    continue
                for (hg, ag), q in self.layer.row(h8 - a9).items():
                    if q > 0.0:
                        out[h8 + hg, a9 + ag] += p0 * q

        out /= out.sum()
        object.__setattr__(self, "_joint", out)
        return out

    def _margin(self) -> tuple[np.ndarray, np.ndarray]:
        j = self.joint()
        n = j.shape[0]
        idx = np.arange(n)
        diff = idx[:, None] - idx[None, :]
        offs = np.arange(-(n - 1), n)
        return offs, np.array([j[diff == d].sum() for d in offs])

    def _total(self) -> tuple[np.ndarray, np.ndarray]:
        j = self.joint()
        n = j.shape[0]
        idx = np.arange(n)
        tot = idx[:, None] + idx[None, :]
        offs = np.arange(0, 2 * n - 1)
        return offs, np.array([j[tot == s].sum() for s in offs])

    # -- protocol ---------------------------------------------------------

    @property
    def is_discrete(self) -> bool:
        return True

    @property
    def pmf_is_exact(self) -> bool:
        """Composed from exact negative-binomial atoms."""
        return True

    def margin_mean(self) -> float:
        o, v = self._margin()
        return float((o * v).sum())

    def margin_sd(self) -> float:
        o, v = self._margin()
        mu = float((o * v).sum())
        return float(np.sqrt(((o - mu) ** 2 * v).sum()))

    def margin_cdf(self, x: float) -> float:
        o, v = self._margin()
        return float(v[o <= np.floor(x)].sum())

    def margin_pmf(self, x: float) -> float:
        if x != np.floor(x):
            return 0.0
        o, v = self._margin()
        hit = v[o == int(x)]
        return float(hit[0]) if len(hit) else 0.0

    def total_mean(self) -> float:
        o, v = self._total()
        return float((o * v).sum())

    def total_sd(self) -> float:
        o, v = self._total()
        mu = float((o * v).sum())
        return float(np.sqrt(((o - mu) ** 2 * v).sum()))

    def total_cdf(self, x: float) -> float:
        o, v = self._total()
        return float(v[o <= np.floor(x)].sum())

    def total_pmf(self, x: float) -> float:
        if x != np.floor(x) or x < 0:
            return 0.0
        o, v = self._total()
        hit = v[o == int(x)]
        return float(hit[0]) if len(hit) else 0.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        j = self.joint()
        flat = j.ravel()
        draws = rng.choice(flat.size, size=n, p=flat / flat.sum())
        return np.column_stack(np.unravel_index(draws, j.shape))

    def ties(self) -> float:
        """P(final margin == 0). Must be zero: extra innings resolve every game."""
        return self.margin_pmf(0.0)
