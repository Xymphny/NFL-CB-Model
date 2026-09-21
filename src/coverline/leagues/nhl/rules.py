"""The two league rules that turn a scoring process into an NHL final score.

WHY THIS FILE EXISTS
An NHL final score is not a draw from a scoring model. It is three things
stacked, and only the first one is a scoring process:

  1. sixty minutes played with both goalies on the ice,
  2. a goalie pull, which fires conditional on the score and changes it,
  3. a tie-break that awards exactly one goal.

ADR 0007 has the measurement. The short version is that the second and third
layers are what independent Poisson gets wrong, and neither is a correlation:
fitting a negative dependence parameter reproduces a moment and misses the
shape. Specifically, the margin distribution is NON-MONOTONE -- more
three-goal games than two-goal games -- across exactly the 1.5 line the puck
line is priced on, and no independent Poisson produces that at any rates.

WHAT THE LAYERS CONDITION ON, AND WHY IT IS NOT QUITE RIGHT
The pull layer conditions on the NO-PULL margin: the score with every
pull-period goal removed. A coach does not see that number; they see the
score as played, which differs whenever an earlier empty-net goal has already
landed. The no-pull margin is used anyway because it is the quantity a
scoring model produces, so it is the only one available at prediction time.
The approximation is stated rather than hidden, and it is the first thing to
revisit if the layer underperforms.

WHAT IS DELIBERATELY NOT MODELLED
Timing. The layer is a per-game table, not a hazard over the final minutes,
so it cannot express "pulled with 90 seconds left" differently from "pulled
with three minutes left". The measured mean empty-net goal arrives 18.6
minutes into the third and has moved about twelve seconds earlier across
2016-2023, so timing is drifting and a per-game table will drift with it.
That is a reason to re-measure per season, not a reason to fit a hazard now
on data whose holdout is already thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

#: Goals per side considered when composing. NHL teams have scored more than
#: this, but never with mass worth carrying: P(score > 15) under any plausible
#: rate is below 1e-9, and the grid cost is quadratic.
MAX_GOALS = 16


@dataclass(frozen=True)
class OvertimeLayer:
    """A regulation tie becomes a one-goal win. Deterministically.

    Measured over 9,576 games in 2016-2023: no game ended tied, and all 2,166
    decided after regulation ended at a margin of EXACTLY one. There is no
    distribution to fit here -- only which side gets the goal.

    HOME ADVANTAGE VANISHES IN THE TIE-BREAK, which is the finding worth
    carrying. Home teams win 54.94% of games decided in regulation and 50.65%
    of those decided after it (50.28% in overtime, 51.34% in the shootout).
    Three-on-three and a shootout neutralise most of what home ice is worth.
    A model that carries its regulation home edge through the tie-break
    overprices every home moneyline by roughly the tie probability times the
    difference -- about a point of win probability, which is more than the
    edge most bets are placed on.
    """

    home_win_prob: float = 0.5065

    def __post_init__(self) -> None:
        if not 0.0 < self.home_win_prob < 1.0:
            raise ValueError("overtime home win probability must be in (0, 1)")


@dataclass(frozen=True)
class GoaliePullLayer:
    """What the pull adds, conditional on how far apart the teams are.

    `table` maps an absolute no-pull margin to a distribution over
    (goals gained by the LEADER, goals gained by the TRAILER). Framing it by
    leader and trailer rather than by home and away is the physical symmetry:
    a coach pulls because they are behind, not because they are away. The
    measured asymmetry is small -- at a one-goal deficit a trailing home team
    draws a pull goal 51.2% of the time against 54.0% for a trailing away
    team -- and pooling is what makes the table estimable per season.

    Leads at or beyond `max_lead` are treated as that lead. Pull goals at four
    or more are 7% of games and falling, so the tail is flat rather than
    extrapolated.

    A TIED no-pull score has no leader. Its row is applied symmetrically,
    splitting each (i, j) evenly between the two sides. This is not a
    convenience: 17.7% of games have a tied no-pull score and 5.2% of those
    still contain a pull goal, because the score AS PLAYED was not tied when
    the coach decided.
    """

    table: Mapping[int, Mapping[tuple[int, int], float]]
    max_lead: int = 4

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("pull layer has no measured table")
        for lead, row in self.table.items():
            if lead < 0:
                raise ValueError(f"lead must be non-negative, got {lead}")
            total = sum(row.values())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"pull table row for lead {lead} sums to {total}, not 1"
                )

    def rows(self):
        """Every measured row. Used to derive composition headroom."""
        return list(self.table.values())

    def row(self, lead: int) -> Mapping[tuple[int, int], float]:
        lead = min(abs(int(lead)), self.max_lead)
        while lead >= 0 and lead not in self.table:
            lead -= 1
        if lead < 0:
            raise KeyError("pull table has no usable row")
        return self.table[lead]


@dataclass(frozen=True)
class NHLFinalScoreDistribution:
    """A no-pull scoring model pushed through both rules.

    Satisfies core.interfaces.ScoreDistribution. The rates are for SIXTY
    MINUTES WITH BOTH GOALIES ON THE ICE, which is not the same object as the
    rates in data/nhl_fitted.json -- those were fitted to final scores and
    therefore already contain the empty-net goals this layer adds. Passing
    final-score rates in here double-counts them.

    Caches are LAZY, matching NormalMarginDistribution. Constructing one of
    these to read its mean should not cost a 16-by-16 convolution.
    """

    lam_home: float
    lam_away: float
    pull: GoaliePullLayer
    overtime: OvertimeLayer = field(default_factory=OvertimeLayer)
    _joint: np.ndarray | None = field(default=None, init=False, repr=False,
                                      compare=False)

    def __post_init__(self) -> None:
        if self.lam_home <= 0 or self.lam_away <= 0:
            raise ValueError("no-pull scoring rates must be positive")
        object.__setattr__(self, "_joint", None)

    # -- composition ------------------------------------------------------

    def joint(self) -> np.ndarray:
        """P(final home score, final away score), shape (MAX_GOALS+2)^2."""
        if self._joint is not None:
            return self._joint

        from scipy import stats

        n = MAX_GOALS
        ph = stats.poisson.pmf(np.arange(n + 1), self.lam_home)
        pa = stats.poisson.pmf(np.arange(n + 1), self.lam_away)
        # Headroom is DERIVED, not guessed: the largest gain any row of the
        # measured table can add, plus one for the overtime goal. A fixed pad
        # of two was enough for the tune table and raised IndexError on the
        # holdout, which is the kind of failure that would otherwise be
        # discovered by a live slate.
        pad = max(max(max(i, j) for i, j in row) for row in self.pull.rows()) + 1
        out = np.zeros((n + 1 + pad, n + 1 + pad))

        for hr in range(n + 1):
            if ph[hr] < 1e-12:
                continue
            for ar in range(n + 1):
                p0 = ph[hr] * pa[ar]
                if p0 < 1e-12:
                    continue
                m = hr - ar
                for (gain_lead, gain_trail), q in self.pull.row(m).items():
                    if q <= 0.0:
                        continue
                    if m > 0:
                        pairs = (((hr + gain_lead, ar + gain_trail), 1.0),)
                    elif m < 0:
                        pairs = (((hr + gain_trail, ar + gain_lead), 1.0),)
                    else:
                        # No leader. Split the row evenly across the two sides.
                        pairs = (
                            ((hr + gain_lead, ar + gain_trail), 0.5),
                            ((hr + gain_trail, ar + gain_lead), 0.5),
                        )
                    for (h1, a1), share in pairs:
                        w = p0 * q * share
                        if h1 == a1:
                            out[h1 + 1, a1] += w * self.overtime.home_win_prob
                            out[h1, a1 + 1] += w * (
                                1.0 - self.overtime.home_win_prob
                            )
                        else:
                            out[h1, a1] += w

        out /= out.sum()
        object.__setattr__(self, "_joint", out)
        return out

    def _margin_pmf_vector(self) -> tuple[np.ndarray, np.ndarray]:
        j = self.joint()
        n = j.shape[0]
        idx = np.arange(n)
        m = idx[:, None] - idx[None, :]
        offs = np.arange(-(n - 1), n)
        vals = np.array([j[m == d].sum() for d in offs])
        return offs, vals

    def _total_pmf_vector(self) -> tuple[np.ndarray, np.ndarray]:
        j = self.joint()
        n = j.shape[0]
        idx = np.arange(n)
        t = idx[:, None] + idx[None, :]
        offs = np.arange(0, 2 * n - 1)
        vals = np.array([j[t == s].sum() for s in offs])
        return offs, vals

    # -- protocol ---------------------------------------------------------

    @property
    def is_discrete(self) -> bool:
        return True

    def margin_mean(self) -> float:
        o, v = self._margin_pmf_vector()
        return float((o * v).sum())

    def margin_sd(self) -> float:
        o, v = self._margin_pmf_vector()
        mu = float((o * v).sum())
        return float(np.sqrt(((o - mu) ** 2 * v).sum()))

    def margin_cdf(self, x: float) -> float:
        o, v = self._margin_pmf_vector()
        return float(v[o <= np.floor(x)].sum())

    def margin_pmf(self, x: float) -> float:
        if x != np.floor(x):
            return 0.0
        o, v = self._margin_pmf_vector()
        hit = v[o == int(x)]
        return float(hit[0]) if len(hit) else 0.0

    def total_mean(self) -> float:
        o, v = self._total_pmf_vector()
        return float((o * v).sum())

    def total_sd(self) -> float:
        o, v = self._total_pmf_vector()
        mu = float((o * v).sum())
        return float(np.sqrt(((o - mu) ** 2 * v).sum()))

    def total_cdf(self, x: float) -> float:
        o, v = self._total_pmf_vector()
        return float(v[o <= np.floor(x)].sum())

    def total_pmf(self, x: float) -> float:
        if x != np.floor(x) or x < 0:
            return 0.0
        o, v = self._total_pmf_vector()
        hit = v[o == int(x)]
        return float(hit[0]) if len(hit) else 0.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        j = self.joint()
        flat = j.ravel()
        draws = rng.choice(flat.size, size=n, p=flat / flat.sum())
        return np.column_stack(np.unravel_index(draws, j.shape))

    def ties(self) -> float:
        """P(final margin == 0). Must be zero: the league does not permit it.

        Exposed because it is the cheapest check that the overtime layer is
        wired in at all, and a reader pricing a three-way market needs to
        know the draw is not merely unlikely here but impossible.
        """
        return self.margin_pmf(0.0)
