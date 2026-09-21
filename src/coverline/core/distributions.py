"""Two implementations, not five.

Three leagues share a near-normal margin (NFL, CFB, NBA) and two share a
low-count structure (NHL, MLB). Everything downstream sees only the
ScoreDistribution protocol, so adding a sixth league means picking one of these
or writing a third -- it does not mean touching pricing, staking or grading.

KEY NUMBERS: MEASURED, AND THE FIRST API DESIGN WAS WRONG
---------------------------------------------------------
NFL and CFB margins carry large excess mass on 3 and 7 that a rounded normal
does not reproduce. This class originally accepted an absolute mass table
(``key_number_mass``) that overrode pmf(k) outright. That was a design error,
caught by measuring the effect properly: an absolute table cannot be right for
a CONDITIONAL distribution, because P(margin = 3) must depend on how large the
spread is. A game with a 14-point spread and a pick-em cannot share a fixed
P(margin = 3).

What is actually true is multiplicative. The excess is a property of how
football scores -- field goals and touchdowns land on particular numbers --
not of any one matchup. So the correction is a weight w(k) applied to the
rounded-normal mass and then RENORMALISED, which adjusts the shape of the
distribution without inventing or destroying total probability.

Measured on nflverse 2010-2021 and graded once on 2022-2025
(data/nfl_key_numbers.json): mean held-out log-likelihood improves by +0.114
per game, SE 0.016, t = +7.00. The plain rounded normal puts P(margin = 3) at
2.74% against an empirical 7.36% -- so every push price on a 3 computed
without these weights was wrong by nearly a factor of three.

Weights are still not a default. A distribution built without them reports
``has_key_number_correction`` as False, and callers pricing pushes on key
numbers should check it rather than assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import stats


def _is_integer(x: float) -> bool:
    return float(x).is_integer()


@dataclass(frozen=True)
class NormalMarginDistribution:
    """Near-normal margin and total. Used by NFL, CFB and NBA.

    NBA passes ``discrete=False``: its margin support is wide enough that atom
    mass at any single value is small, and the modelling convention treats it
    as continuous. NFL and CFB pass ``discrete=True``, which turns on
    continuity-corrected integer masses so pushes price correctly.

    ``margin_sd`` is a constructor argument rather than a class constant on
    purpose. In the NBA, margin variance correlates about 0.60 with spread
    magnitude, against roughly -0.06 in the NFL -- a constant sigma is
    defensible in football and wrong in basketball, so the caller owns it.
    """

    mu_margin: float
    sd_margin: float
    mu_total: float
    sd_total: float
    discrete: bool = True
    rho: float = 0.0
    key_number_weights: Mapping[int, float] | None = field(default=None)
    _support: int = 60
    _norm_cache: Any = field(default=None, repr=False, compare=False)
    _cdf_cache: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.sd_margin <= 0 or self.sd_total <= 0:
            raise ValueError("standard deviations must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be strictly between -1 and 1")
        if self.key_number_weights is not None:
            if not self.discrete:
                raise ValueError(
                    "key_number_weights are meaningless for a continuous margin"
                )
            if any(w < 0 for w in self.key_number_weights.values()):
                raise ValueError("key-number weights must be non-negative")
            # Caches are built LAZILY, on first use of pmf or cdf.
            #
            # They were originally built here, eagerly. That fixed the
            # O(support^2) cdf -- margin_cdf had been recomputing the
            # normaliser for every atom it summed -- but replaced it with a
            # subtler cost: constructing a distribution to read its MEAN paid
            # for a 121-point table it never touched. A parity run over 1,945
            # games spent 39 seconds doing exactly that. Building on demand
            # keeps the cdf fast and makes construction free again.
            object.__setattr__(self, "_norm_cache", None)
            object.__setattr__(self, "_cdf_cache", None)

    # -- protocol ---------------------------------------------------------

    @property
    def is_discrete(self) -> bool:
        return self.discrete

    @property
    def has_key_number_correction(self) -> bool:
        """Whether measured key-number weights are in force.

        Exposed so pricing code can refuse to quote a push price on a key
        number rather than quoting one that is wrong by ~3x. See the module
        docstring for the measurement.
        """
        return self.key_number_weights is not None

    def margin_mean(self) -> float:
        return self.mu_margin

    def margin_sd(self) -> float:
        return self.sd_margin

    def margin_cdf(self, x: float) -> float:
        if self.discrete and self.key_number_weights is not None:
            lo = int(np.floor(x))
            if lo < -self._support:
                return 0.0
            if lo >= self._support:
                return 1.0
            return float(self._weighted_cdf()[lo + self._support])
        return self._cdf(x, self.mu_margin, self.sd_margin)

    def margin_pmf(self, x: float) -> float:
        if not self.discrete or not _is_integer(x):
            return 0.0
        base = self._raw_pmf(int(x), self.mu_margin, self.sd_margin)
        if self.key_number_weights is None:
            return base
        return base * self._weight(int(x)) / self._normaliser()

    def total_mean(self) -> float:
        return self.mu_total

    def total_sd(self) -> float:
        return self.sd_total

    def total_cdf(self, x: float) -> float:
        return self._cdf(x, self.mu_total, self.sd_total)

    def total_pmf(self, x: float) -> float:
        return self._pmf(x, self.mu_total, self.sd_total, None)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cov = [
            [self.sd_margin**2, self.rho * self.sd_margin * self.sd_total],
            [self.rho * self.sd_margin * self.sd_total, self.sd_total**2],
        ]
        draws = rng.multivariate_normal([self.mu_margin, self.mu_total], cov, size=n)
        margin, total = draws[:, 0], draws[:, 1]
        if self.discrete:
            margin = np.rint(margin)
            total = np.rint(total)
            # A margin and total of differing parity cannot be realised by any
            # integer scoreline. Nudge the total, which is the less
            # price-sensitive of the two.
            total = np.where((total - margin) % 2 != 0, total + 1, total)
        home = (total + margin) / 2.0
        away = (total - margin) / 2.0
        return np.column_stack([home, away])

    # -- internals --------------------------------------------------------

    def _cdf(self, x: float, mu: float, sd: float) -> float:
        if self.discrete:
            return float(stats.norm.cdf((np.floor(x) + 0.5 - mu) / sd))
        return float(stats.norm.cdf((x - mu) / sd))

    def _pmf(
        self, x: float, mu: float, sd: float, table: Mapping[int, float] | None
    ) -> float:
        if not self.discrete or not _is_integer(x):
            return 0.0
        return self._raw_pmf(int(x), mu, sd)

    @staticmethod
    def _raw_pmf(k: int, mu: float, sd: float) -> float:
        upper = stats.norm.cdf((k + 0.5 - mu) / sd)
        lower = stats.norm.cdf((k - 0.5 - mu) / sd)
        return float(upper - lower)

    def _weight(self, k: int) -> float:
        return float(self.key_number_weights.get(k, 1.0))  # type: ignore[union-attr]

    def _normaliser(self) -> float:
        if self._norm_cache is None:
            object.__setattr__(self, "_norm_cache", self._compute_normaliser())
        return self._norm_cache

    def _weighted_cdf(self) -> np.ndarray:
        if self._cdf_cache is None:
            object.__setattr__(self, "_cdf_cache", self._compute_weighted_cdf())
        return self._cdf_cache

    def _compute_normaliser(self) -> float:
        """Sum of weighted raw masses over the support.

        Renormalising is what makes the weights a SHAPE adjustment. Without it
        the reweighted masses would not sum to one and every probability the
        distribution reports would be inflated by the average weight.
        """
        total = 0.0
        for k in range(-self._support, self._support + 1):
            total += self._raw_pmf(k, self.mu_margin, self.sd_margin) * self._weight(k)
        return total

    def _compute_weighted_cdf(self) -> np.ndarray:
        norm = self._normaliser()
        ks = np.arange(-self._support, self._support + 1)
        masses = np.array([
            self._raw_pmf(int(k), self.mu_margin, self.sd_margin) * self._weight(int(k))
            for k in ks
        ]) / norm
        return np.cumsum(masses)


@dataclass(frozen=True)
class BivariatePoissonDistribution:
    """Low-count scoring with a shared component. Used by NHL and MLB.

    Scores are X = X1 + X3 and Y = X2 + X3 with X1~Pois(l1), X2~Pois(l2),
    X3~Pois(l3). The shared component X3 carries game-level effects that lift
    both teams together -- pace, officiating, conditions -- and it CANCELS in
    the margin, which is why the margin is exactly Skellam(l1, l2) regardless
    of l3 while the total is X1 + X2 + 2*X3. That asymmetry is the whole reason
    to use this rather than two independent Poissons: it lets a league raise
    total variance without touching the margin.

    NHL CAVEAT, recorded because it will otherwise be forgotten: this class
    models scoring as a homogeneous process and knows nothing about empty-net
    goals, which are now about 7% of NHL goals, arrive conditional on a late
    one-goal deficit, and therefore inflate the winner's score in exactly the
    games that decide the puck line. An NHL caller that prices -1.5 off this
    class without an empty-net layer on top is pricing the wrong distribution.
    The layer belongs in leagues/nhl/, not here.
    """

    lam_home: float
    lam_away: float
    lam_shared: float = 0.0
    _max_goals: int = 40

    def __post_init__(self) -> None:
        if self.lam_home <= 0 or self.lam_away <= 0:
            raise ValueError("team rate parameters must be positive")
        if self.lam_shared < 0:
            raise ValueError("shared rate must be non-negative")

    # -- protocol ---------------------------------------------------------

    @property
    def is_discrete(self) -> bool:
        return True

    def margin_mean(self) -> float:
        return self.lam_home - self.lam_away

    def margin_sd(self) -> float:
        return float(np.sqrt(self.lam_home + self.lam_away))

    def margin_cdf(self, x: float) -> float:
        return float(stats.skellam.cdf(np.floor(x), self.lam_home, self.lam_away))

    def margin_pmf(self, x: float) -> float:
        if not _is_integer(x):
            return 0.0
        return float(stats.skellam.pmf(int(x), self.lam_home, self.lam_away))

    def total_mean(self) -> float:
        return self.lam_home + self.lam_away + 2.0 * self.lam_shared

    def total_sd(self) -> float:
        return float(np.sqrt(self.lam_home + self.lam_away + 4.0 * self.lam_shared))

    def total_cdf(self, x: float) -> float:
        k = int(np.floor(x))
        if k < 0:
            return 0.0
        return float(sum(self.total_pmf(j) for j in range(0, k + 1)))

    def total_pmf(self, x: float) -> float:
        if not _is_integer(x) or x < 0:
            return 0.0
        k = int(x)
        # total = X1 + X2 + 2*X3, so condition on X3 = m and require k - 2m >= 0.
        lam_sum = self.lam_home + self.lam_away
        if self.lam_shared == 0.0:
            return float(stats.poisson.pmf(k, lam_sum))
        total = 0.0
        for m in range(0, k // 2 + 1):
            total += stats.poisson.pmf(m, self.lam_shared) * stats.poisson.pmf(
                k - 2 * m, lam_sum
            )
        return float(total)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x1 = rng.poisson(self.lam_home, size=n)
        x2 = rng.poisson(self.lam_away, size=n)
        x3 = rng.poisson(self.lam_shared, size=n) if self.lam_shared > 0 else 0
        return np.column_stack([x1 + x3, x2 + x3])
