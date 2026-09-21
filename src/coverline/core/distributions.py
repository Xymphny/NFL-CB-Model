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


class UnvalidatedTotal(NotImplementedError):
    """Raised when a distribution is asked for a total it does not model.

    THE HAZARD THIS CLOSES
    NFL, CFB and NBA all withhold totals from primary_markets, and all three
    were still handing out a total when asked. A caller iterating markets was
    safe; a caller asking the DISTRIBUTION -- which is the whole point of the
    seam, since everything downstream imports core and talks to a
    ScoreDistribution -- got a confident number built on a guess.

    Two of the three guesses are measurably wrong by about a third. CFB ships
    sd_total = 14.0 against a measured 18.79 over 3,863 games, where its
    mu_total is a constant so residual and unconditional dispersion are the
    same number; its nominal 95% interval covers 87.0%. NFL ships 10.0 against
    the 13.353 its own totals_validation.json records as model RMSE. NBA's
    18.0 is simply unmeasured here, which is not better.

    Withholding a market in a list and answering it in the object is the kind
    of half-connection this seam exists to prevent, so the object refuses.
    """


@dataclass(frozen=True)
class NormalMarginDistribution:
    """Near-normal margin and total. Used by NFL, CFB and NBA.

    NBA passes ``discrete=False``: its margin support is wide enough that atom
    mass at any single value is small, and the modelling convention treats it
    as continuous. NFL and CFB pass ``discrete=True``, which turns on
    continuity-corrected integer masses so pushes price correctly.

    ``margin_sd`` is a constructor argument rather than a class constant on
    purpose, and the reason first written here was WRONG.

    It said NBA margin variance correlates about 0.60 with spread magnitude,
    so a constant sigma is defensible in football and wrong in basketball.
    That 0.60 came from ADR 0005 by way of a bucketed correlation, and a
    correlation over a handful of bucket means is not evidence of anything at
    that size. Simulating CONSTANT-variance noise through the same bucketing
    on 3,540 walk-forward NBA games gives a median absolute correlation of
    0.397 at five buckets and a 90th percentile of 0.801. The same data gives
    an observed -0.730 -- larger, opposite in sign, and equally meaningless.

    The per-game correlation, which uses every game rather than a few means,
    is -0.016 between predicted spread magnitude and absolute residual, over
    predicted spreads spanning 0.003 to 23.3 points. Residual sd by decile is
    flat from 13.1 to 14.4 with no trend. A constant sigma is defensible in
    basketball too, as far as anything here can measure.

    The argument stays a constructor argument anyway, for a better reason
    than the original: a league should own its dispersion rather than inherit
    a class constant, and NBAModel takes sigma as a FUNCTION so the question
    stays open rather than being closed by a default. What is settled is that
    the 0.60 was never a reason. See ADR 0012 and
    model/distribution_grades.json.
    """

    mu_margin: float
    sd_margin: float
    mu_total: float
    sd_total: float
    discrete: bool = True
    rho: float = 0.0
    #: False when the total is a placeholder or an ungraded guess. Every
    #: method that depends on it then raises rather than answering. Default
    #: True so that a caller who HAS validated a total gets the old behaviour
    #: without saying anything; the three leagues that have not say so.
    total_validated: bool = True
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
        self._require_total("total_mean()")
        return self.mu_total

    def total_sd(self) -> float:
        self._require_total("total_sd()")
        return self.sd_total

    def total_cdf(self, x: float) -> float:
        self._require_total("total_cdf()")
        return self._cdf(x, self.mu_total, self.sd_total)

    def total_pmf(self, x: float) -> float:
        self._require_total("total_pmf()")
        return self._pmf(x, self.mu_total, self.sd_total, None)

    def _require_total(self, what: str) -> None:
        if not self.total_validated:
            raise UnvalidatedTotal(
                f"{what} was asked of a distribution whose total is not "
                "validated. mu_total and sd_total here are a placeholder, the "
                "market is withheld from primary_markets, and answering would "
                "hand back a confident number built on a guess. Use "
                "sample_margin() if you need draws, or construct with "
                "total_validated=True once a total model has been graded."
            )

    def sample_margin(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Margins only, valid whether or not the total is modelled.

        Exists so that refusing the total does not also refuse correlated
        slate staking on spreads, which is the one thing sample() is needed
        for and which never touches the total dimension.
        """
        draws = rng.normal(self.mu_margin, self.sd_margin, size=n)
        return np.rint(draws) if self.discrete else draws

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        self._require_total("sample()")
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


@dataclass(frozen=True)
class NegativeBinomialScoreDistribution:
    """Independent overdispersed scores. Used by MLB.

    WHY NOT POISSON, WHICH IS THE OBVIOUS CHOICE
    Baseball looks like the textbook Poisson sport and is not. Measured over
    12,148 walk-forward games, runs are overdispersed by a factor of roughly
    2.2: home variance/mean is 2.15, away 2.37, where Poisson requires 1.
    Pricing MLB with a Poisson would understate run variance by more than
    half, which matters most exactly where it is used -- totals, and the tails
    of any derived probability.

    WHY NOT THE SHARED COMPONENT IN BivariatePoissonDistribution
    That was the natural fix and it is wrong here. The shared term inflates
    total variance by creating POSITIVE CORRELATION between the two scores,
    and the measured correlation is +0.0006 -- independence, to three decimal
    places. Using it would buy the right total variance with a dependency that
    does not exist, and would leave the margin variance still too small.

    Independence is confirmed rather than assumed: measured margin variance
    (20.11) and total variance (20.13) both equal the sum of the individual
    variances (20.12), which is what independence predicts and what a shared
    component would break.

    So: independent negative binomials, one dispersion parameter per side.
    Fitted r is about 3.9 (home) and 3.2 (away).

    MECHANICS
    The margin and total of two independent NBs have no convenient closed
    form, so both are convolved numerically over a bounded support. Baseball
    scores are small and the support is generous, which makes this exact to
    floating point rather than approximate.
    """

    mu_home: float
    mu_away: float
    r_home: float
    r_away: float
    _max_runs: int = 40

    def __post_init__(self) -> None:
        for name, v in (("mu_home", self.mu_home), ("mu_away", self.mu_away)):
            if v <= 0:
                raise ValueError(f"{name} must be positive")
        for name, v in (("r_home", self.r_home), ("r_away", self.r_away)):
            if v <= 0:
                raise ValueError(
                    f"{name} must be positive; a non-positive dispersion "
                    "parameter is not an overdispersed distribution"
                )
        object.__setattr__(self, "_pmf_cache", None)

    # -- protocol ---------------------------------------------------------

    @property
    def is_discrete(self) -> bool:
        return True

    def margin_mean(self) -> float:
        return self.mu_home - self.mu_away

    def margin_sd(self) -> float:
        return float(np.sqrt(self._var(self.mu_home, self.r_home)
                             + self._var(self.mu_away, self.r_away)))

    def total_mean(self) -> float:
        return self.mu_home + self.mu_away

    def total_sd(self) -> float:
        return self.margin_sd()   # independent: both are the sum of variances

    def margin_pmf(self, x: float) -> float:
        if not _is_integer(x):
            return 0.0
        m, _ = self._grids()
        k = int(x) + self._max_runs
        return float(m[k]) if 0 <= k < len(m) else 0.0

    def margin_cdf(self, x: float) -> float:
        m, _ = self._grids()
        k = int(np.floor(x)) + self._max_runs
        if k < 0:
            return 0.0
        if k >= len(m):
            return 1.0
        return float(np.cumsum(m)[k])

    def total_pmf(self, x: float) -> float:
        if not _is_integer(x) or x < 0:
            return 0.0
        _, t = self._grids()
        k = int(x)
        return float(t[k]) if k < len(t) else 0.0

    def total_cdf(self, x: float) -> float:
        _, t = self._grids()
        k = int(np.floor(x))
        if k < 0:
            return 0.0
        if k >= len(t):
            return 1.0
        return float(np.cumsum(t)[k])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        h = self._draw(rng, n, self.mu_home, self.r_home)
        a = self._draw(rng, n, self.mu_away, self.r_away)
        return np.column_stack([h, a])

    # -- internals --------------------------------------------------------

    @staticmethod
    def _var(mu: float, r: float) -> float:
        return mu + mu * mu / r

    @staticmethod
    def _p(mu: float, r: float) -> float:
        """scipy's nbinom uses (n=r, p), with mean = r(1-p)/p."""
        return r / (r + mu)

    def _side_pmf(self, mu: float, r: float) -> np.ndarray:
        k = np.arange(0, self._max_runs + 1)
        return stats.nbinom.pmf(k, r, self._p(mu, r))

    def _grids(self) -> tuple[np.ndarray, np.ndarray]:
        """(margin pmf indexed from -max_runs, total pmf indexed from 0)."""
        if self._pmf_cache is not None:
            return self._pmf_cache
        h = self._side_pmf(self.mu_home, self.r_home)
        a = self._side_pmf(self.mu_away, self.r_away)
        total = np.convolve(h, a)
        # margin: P(H - A = d) = sum_j h[j + d] * a[j]
        size = 2 * self._max_runs + 1
        margin = np.zeros(size)
        for d in range(-self._max_runs, self._max_runs + 1):
            lo = max(0, -d)
            hi = min(self._max_runs, self._max_runs - d)
            if lo > hi:
                continue
            j = np.arange(lo, hi + 1)
            margin[d + self._max_runs] = float(np.sum(h[j + d] * a[j]))
        out = (margin, total)
        object.__setattr__(self, "_pmf_cache", out)
        return out

    def _draw(self, rng: np.random.Generator, n: int, mu: float, r: float) -> np.ndarray:
        return rng.negative_binomial(r, self._p(mu, r), size=n)
