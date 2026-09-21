"""The seam.

Every league produces a ScoreDistribution. Every line of pricing, staking,
grading and CLV code downstream imports only this module and its neighbours in
``core`` -- never a league package. That is what makes five leagues cheaper to
run than three are today: one implementation of everything that touches money,
five callers.

The non-obvious requirement is ``is_discrete`` and the ``*_pmf`` methods. Four
of the five leagues price on integer lines, so the probability mass sitting
exactly ON a line -- the push -- is a first-class quantity, not a rounding
detail. General forecasting libraries (GluonTS, skpro) model cdf/ppf/sample and
leave atom mass implicit, which is fine for continuous forecasting and wrong
for anything that has to price a push. A -3 spread in the NFL and a -1.5 puck
line in the NHL are different questions precisely because of where the mass
sits.

Design note recorded here rather than in a commit message, because this is the
decision the whole package rests on: the protocols below are deliberately
narrow. They expose what PRICING needs, not what modelling wants. A league is
free to carry any internal state it likes; it must only be able to answer these
questions about the distribution it produces.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


class ScoreDistribution(Protocol):
    """A predictive distribution over a game's margin and total.

    Implementations must be immutable and cheap to query: pricing code calls
    these methods once per candidate line per game, across a full slate.
    """

    @property
    def is_discrete(self) -> bool:
        """True when outcomes fall on integers and lines can be pushed.

        NFL, CFB, NHL and MLB are discrete. NBA margins are integers too, but
        the modelling convention there treats them as continuous because the
        support is wide enough that atom mass is small -- see
        NormalMarginDistribution, which takes this as a constructor argument
        rather than assuming it.
        """
        ...

    def margin_mean(self) -> float:
        """Expected home margin (home score minus away score)."""
        ...

    def margin_sd(self) -> float:
        """Standard deviation of home margin."""
        ...

    def margin_cdf(self, x: float) -> float:
        """P(margin <= x). For discrete distributions this INCLUDES the atom at x."""
        ...

    def margin_pmf(self, x: float) -> float:
        """P(margin == x). Zero for continuous distributions and for non-integer x."""
        ...

    def total_mean(self) -> float:
        """Expected combined score."""
        ...

    def total_cdf(self, x: float) -> float:
        """P(total <= x), including the atom at x when discrete."""
        ...

    def total_pmf(self, x: float) -> float:
        """P(total == x). Zero for continuous distributions and for non-integer x."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n (home_score, away_score) pairs, shape (n, 2).

        Required rather than optional because correlated slate staking needs
        joint draws, and because a distribution that cannot simulate itself
        cannot be checked against its own analytic methods -- which is exactly
        what tests/core/test_distributions.py does.
        """
        ...


class StatDistribution(Protocol):
    """A predictive distribution over one player's counting stat.

    Carries BOTH median() and mean() deliberately. Books price player props to
    the median; public projection systems publish the mean; counting stats are
    right-skewed, so the two differ and betting means against median lines
    carries a structural over-bias toward the over. Any prop code that can only
    see one of the two numbers will make that mistake silently, so the
    interface refuses to let a caller forget which one they are holding.
    """

    @property
    def is_discrete(self) -> bool: ...

    def mean(self) -> float:
        """Expected value of the stat."""
        ...

    def median(self) -> float:
        """50th percentile. For discrete stats, the smallest x with cdf(x) >= 0.5."""
        ...

    def cdf(self, x: float) -> float:
        """P(stat <= x), including the atom at x when discrete."""
        ...

    def pmf(self, x: float) -> float:
        """P(stat == x)."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n values, shape (n,)."""
        ...


@runtime_checkable
class LeagueModel(Protocol):
    """What a league package must provide to be registered.

    NOTE on @runtime_checkable: it is applied here only so that diagnostic code
    can ask "does this look like a LeagueModel at all". It does NOT check method
    signatures -- the typing docs are explicit about this -- so it must never be
    the thing standing between a broken league and production. The real
    enforcement is the annotated REGISTRY dict in registry.py, which makes mypy
    fail at the @register call site when a method is missing or mistyped.
    """

    @property
    def league(self) -> str:
        """Short lowercase identifier: 'nfl', 'cfb', 'mlb', 'nhl', 'nba'."""
        ...

    @property
    def primary_markets(self) -> Sequence[str]:
        """Markets this league actually prices, e.g. ('spread', 'total', 'moneyline').

        Declared rather than assumed because the leagues genuinely differ:
        the NHL's primary market is the moneyline with a fixed 1.5 puck line,
        not a variable spread, and conformance tests use this to know which
        assertions apply.
        """
        ...

    def predict(self, game_id: str, asof: str) -> ScoreDistribution:
        """Produce a distribution for one game using only data available at `asof`.

        `asof` is an ISO-8601 timestamp and is not decorative: it is the
        point-in-time contract. An implementation that reads any datum whose
        availability timestamp is later than `asof` is leaking, and the
        two-run guard test in tests/core/test_point_in_time.py is designed to
        catch exactly that.
        """
        ...
