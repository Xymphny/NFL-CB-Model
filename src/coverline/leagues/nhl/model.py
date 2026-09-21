"""Hockey on the seam. STRUCTURE ONLY -- no fitted coefficients.

WHY THERE ARE NO NUMBERS IN THIS FILE
NHL is a new league for this project. There is no legacy model to port, no
committed walk-forward cache, and no held-out grading. Shipping coefficients
fitted in an afternoon would be exactly the failure the whole evidence
apparatus exists to prevent: plausible constants indistinguishable, six months
later, from measured ones.

So this package defines the SHAPE -- what a hockey model must provide and
which distribution family fits -- and refuses to run without fitted
parameters supplied by a caller who has measured them. `NHLModel` takes rates
as arguments; there is no default and no module-level constant to import by
accident.

THE FAMILY, AND THE ONE THING KNOWN TO BREAK IT
Low-scoring counts, so the Poisson family, with two qualifications recorded
now because they will otherwise be rediscovered expensively:

EMPTY-NET GOALS. Roughly 7% of NHL goals, about 0.42 a game, and they are not
random: they arrive conditional on a late one-goal deficit, in a league where
about 57% of games are one-goal games. So they inflate the winner's score in
exactly the games that decide the puck line. A homogeneous scoring process
over-prices the underdog on -1.5 and under-prices the favourite. Whoever fits
this needs a state-conditional empty-net layer, not a flat adjustment.

THE FIXED PUCK LINE. Unlike football, the spread does not move -- it is
always 1.5, and the book expresses information through price instead. That
makes the SHAPE of the goal distribution do work the mean alone cannot: two
models agreeing on the moneyline can disagree on the puck line. A
win-probability model is therefore insufficient here in a way it is not for
football.

MLB's lesson applies as a warning, not a template: baseball looked like the
textbook Poisson sport and measured 2.2x overdispersed. Measure hockey before
assuming it is Poisson, including whether the two teams' goals are
independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from coverline.core.distributions import BivariatePoissonDistribution
from coverline.core.interfaces import ScoreDistribution


class NotFitted(NotImplementedError):
    """No fitted parameters exist for this league yet."""


@dataclass(frozen=True)
class GameFeatures:
    """Expected goals per side, and any shared game-level rate.

    `lam_shared` is exposed because hockey plausibly HAS the correlation
    baseball does not -- pace, officiating and score effects lift both teams
    together. Whether it is non-zero is a measurement nobody has made here,
    so it defaults to 0.0 and a fitter must set it deliberately.
    """

    lam_home: float
    lam_away: float
    lam_shared: float = 0.0


class FeatureSource(Protocol):
    def features(self, game_id: str, asof: str) -> GameFeatures: ...


class NHLModel:
    """Implements core.interfaces.LeagueModel, once given a source.

    Deliberately has no fitted constants. If this class can be constructed
    and used without anyone having measured anything, the guard rails are
    doing nothing.
    """

    def __init__(self, source: FeatureSource,
                 empty_net_layer: object | None = None) -> None:
        self._source = source
        self._empty_net = empty_net_layer

    @property
    def league(self) -> str:
        return "nhl"

    @property
    def primary_markets(self) -> Sequence[str]:
        """Moneyline and total only.

        The PUCK LINE is deliberately absent. It is hockey's most distinctive
        market and the one most exposed to the empty-net problem above;
        offering it without a state-conditional layer would mean pricing the
        market whose bias is best understood and least corrected.
        """
        return ("moneyline", "total")

    @property
    def has_key_number_correction(self) -> bool:
        return True   # counts, not margins on a line

    @property
    def models_empty_net(self) -> bool:
        """False until someone builds the layer. Exposed so a caller pricing
        anything sensitive to the goal distribution's tail can check rather
        than assume."""
        return self._empty_net is not None

    def predict(self, game_id: str, asof: str) -> ScoreDistribution:
        f = self._source.features(game_id, asof)
        if f.lam_home <= 0 or f.lam_away <= 0:
            raise ValueError("expected goals must be positive")
        return BivariatePoissonDistribution(
            lam_home=f.lam_home, lam_away=f.lam_away, lam_shared=f.lam_shared,
        )
