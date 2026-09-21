"""Basketball on the seam. STRUCTURE ONLY -- no fitted coefficients.

WHY THERE ARE NO NUMBERS IN THIS FILE
Same reason as NHL: no legacy model, no committed cache, no held-out grading.
A model fitted in an afternoon and shipped would be indistinguishable later
from one that was validated.

THE FAMILY, AND THE ONE THING THAT WILL BREAK A PORTED FOOTBALL MODEL
Near-normal margins, so NormalMarginDistribution -- the same family NFL and
CFB use. That similarity is a trap, and the specific number that springs it:

SIGMA IS NOT CONSTANT. NBA margin variance correlates about 0.60 with spread
magnitude, against roughly -0.06 in the NFL. A constant sigma is defensible
in football and wrong in basketball: blowout-expected games really are more
variable. Any port that copies NFL's fixed MARGIN_SD will be systematically
over-confident on small spreads and under-confident on large ones -- in a way
that looks like a calibration problem rather than a modelling one.

So NBAModel takes a sigma FUNCTION of the predicted margin, not a constant,
and there is no default. Passing a constant is possible and must be a
deliberate act.

MINUTES ARE THE REAL MODEL. The published work that beats the market in this
sport is a player-impact metric weighted by PROJECTED MINUTES, with injury
designations applied -- not a team rating. Rest, load management and late
scratches move a line more than form does, and the information arrives hours
before tip. A team-level port of the football approach is a different and
weaker model, and calling it an NBA model would overstate it.

TREATING THE MARGIN AS CONTINUOUS IS DEFENSIBLE HERE, unlike football:
the support is wide enough that atom mass at any single value is small, so
`discrete=False` is the right default and key numbers do not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from coverline.core.distributions import NormalMarginDistribution
from coverline.core.interfaces import ScoreDistribution


class NotFitted(NotImplementedError):
    """No fitted parameters exist for this league yet."""


@dataclass(frozen=True)
class GameFeatures:
    """Team strength and pace, plus the minutes layer this sport turns on."""

    rating_diff: float
    pace: float
    minutes_projected: bool = False
    """Whether a minutes projection was applied. False means the prediction is
    a team-level approximation of a player-level sport, which is a weaker
    claim and should be recorded as one."""


class FeatureSource(Protocol):
    def features(self, game_id: str, asof: str) -> GameFeatures: ...


#: Sigma as a function of predicted margin. NO DEFAULT. See module docstring:
#: NBA margin variance rises with spread magnitude, and a constant is the
#: specific mistake a ported football model makes.
SigmaModel = Callable[[float], float]


class NBAModel:
    """Implements core.interfaces.LeagueModel, once given fitted pieces."""

    def __init__(self, source: FeatureSource, sigma: SigmaModel,
                 margin_model: Callable[[GameFeatures], float],
                 total_model: Callable[[GameFeatures], float] | None = None) -> None:
        self._source = source
        self._sigma = sigma
        self._margin = margin_model
        self._total = total_model

    @property
    def league(self) -> str:
        return "nba"

    @property
    def primary_markets(self) -> Sequence[str]:
        """Spread and moneyline. Totals absent until a total model exists,
        and player props absent because they are a bottom-up simulation
        problem rather than a game-model one -- books price props to the
        MEDIAN while projection systems publish the MEAN, and counting stats
        are right-skewed, so the two differ systematically."""
        return ("spread", "moneyline")

    @property
    def has_key_number_correction(self) -> bool:
        return True   # continuous margin: no atoms to correct

    def predict(self, game_id: str, asof: str) -> ScoreDistribution:
        f = self._source.features(game_id, asof)
        mu = self._margin(f)
        sd = self._sigma(mu)
        if sd <= 0:
            raise ValueError("sigma model returned a non-positive sd")
        total = self._total(f) if self._total else 225.0
        return NormalMarginDistribution(
            mu_margin=mu, sd_margin=sd, mu_total=total, sd_total=18.0,
            discrete=False,
        )
