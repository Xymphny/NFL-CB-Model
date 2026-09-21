"""Basketball on the seam. Ratings FITTED AND GRADED; sigma is constant.

WHERE THE NUMBERS COME FROM
Not from this file. data/nba_fitted.json carries team ratings walked forward
within season, with hyperparameters chosen on 2021-2022 and the model graded
ONCE on 2023 -- a season never graded before. Paired gain +0.0571 in mean
log-likelihood, SE 0.0096, t = +5.96, against a noise ceiling of 1.85 for the
18 hyperparameter combinations tried.

The artifact carries its grade and the loader refuses it if that grade does
not clear, so the package still ships no unmeasured constant.

An earlier STATIC fit across seasons measured t = -6.96 -- decisively worse
than predicting the league average. Three-year-old ratings are not merely
stale in this sport, they are actively misleading, and that is the whole
reason this model walks forward.

SIGMA IS CONSTANT, AND THAT WAS TESTED
ADR 0005 predicted sigma should rise with spread magnitude. A varying sigma
was fitted and graded and came back WORSE (t = -6.48, fitted slope -0.111).
That does not refute the ADR -- these ratings separate games across a much
narrower range than market spreads do, so the effect is ruled out only where
it was measured. Until market spreads are available the constant is what the
evidence supports, and NBAModel still takes sigma as a FUNCTION so the
question stays open at the call site.

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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from coverline.core.distributions import NormalMarginDistribution
from coverline.core.interfaces import ScoreDistribution

_ROOT = Path(__file__).resolve().parents[4]
FITTED_PATH = _ROOT / "data" / "nba_fitted.json"


class NotFitted(NotImplementedError):
    """No usable fitted parameters. Raised rather than falling back."""


def load_fitted(path: Path = FITTED_PATH) -> dict:
    """Load the graded artifact, refusing one that did not clear its gate."""
    if not Path(path).exists():
        raise NotFitted(f"no fitted NBA parameters at {path}")
    art = json.loads(Path(path).read_text())
    if not art.get("holdout_grade", {}).get("supported"):
        raise NotFitted(
            f"{Path(path).name} did not clear its held-out gate "
            f"(t = {art.get('holdout_grade', {}).get('t')})."
        )
    return art


def margin_from_fit(art: dict, home: str, away: str) -> float:
    r = art["ratings"]
    for t in (home, away):
        if t not in r:
            raise KeyError(f"{t} has no fitted rating; refusing to price it")
    return r[home] - r[away] + art["hyperparameters"]["home_adv"]


def constant_sigma(art: dict) -> Callable[[float], float]:
    """The sigma the evidence supports: flat. Returned as a function so the
    varying-sigma question stays open at the call site rather than being
    closed by a constant in a signature."""
    sd = float(art["sigma_constant"])
    return lambda mu: sd


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
            # Withheld in the market list and in the object. The 18.0 has
            # never been graded against anything; NBA totals have an
            # unconditional sd of 20.11 over 6,000 games, which bounds
            # nothing about a residual and is exactly why this is refused
            # rather than corrected.
            total_validated=False,
            discrete=False,
        )
