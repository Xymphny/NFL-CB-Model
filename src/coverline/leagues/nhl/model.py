"""Hockey on the seam. Rates FITTED AND GRADED; the joint distribution is not.

WHERE THE NUMBERS COME FROM
Not from this file. data/nhl_fitted.json carries attack and defence ratings
walked forward within the 2024 season and graded ONCE on it -- paired gain
+0.0267 in mean log-likelihood, SE 0.0089, t = +3.02, with ZERO
hyperparameter trials, because only one ungraded season existed and searching
it before grading on it is how a result gets manufactured.

The artifact carries its own grade and the loader REFUSES it if that grade
does not clear. So the package still ships no unmeasured constant: what it
ships is a number that earned its place and travels with the evidence.

An earlier static fit across seasons measured t = -1.33 and did not ship.

THE RATES ARE GRADED. THE JOINT DISTRIBUTION IS NOT.
Independent Poisson under-predicts one-goal games by about ten points -- 28.4%
against an actual 38.1% -- because the two scores are NEGATIVELY correlated
(-0.14), and BivariatePoissonDistribution's shared component can only express
POSITIVE correlation. Real games stay closer than independent rates allow: a
leading team defends, a trailing team presses.

That is a property of the joint distribution, not of the rates, and walking
forward does not fix it. Any market priced on the closeness of the game is
wrong here, which is why the puck line stays out of primary_markets.

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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from coverline.core.distributions import BivariatePoissonDistribution
from coverline.core.interfaces import ScoreDistribution

_ROOT = Path(__file__).resolve().parents[4]
FITTED_PATH = _ROOT / "data" / "nhl_fitted.json"


class NotFitted(NotImplementedError):
    """No usable fitted parameters. Raised rather than falling back."""


def load_fitted(path: Path = FITTED_PATH) -> dict:
    """Load the graded artifact, refusing one that did not clear its gate."""
    if not Path(path).exists():
        raise NotFitted(f"no fitted NHL parameters at {path}")
    art = json.loads(Path(path).read_text())
    if not art.get("holdout_grade", {}).get("supported"):
        raise NotFitted(
            f"{Path(path).name} did not clear its held-out gate "
            f"(t = {art.get('holdout_grade', {}).get('t')}). Refusing to price "
            "from parameters that failed."
        )
    return art


def rates_from_fit(art: dict, home: str, away: str) -> tuple[float, float]:
    """Expected goals for one matchup, from the fitted ratings."""
    atk, dfn = art["attack"], art["defence"]
    for t in (home, away):
        if t not in atk or t not in dfn:
            raise KeyError(
                f"{t} has no fitted rating; refusing to price it as league "
                "average"
            )
    lam_h = math.exp(art["base_log_rate_home"] + atk[home] + dfn[away])
    lam_a = math.exp(art["base_log_rate_away"] + atk[away] + dfn[home])
    return lam_h, lam_a


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
