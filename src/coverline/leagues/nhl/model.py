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
against an actual 38.1%. That much is right, and the reason first recorded
here for it was WRONG.

It said the two scores are negatively correlated at -0.14, so real games stay
closer than independent rates allow. Negative covariance makes margins MORE
dispersed, not less: Var(H - A) = Var(H) + Var(A) - 2Cov(H, A). The stated
mechanism predicts FEWER one-goal games than independence, which is the
opposite of the observed problem, and the measured dispersion ratio over
9,576 games is 1.06 -- above one, as the algebra says it must be.

WHAT IS ACTUALLY HAPPENING is two league RULES, and neither is a correlation.

THE OVERTIME RULE IS DETERMINISTIC. No NHL game ends tied, and all 2,166
games decided after regulation across 2016-2023 end at a margin of EXACTLY
one, with no exceptions. So roughly 22% of the distribution's tie mass is
relocated onto plus and minus one by fiat. That single rule is the larger
part of the ten points, and no amount of fitting reaches it, because it is not
a scoring process.

EMPTY-NET GOALS ARE CONDITIONAL ON THE SCORE. 92% of them come with the
scoring team leading by one or two. A team down two pulls its goalie and
rarely comes back; a team down one pulls and often ties, vanishing into
overtime and back out at one. The result is a NON-MONOTONE margin
distribution -- more three-goal games than two-goal games, 22.7% against
20.3% in the league's own data and 25.1% against 20.2% in a second source --
across exactly the 1.5 line the puck line is priced on. Stripping empty-net
goals out of the final scores restores monotonicity, which is what makes this
the mechanism rather than a coincidence.

SO THE REPAIR IS A RULES LAYER, NOT A FITTED CORRELATION. And a fitted
correlation would be the wrong object anyway: it drifts from -0.055 in 2017 to
-0.142 in 2023 as the league pulls goalies earlier every year, so a pooled
estimate describes no season and a recent one has no reason to persist.

Measured in model/measure_nhl_joint.py, recorded in
model/nhl_joint_structure.json, pinned in tests/core/test_nhl_joint_structure.py.
Until that layer exists the puck line stays out of primary_markets.

THE FAMILY, AND THE ONE THING KNOWN TO BREAK IT
Low-scoring counts, so the Poisson family, with two qualifications recorded
now because they will otherwise be rediscovered expensively:

EMPTY-NET GOALS. Now measured rather than estimated, over 56,834 goals in
2016-2023: 5.15% of goals and 0.306 a game, rising from 0.270 a game in
2016-2017 to 0.348 in 2022-2023 while the average one moved about twelve
seconds earlier in the third period. The earlier note here said 7% and 0.42 a
game, which was a guess and is high; it may yet be right for 2024-2025, where
no goal-level data has been pulled.

That note also said about 57% of games are one-goal games. As played the
figure is 42.6%. Both can be defended and they are not the same quantity:
55.5% of games are within one goal once empty-net goals are stripped out,
which is the convention the 57% came from. Stated here because the ambiguity
is exactly the kind that silently changes a puck-line price.

They are not random. 92% arrive with the scoring team leading by one or two,
so they inflate the winner's score in precisely the games that decide the
puck line, and a homogeneous scoring process over-prices the underdog on -1.5
and under-prices the favourite. Whoever fits this needs a state-conditional
layer, not a flat adjustment.

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
