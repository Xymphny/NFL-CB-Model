"""Baseball on the seam.

WHAT THE LEGACY MODEL PRODUCES, AND WHY THAT FITS CLEANLY
model/mlb_model.py runs a credibility-weighted walk-forward over starters,
bullpens, lineups and parks, and emits `exp_home` and `exp_away` -- expected
runs per side. Those are exactly the two parameters a count distribution
needs, so the port is a thin adapter rather than a reimplementation.

THE DISTRIBUTION IS NOT POISSON, AND THAT WAS MEASURED
Baseball is the sport people reach for Poisson with, and over 12,148
walk-forward games it does not hold: runs are overdispersed by about 2.2x
(variance/mean 2.15 home, 2.37 away, against Poisson's 1). Pricing with a
Poisson would understate run variance by more than half.

The shared-component Poisson was the natural fix and is also wrong: it buys
total variance by creating positive correlation between the scores, and the
measured correlation is +0.0006. Independence is confirmed rather than
assumed -- the measured margin variance (20.11) and total variance (20.13)
both equal the sum of the individual variances (20.12), which is what
independence predicts and what a shared component would break.

So: independent negative binomials, dispersion measured per side.

CREDIBILITY IS THE LEGACY'S JOB, NOT THIS MODULE'S
Small-sample starters are regressed toward league by the walk-forward's own
credibility denominators. This adapter must not re-shrink what has already
been shrunk, so it takes exp_home and exp_away as given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from coverline.core.distributions import NegativeBinomialScoreDistribution
from coverline.core.interfaces import ScoreDistribution

#: Dispersion, measured over 12,148 walk-forward games by
#: model/mlb_dispersion.py. r is the negative binomial's shape: variance is
#: mu + mu^2/r, so smaller r means more overdispersion. Poisson is r -> inf.
R_HOME = 3.8880
R_AWAY = 3.2426

#: Sanity bounds on expected runs. A walk-forward row outside these is a bug
#: upstream, not a real game, and pricing it would produce confident nonsense.
MIN_EXPECTED_RUNS = 1.0
MAX_EXPECTED_RUNS = 15.0


@dataclass(frozen=True)
class GameFeatures:
    """Expected runs per side, as the walk-forward produced them."""

    exp_home: float
    exp_away: float


class FeatureSource(Protocol):
    def features(self, game_id: str, asof: str) -> GameFeatures: ...


class MLBModel:
    """Implements core.interfaces.LeagueModel."""

    def __init__(self, source: FeatureSource,
                 r_home: float = R_HOME, r_away: float = R_AWAY) -> None:
        self._source = source
        self._r_home = r_home
        self._r_away = r_away

    @property
    def league(self) -> str:
        return "mlb"

    @property
    def primary_markets(self) -> Sequence[str]:
        """Moneyline first: baseball's primary market is the winner, with the
        runline a fixed 1.5 rather than a variable spread.

        Totals are included here and NOT in the football packages, because the
        reason totals are withheld there is a graded failure on the football
        model. No such grading exists for MLB either way -- so this is an
        untested market being offered, which is recorded in the ledger row
        rather than hidden behind a confident-looking list.

        THE MONEYLINE IS WITHHELD, and it is the market that looked safest.
        Measured over 12,148 games in model/mlb_rules_structure.json: the
        model's conditioned P(home) averages 0.5063 against an actual home win
        rate of 0.5315. A 2.5 point understatement, systematic, in one
        direction, on every game -- which is several times a typical edge and
        would manufacture a false edge on the away side of every card.

        The cause is two league RULES, which is the shape ADR 0007 found in
        hockey. Extra innings resolve every game, so the 10.07% this model
        puts on a tied final score is impossible. And the home team never bats
        in the ninth while leading -- a walk-off ends play the instant it
        takes the lead -- so P(home wins by exactly one) is 0.1725 against
        0.1111 for the away side. recommend.py conditions the tie out, but it
        redistributes that mass PROPORTIONALLY and the walk-off rule gives it
        overwhelmingly to the home side.

        THE RUNLINE STAYS, and it is right for a reason that could change.
        The fictitious tie mass and the missing one-run wins sit on the SAME
        side of 1.5, so the errors cancel where that market is priced: 0.6420
        below the line against an actual 0.6409. That is the measurement, not
        an argument, and if either component moves the cancellation goes with
        it.
        """
        return ("runline", "total")

    @property
    def has_key_number_correction(self) -> bool:
        """Not applicable: baseball has no key numbers in the football sense.
        The count distribution already puts mass exactly where it belongs."""
        return True

    def predict(self, game_id: str, asof: str) -> ScoreDistribution:
        f = self._source.features(game_id, asof)
        for name, v in (("exp_home", f.exp_home), ("exp_away", f.exp_away)):
            if not MIN_EXPECTED_RUNS <= v <= MAX_EXPECTED_RUNS:
                raise ValueError(
                    f"{game_id}: {name} = {v:.3f} is outside "
                    f"[{MIN_EXPECTED_RUNS}, {MAX_EXPECTED_RUNS}]. That is an "
                    "upstream bug, not a real game; refusing to price it."
                )
        return NegativeBinomialScoreDistribution(
            mu_home=f.exp_home, mu_away=f.exp_away,
            r_home=self._r_home, r_away=self._r_away,
        )
