"""College football on the seam.

WHAT MAKES CFB DIFFERENT FROM NFL, STRUCTURALLY
Three things, and all of them show up in the coefficients rather than in the
code around them.

There is NO home_field term. NFL carries one (badly -- see the collinearity
defect in leagues/nfl/model.py); CFB's fit puts home advantage in the
intercept, so a neutral-site game is not expressible by dropping a term. That
is a real limitation of the shipped vector, recorded rather than patched.

The rating coefficient is an order of magnitude larger than NFL's ensemble
(15.69 against 0.1078) because CFB ratings are on a different scale entirely.
Comparing the two numbers as if they measured the same thing is a mistake the
audit already made once.

And the vector SPLIT is by Elo availability, not by NGS. Where NFL falls back
when player-tracking is missing, CFB falls back when Elo is missing -- to a
DVOA-only vector whose rating coefficient is 39.19, more than twice the
ensemble's, because it has to carry alone what two terms carry together.

WHY THAT FALLBACK IS NOT elo_diff = 0
The coefficients were co-calibrated with Elo present. Treating "Elo
unavailable" as "Elo available and exactly zero" would systematically
understate rating_diff's real effect, so the absent case gets its own fitted
vector. Same reasoning as NFL's NGS split, and the same trap: passing a
zero-valued feature is not the same as declaring it missing.

THE WIDER SPREAD IS REAL
CFB margins are far more dispersed than NFL's -- mismatches between a
playoff team and a bottom-tier program have no NFL analogue -- so the
residual sd here is measured separately and is much larger. It is NOT the
NFL number, and using NFL's would make every CFB probability too confident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from coverline.core.distributions import NormalMarginDistribution
from coverline.core.interfaces import ScoreDistribution

#: Verbatim from model/cfb_prediction.py. Used when Elo is available.
MARGIN_COEFFICIENTS: Mapping[str, float] = {
    "rating_diff": 15.6918,
    "elo_diff": 0.0673,
    "intercept": -1.9871,
}

#: Verbatim. Used when Elo is NOT available -- a separate fit, not the
#: ensemble with a zero substituted in.
MARGIN_COEFFICIENTS_DVOA_ONLY: Mapping[str, float] = {
    "rating_diff": 39.1897,
    "intercept": 0.8030,
}

#: MEASURED from model/cfb_full_walk_forward_cache.csv (1,731 games,
#: 2021-2023) by model/cfb_margin_sd.py, which recomputes it and which a
#: guard test holds this constant to.
#:
#: 17.54 against NFL's 13.30 -- CFB margins really are that much more
#: dispersed, and borrowing NFL's number would make every CFB probability
#: too confident. This constant was first written as a guessed 18.65; the
#: measurement disagreed, and the measurement won.
MARGIN_SD = 17.5401

#: The same measurement found the DVOA-only path carries a +2.16 point mean
#: residual on this cache: it under-predicts the home margin systematically,
#: not just noisily.
#:
#: NOT CORRECTED HERE. The shipped coefficients were fitted as a vector and
#: subtracting a constant from their output is a model change, which goes
#: through the gate with a held-out measurement like anything else. It is
#: recorded so that a caller comparing CFB numbers to a market knows the
#: model leans one way before the market does anything.
DVOA_ONLY_MEAN_RESIDUAL = 2.1644

#: Absent, and deliberately. CFB key numbers (3 and 7 again, but with a much
#: wider margin distribution diluting them) have not been measured the way
#: data/nfl_key_numbers.json measures NFL's. Pricing a push on an integer CFB
#: line is therefore withheld by the recommender, which checks
#: has_key_number_correction.
KEY_NUMBER_WEIGHTS: Mapping[int, float] | None = None

#: No validated total model. The NFL totals model already graded
#: supported=false on 1,039 games; CFB's has never been graded at all, which
#: is a weaker position, not a stronger one.
TOTAL_MEAN_PLACEHOLDER = 52.0
TOTAL_SD_UNVALIDATED = 14.0


@dataclass(frozen=True)
class GameFeatures:
    """CFB inputs. Note the absence of rest and home_field.

    `elo_present` is explicit for the same reason NFL's `ngs_present` is: a
    genuine zero and a missing measurement select different fitted vectors,
    and inferring one from the other is how a feed outage silently changes
    which model is running.
    """

    rating_diff: float
    elo_diff: float = 0.0
    elo_present: bool = True


class FeatureSource(Protocol):
    def features(self, game_id: str, asof: str) -> GameFeatures: ...


def predict_margin(f: GameFeatures) -> float:
    """The shipped linear model, including the split by Elo availability."""
    if not f.elo_present:
        c = MARGIN_COEFFICIENTS_DVOA_ONLY
        return c["intercept"] + c["rating_diff"] * f.rating_diff
    c = MARGIN_COEFFICIENTS
    return (c["intercept"] + c["rating_diff"] * f.rating_diff
            + c["elo_diff"] * f.elo_diff)


def home_edge_for_equal_teams(elo_present: bool = True) -> float:
    """CFB has no home_field term, so this is just the intercept.

    Exposed to make the difference from NFL visible: NFL's equivalent is
    home_field + intercept and comes out at -1.13. Here there is only one
    constant, and it is doing both jobs.
    """
    c = MARGIN_COEFFICIENTS if elo_present else MARGIN_COEFFICIENTS_DVOA_ONLY
    return c["intercept"]


class CFBModel:
    """Implements core.interfaces.LeagueModel."""

    def __init__(self, source: FeatureSource) -> None:
        self._source = source

    @property
    def league(self) -> str:
        return "cfb"

    @property
    def primary_markets(self) -> Sequence[str]:
        """Totals absent: never graded, which is weaker than NFL's graded
        failure, not stronger."""
        return ("spread", "moneyline")

    @property
    def has_key_number_correction(self) -> bool:
        return KEY_NUMBER_WEIGHTS is not None

    def predict(self, game_id: str, asof: str) -> ScoreDistribution:
        f = self._source.features(game_id, asof)
        return NormalMarginDistribution(
            mu_margin=predict_margin(f),
            sd_margin=MARGIN_SD,
            mu_total=TOTAL_MEAN_PLACEHOLDER,
            sd_total=TOTAL_SD_UNVALIDATED,
            discrete=True,
            key_number_weights=KEY_NUMBER_WEIGHTS,
        )
