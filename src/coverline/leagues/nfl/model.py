"""NFL: the first league on the seam.

WHAT IS PORTED AND WHAT IS NOT
This is the pricing-facing leaf of the legacy NFL model: given a game's
features, produce a ScoreDistribution. It carries the SHIPPED coefficient
vectors verbatim from model/prediction.py, including the ensemble/rating-only
split the live board actually uses.

Deliberately NOT ported yet: the rating pipeline that produces rating_diff and
the NGS feature differences. That is upstream, has its own dependencies, and
gets its own ledger row. Until it is ported, features arrive through an
injected FeatureSource -- the same pattern the odds client uses for transport,
and for the same reason: it makes this testable today rather than after a
large migration.

A DEFECT CARRIED FORWARD RATHER THAN QUIETLY FIXED
The shipped coefficients have a known collinearity problem, recorded in
model/prediction.py: every training row had home_field=1, so home_field and
intercept are perfectly collinear and the fit split one constant arbitrarily
between them. Net home edge for two equal teams is 5.5271 - 6.6607 = -1.13
points, against a market consensus near +2.5. That is a real defect in a
validated artifact.

NOT A LIVE DEFECT (measured 2026-09-25, data/nfl_neutral_site.json). It is a
property of the FULL-ENSEMBLE vector, and no live NFL price uses that vector:
RatingsSnapshotSource sets ngs_present=False on every game, so live prices
come from MARGIN_COEFFICIENTS_V1_RATING_ONLY, whose equal-team home edge is
+1.65. That vector's own neutral-site handling -- drop its 2.83-point home
term -- was then checked on the 34 neutral games in the walk-forward cache:
it sits 1.60 points further below the closing line there than at home sites
(SE 0.62). A real lean, too few games to fit, left uncorrected and recorded.

It is reproduced here exactly, and ``home_edge_for_equal_teams`` exposes it so
it cannot be forgotten. Silently correcting it would break the thing that makes
the coefficients worth having -- they were validated as a vector, and changing
one term invalidates the validation. Annotate, do not adjust. The fix belongs
at the next refit, by dropping home_field or including neutral-site games so
the two are identifiable.

WHERE SIGMA COMES FROM
The residual standard deviation around the closing line, measured on nflverse
2010-2021: 13.2979. Held constant across games, which is defensible for the
NFL -- margin variance correlates about -0.06 with spread magnitude here,
against 0.60 in the NBA, so a constant sigma is right for football and would be
wrong for basketball. The NBA package must not copy this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from coverline.core.distributions import NormalMarginDistribution
from coverline.core.interfaces import ScoreDistribution
from coverline.core.registry import register

_ROOT = Path(__file__).resolve().parents[4]
KEY_NUMBERS_PATH = _ROOT / "data" / "nfl_key_numbers.json"

#: Verbatim from model/prediction.py. The full ensemble, used when NGS
#: features are present. See the module docstring on home_field/intercept.
MARGIN_COEFFICIENTS: Mapping[str, float] = {
    "rating_diff": 0.1078,
    "home_field": 5.5271,
    "rest_diff": 0.0229,
    "cpoe_diff": 0.4622,
    "separation_diff": 2.7941,
    "yac_oe_diff": 0.1230,
    "ryoe_diff": 0.3432,
    "elo_diff": 0.0348,
    "intercept": -6.6607,
}

#: Verbatim from model/prediction.py. Selected whenever NGS data is absent.
#: Carries NO elo term -- which is why an NGS outage removes Elo from every
#: game on the board, the exposure recorded in the attempt log as
#: elo-term-for-ngs-absent-games (held out t=1.44, not shipped).
MARGIN_COEFFICIENTS_V1_RATING_ONLY: Mapping[str, float] = {
    "rating_diff": 22.7091,
    "home_field": 2.8321,
    "rest_diff": 0.1310,
    "intercept": -1.1811,
}

TOTAL_COEFFICIENTS: Mapping[str, float] = {
    "combined_offense": 12.4186,
    "wind": -0.2801,
    "intercept": 47.1185,
}

#: Residual sd around the closing line, nflverse 2010-2021 REG games.
#:
#: MEASURED AGAINST ITSELF AND FOUND TO BE A COMPROMISE. Graded in
#: model/grade_distributions.py over 1,945 walk-forward games, 2014-2023: the
#: realised residual sd is 13.75 pooled, a dispersion ratio of 1.034, and the
#: 95% interval covers 93.2% rather than 95% -- tails thinner than this
#: constant implies.
#:
#: The season table is the real finding. Residual sd runs from 11.73 in 2022
#: to 15.15 in 2021 and Bartlett rejects equal variance across the ten seasons
#: at p = 0.014, against a sampling SE of about 0.70 for one season's estimate.
#: So a single constant cannot be right in every season, and Kelly divides by
#: exactly this number.
#:
#: NOT CHANGED, because knowing the value moves is not the same as being able
#: to forecast it. A season-varying sd would have to predict next season's
#: dispersion and be graded on that, and no such forecast exists. The
#: compromise is recorded rather than replaced.
#:
#: The BIAS, unlike the spread, is not systematic: season mean residuals run
#: -1.90 to +1.55 and are consistent with noise at p = 0.24. There is nothing
#: to de-bias here, which is the opposite of CFB's finding.
MARGIN_SD = 13.2979

#: Placeholder, and marked as such. No held-out measurement of total
#: dispersion has been made, so this is the one number here without evidence
#: behind it. NFLTotalsWithheld below is why that is survivable.
#:
#: AND IT IS NOT ONLY UNVALIDATED, IT IS WRONG. data/totals_validation.json
#: -- the artifact that withheld this market in the first place -- records the
#: totals model's RMSE at 13.353 and the actual total sd at 13.678. So the
#: residual dispersion is about 13.35 and this constant is 10.0, understating
#: it by a third. It was sitting in the same file as the number that
#: contradicts it.
#:
#: NOT CORRECTED. The market failed its gate; replacing a wrong sd with a
#: right one would make a withheld market look ready. The distribution now
#: REFUSES total questions instead -- see UnvalidatedTotal in core.
TOTAL_SD_UNVALIDATED = 10.0


@dataclass(frozen=True)
class GameFeatures:
    """Everything the margin and total models consume for one game.

    ``ngs_present`` decides which coefficient vector is used, and is explicit
    rather than inferred from whether the NGS fields are zero -- a genuine
    zero difference and a missing measurement are different states, and
    conflating them is what let an NGS outage run for weeks with the team
    rating effectively switched off.
    """

    rating_diff: float
    elo_diff: float = 0.0
    rest_diff: float = 0.0
    cpoe_diff: float = 0.0
    separation_diff: float = 0.0
    yac_oe_diff: float = 0.0
    ryoe_diff: float = 0.0
    combined_offense: float = 0.0
    wind: float = 0.0
    is_neutral_site: bool = False
    ngs_present: bool = True


class FeatureSource(Protocol):
    """Supplies point-in-time features. Injected; see the module docstring."""

    def features(self, game_id: str, asof: str) -> GameFeatures: ...


def _load_key_number_weights() -> dict[int, float] | None:
    if not KEY_NUMBERS_PATH.exists():
        return None
    art = json.loads(KEY_NUMBERS_PATH.read_text())
    if not art.get("holdout_grade", {}).get("supported"):
        # Refuse to apply weights that did not clear their own gate.
        return None
    return {int(k): float(v) for k, v in art["weights"].items()}


def predict_margin(f: GameFeatures) -> float:
    """The shipped linear model, including the coefficient split by NGS presence."""
    c = MARGIN_COEFFICIENTS if f.ngs_present else MARGIN_COEFFICIENTS_V1_RATING_ONLY
    m = c["intercept"] + c["rating_diff"] * f.rating_diff + c["rest_diff"] * f.rest_diff
    if not f.is_neutral_site:
        m += c["home_field"]
    if f.ngs_present:
        m += (c["cpoe_diff"] * f.cpoe_diff
              + c["separation_diff"] * f.separation_diff
              + c["yac_oe_diff"] * f.yac_oe_diff
              + c["ryoe_diff"] * f.ryoe_diff
              + c["elo_diff"] * f.elo_diff)
    return m


def predict_total(f: GameFeatures) -> float:
    c = TOTAL_COEFFICIENTS
    return c["intercept"] + c["combined_offense"] * f.combined_offense + c["wind"] * f.wind


def home_edge_for_equal_teams(ngs_present: bool = True) -> float:
    """The known defect, exposed as a number rather than a comment.

    Returns home_field + intercept: the model's margin for two identical teams
    at a non-neutral site with no rest difference. The full ensemble gives
    -1.13 against a market consensus near +2.5.
    """
    c = MARGIN_COEFFICIENTS if ngs_present else MARGIN_COEFFICIENTS_V1_RATING_ONLY
    return c["home_field"] + c["intercept"]


class NFLModel:
    """Implements core.interfaces.LeagueModel."""

    def __init__(self, source: FeatureSource,
                 key_number_weights: Mapping[int, float] | None = None) -> None:
        self._source = source
        self._weights = (dict(key_number_weights) if key_number_weights is not None
                         else _load_key_number_weights())

    @property
    def league(self) -> str:
        return "nfl"

    @property
    def primary_markets(self) -> Sequence[str]:
        """Totals are absent deliberately.

        data/totals_validation.json records supported=false: the totals model
        was graded on 1,039 walk-forward games and did not clear break-even at
        any threshold. It is withheld in code rather than published with a
        caveat, so it does not appear here and nothing downstream can price it
        by accident.
        """
        return ("spread", "moneyline")

    @property
    def has_key_number_correction(self) -> bool:
        return self._weights is not None

    def predict(self, game_id: str, asof: str) -> ScoreDistribution:
        f = self._source.features(game_id, asof)
        return NormalMarginDistribution(
            mu_margin=predict_margin(f),
            sd_margin=MARGIN_SD,
            mu_total=predict_total(f),
            sd_total=TOTAL_SD_UNVALIDATED,
            # Withheld in the market list and now in the object. The totals
            # model graded supported=false on 1,039 games, and the sd beside
            # it is 10.0 against the 13.353 RMSE that same artifact records --
            # overconfident by a third on a market that already failed.
            total_validated=False,
            discrete=True,
            key_number_weights=self._weights,
        )


def build(source: FeatureSource) -> NFLModel:
    """Construct and register. Registration is separate from import so tests
    can build unregistered instances freely."""
    return register(NFLModel(source))
