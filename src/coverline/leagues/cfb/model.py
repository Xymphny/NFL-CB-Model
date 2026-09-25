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

import json
from dataclasses import dataclass
from pathlib import Path
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
#:
#: AND IT SURVIVED THE CIRCLE BEING BROKEN. Measuring it on the same cache it
#: is then judged against gives a dispersion ratio of exactly 1.0000, which is
#: a tautology and not evidence. Re-estimated on 2021-2022 alone and checked
#: on 2023, which it never saw: 17.62 against a realised 17.40, a ratio of
#: 0.988, with the 95% interval covering 95.1%. Bartlett finds no evidence the
#: spread varies by season (p = 0.23). Unlike NFL's, this constant is doing
#: its job. model/grade_distributions.py has the table.
#:
#: CORRECTED 2026-09-21, from 17.5401. The cache it is measured from carries
#: 76 rows in 1,731 -- 4.39% -- with a final margin of ZERO, and college
#: football has not permitted a tie since 1996. Two upstream causes: scores
#: never fetched and stored as 0-0, and rows frozen at an intermediate score
#: (Auburn 22-22 Alabama in 2021, which Alabama won 24-22 in four overtimes).
#: Every one is also recorded as a home LOSS, so a tie became an away win.
#:
#: They shrank this constant by 1.93%, in the OVERCONFIDENT direction -- a
#: dispersion that is too small oversizes every stake that divides by it.
#: model/fit_data_checks.py now refuses a frame with more ties than its league
#: permits, so this cannot recur silently.
#:
#: CORRECTED AGAIN, same day, to 17.8047 -- and this time on REPAIRED data
#: rather than on the survivors of corrupt data. A second source (ESPN, same
#: event ids, with the completion flag the original cache lacks) showed the
#: corruption is worse than a tie check can see: 159 of 1,731 rows carry a
#: wrong score, 9.19% against the 4.39% that happened to land level, and 58
#: of them FLIP THE WINNER. The pattern is a frozen score, not a missing one
#: -- Vanderbilt 27-28 UConn was really 30-28.
#:
#: With the scores repaired there are ZERO ties across all 1,731 games, which
#: is what a sport that abolished them in 1996 should look like, and the
#: constant is measured on every game rather than on 1,655 survivors.
MARGIN_SD = 17.8047

#: The same measurement found the DVOA-only path carries a +2.16 point mean
#: residual on this cache: it under-predicts the home margin systematically,
#: not just noisily.
#:
#: NOT CORRECTED HERE. The shipped coefficients were fitted as a vector and
#: subtracting a constant from their output is a model change, which goes
#: through the gate with a held-out measurement like anything else. It is
#: recorded so that a caller comparing CFB numbers to a market knows the
#: model leans one way before the market does anything.
#:
#: NOW KNOWN TO BE SYSTEMATIC RATHER THAN A PERIOD. Every season carries it --
#: +2.11 in 2021, +1.94 in 2022, +2.44 in 2023 -- and a weighted test finds no
#: evidence it varies at all (p = 0.88). A bias that is the same size every
#: year is a model defect, not a run of luck, and it is still not corrected
#: here for the reason above.
#:
#: CORRECTED 2026-09-21, from 2.1644, for the same reason as MARGIN_SD above:
#: 76 impossible tied rows were in the measurement. Recomputing after finding
#: a data fault is the same look with a corrected estimator, not a second
#: look, and both numbers stay recorded.
#:
#: CORRECTED AGAIN to 2.0540 on the repaired scores. This one moved further
#: than the dispersion did -- 0.19 of a point -- because a frozen score
#: understates one side systematically, which biases a mean far more than it
#: biases a spread.
DVOA_ONLY_MEAN_RESIDUAL = 2.0540

#: MEASURED 2026-09-25 (data/cfb_key_numbers.json, model/cfb_key_numbers.py).
#: Fitted on 2021-2022 CFBD closes, graded once on 2023: held-out
#: log-likelihood +0.135 per game at this model's MARGIN_SD, t = +9.3. The
#: plain rounded normal puts P(margin = 3) at 1.7% against 5.7% observed, and
#: P(0) at 1.7% where overtime makes it impossible. Until this table existed,
#: every whole-number CFB spread was withheld by the recommender (ADR 0027).
#: Loaded only if the artifact says it cleared its own gate.
KEY_NUMBERS_PATH = Path(__file__).resolve().parents[4] / "data" / "cfb_key_numbers.json"


def _load_key_number_weights() -> dict[int, float] | None:
    if not KEY_NUMBERS_PATH.exists():
        return None
    art = json.loads(KEY_NUMBERS_PATH.read_text())
    if not art.get("holdout_grade", {}).get("supported"):
        return None
    return {int(k): float(v) for k, v in art["weights"].items()}


KEY_NUMBER_WEIGHTS: Mapping[int, float] | None = _load_key_number_weights()

#: No validated total model. The NFL totals model already graded
#: supported=false on 1,039 games; CFB's has never been graded at all, which
#: is a weaker position, not a stronger one.
#:
#: MEASURED, AND THE LABELS ARE THE WRONG WAY ROUND. Over 3,863 games in
#: model/cfb_schedule_cache.csv the actual total averages 51.93, so the number
#: labelled PLACEHOLDER is accurate to seven hundredths of a point (t = -0.23).
#: The number labelled merely UNVALIDATED is the dangerous one: the actual sd
#: is 18.79 against this 14.0, and because mu_total here is a CONSTANT the
#: residual and unconditional dispersions are the same quantity, so that
#: comparison is exact rather than indicative. The nominal 95% interval covers
#: 87.0%, the 80% covers 67.7%, and the 50% covers 40.7%.
#:
#: NOT CORRECTED, because a correct sd on an ungraded mean is still an
#: ungraded total, and fixing the visible half would make the market look
#: ready. The distribution REFUSES total questions instead -- see
#: UnvalidatedTotal in core.
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
            # The market is withheld and now the OBJECT refuses too. Asking
            # this distribution for a total raises UnvalidatedTotal rather
            # than returning 52.0 with a 14.0 spread, which is overconfident
            # by a third -- see the note on TOTAL_SD_UNVALIDATED.
            total_validated=False,
            discrete=True,
            key_number_weights=KEY_NUMBER_WEIGHTS,
        )
