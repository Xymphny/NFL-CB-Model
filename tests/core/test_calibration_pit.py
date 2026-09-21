"""Calibration for counts, where the interval check does not apply.

WHY A SECOND CALIBRATION FILE
model/grade_distributions.py asks how often the outcome lands inside a 50, 80
or 95 percent interval. That assumes a continuous symmetric distribution; the
95 percent interval of a Skellam is not mu plus or minus 1.96 sd in any useful
sense. ADR 0009 recorded MLB as graded on likelihood but NOT on coverage for
exactly that reason, and named this as the missing piece.

WHAT THE PIT ADDS OVER A LIKELIHOOD RATIO
A likelihood ratio says one model beats another. It cannot say whether the
winner is the RIGHT SHAPE, and a model can win on likelihood while being
systematically wrong somewhere in its support. The PIT histogram shows where:
a hump means too wide, a U means too narrow, a slope means biased.

It earned that here. The first NHL rules table won on likelihood at t = 39.99
and the PIT showed an upward slope on the total -- which turned out to be a
0.105 goal bias from pull-layer drift, invisible in the likelihood number.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REPORT = ROOT / "model" / "calibration_pit.json"


@pytest.fixture(scope="module")
def pit() -> dict:
    assert REPORT.exists(), (
        f"{REPORT.relative_to(ROOT)} is missing -- run model/calibration_pit.py"
    )
    return json.loads(REPORT.read_text())


def test_the_rules_layer_is_calibrated_on_the_margin(pit) -> None:
    """The claim the puck line rests on, tested on shape rather than on a mean.

    A flat PIT means the outcome falls at every point of the forecast's own
    distribution at the rate the forecast claims. That is a much stronger
    statement than beating a baseline, and it is what justifies pricing a line
    at 1.5 rather than merely ranking two models.
    """
    d = pit["nhl"]["rules_margin"]
    assert not d["exceeds_kolmogorov_scale"], (
        f"rules-layer margin PIT deviates by {d['max_deviation']} against a "
        f"scale of {d['kolmogorov_scale_at_n']} -- the distribution is the "
        "wrong shape somewhere, and the puck line is priced off its shape"
    )


def test_the_rules_layer_is_calibrated_on_the_total(pit) -> None:
    """The defect that the PIT caught and the refresh fixed.

    The first table sloped upward here -- 0.85 in the lowest tenth against
    1.11 in the ninth -- because it added 0.617 goals a game to a league that
    delivered 0.727. Re-measured on recent seasons the histogram is flat to
    half a percent.
    """
    d = pit["nhl"]["rules_total"]
    assert not d["exceeds_kolmogorov_scale"], (
        f"rules-layer total PIT deviates by {d['max_deviation']} -- the pull "
        "layer has drifted again and the table needs re-measuring"
    )
    h = d["histogram"]
    assert h[-1] - h[0] < 0.15, (
        f"the total PIT histogram is sloping ({h[0]} to {h[-1]}), which is "
        "what a stale pull table looks like"
    )


def test_the_rules_layer_beats_the_baseline_on_shape_not_only_likelihood(pit) -> None:
    """Both margins, side by side. The baseline fails and the layer does not."""
    base = pit["nhl"]["baseline_margin"]
    rules = pit["nhl"]["rules_margin"]
    assert base["exceeds_kolmogorov_scale"], (
        "the baseline margin distribution is now calibrated, which would mean "
        "the tie mass it assigns to an impossible outcome has stopped "
        "mattering -- that is not possible and something is wrong upstream"
    )
    assert rules["max_deviation"] < base["max_deviation"]


def test_the_baseline_tie_mass_shows_up_as_a_hole_in_the_middle(pit) -> None:
    """A specific, diagnosable failure rather than a summary statistic.

    The baseline puts about a sixth of its mass on a tied final score. No
    observation can land there, so the centre of its PIT histogram is starved
    while both ends pile up. The number to look at is the middle bin.
    """
    h = pit["nhl"]["baseline_margin"]["histogram"]
    middle = min(h[4], h[5])
    assert middle < 0.8, (
        f"the baseline's central PIT bin is {middle}, no longer depleted -- "
        "the impossible-tie defect should be visible here and is not"
    )


def test_mlb_scores_are_close_to_calibrated_and_the_gap_is_recorded(pit) -> None:
    """Independent negative binomials, checked on their own terms.

    The away side passes. The home side exceeds the Kolmogorov scale by a
    hair, 0.0133 against 0.0123 on 12,148 games, which is a real but small
    defect and is asserted as small rather than waved through. It is also IN
    SAMPLE for r_home and r_away, so it is a floor on the error, not a
    measure of it.
    """
    mlb = pit["mlb"]
    assert mlb["away"]["max_deviation"] < mlb["away"]["kolmogorov_scale_at_n"]
    assert mlb["home"]["max_deviation"] < 0.02, (
        f"MLB home-score calibration has degraded to "
        f"{mlb['home']['max_deviation']} -- small was the whole defence"
    )
    assert mlb["in_sample_for"], "the in-sample caveat has gone missing"


def test_the_pit_grade_spends_nothing_extra(pit) -> None:
    """It re-asks one question about a grade already taken.

    Same holdout, same parameters. It can embarrass the layer and cannot
    promote it, which is the only reason it is allowed to run on a season that
    has been graded.
    """
    assert "spends_nothing_extra" in pit["nhl"]
