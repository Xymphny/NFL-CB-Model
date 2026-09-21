"""The NFL package: the first league on the seam.

Two things are being checked. That the port reproduces the shipped model
exactly -- including a defect it is deliberately carrying forward -- and that
what it produces satisfies the same ScoreDistribution contract every other
league will have to satisfy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.leagues.nfl import model as nfl  # noqa: E402
from tests.core.test_conformance import assert_distribution_conforms  # noqa: E402


class StubSource:
    """Deterministic features. Stands in for the rating pipeline, which is
    not ported yet and has its own ledger row."""

    def __init__(self, **kw):
        self.kw = kw
        self.calls: list[tuple[str, str]] = []

    def features(self, game_id, asof):
        self.calls.append((game_id, asof))
        return nfl.GameFeatures(**self.kw)


def _model(**kw):
    return nfl.NFLModel(StubSource(**(kw or {"rating_diff": 1.5})))


# ---------------------------------------------------- shipped coefficients ----

def test_coefficients_match_the_shipped_vectors_exactly():
    """Ported verbatim. A drift here means the new core is pricing a
    different model from the one that was validated."""
    assert nfl.MARGIN_COEFFICIENTS["rating_diff"] == 0.1078
    assert nfl.MARGIN_COEFFICIENTS["elo_diff"] == 0.0348
    assert nfl.MARGIN_COEFFICIENTS["intercept"] == -6.6607
    assert nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY["rating_diff"] == 22.7091
    assert "elo_diff" not in nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY


def test_the_known_home_field_defect_is_carried_forward_not_fixed():
    """model/prediction.py records that home_field and intercept are
    perfectly collinear, leaving a net home edge of -1.13 against a market
    consensus near +2.5. Reproduced exactly.

    If someone 'fixes' this by adjusting a coefficient, the vector stops being
    the one that was validated, and this test is what should stop them."""
    assert nfl.home_edge_for_equal_teams() == pytest.approx(-1.1336, abs=1e-4)
    assert nfl.home_edge_for_equal_teams() < 0, (
        "the collinearity defect appears to have been corrected in place; the "
        "coefficients were validated as a vector, so changing one term "
        "invalidates the validation. Refit instead."
    )


def test_ngs_absence_selects_the_rating_only_vector():
    """The state that ran for weeks with the team rating effectively switched
    off. Presence is explicit, never inferred from zeros."""
    present = nfl.predict_margin(nfl.GameFeatures(rating_diff=0.15, elo_diff=50.0))
    absent = nfl.predict_margin(
        nfl.GameFeatures(rating_diff=0.15, elo_diff=50.0, ngs_present=False))
    assert present != absent
    # the rating-only vector ignores elo entirely
    a = nfl.predict_margin(nfl.GameFeatures(rating_diff=0.15, elo_diff=0.0,
                                            ngs_present=False))
    b = nfl.predict_margin(nfl.GameFeatures(rating_diff=0.15, elo_diff=500.0,
                                            ngs_present=False))
    assert a == pytest.approx(b), "elo leaked into the rating-only path"


def test_zero_features_and_missing_features_are_different_states():
    """Conflating them is what let the NGS outage go unnoticed."""
    zeros = nfl.GameFeatures(rating_diff=1.0, cpoe_diff=0.0, ngs_present=True)
    missing = nfl.GameFeatures(rating_diff=1.0, cpoe_diff=0.0, ngs_present=False)
    assert nfl.predict_margin(zeros) != nfl.predict_margin(missing)


def test_neutral_site_removes_home_field_only():
    home = nfl.GameFeatures(rating_diff=2.0)
    neutral = nfl.GameFeatures(rating_diff=2.0, is_neutral_site=True)
    diff = nfl.predict_margin(home) - nfl.predict_margin(neutral)
    assert diff == pytest.approx(nfl.MARGIN_COEFFICIENTS["home_field"])


# ------------------------------------------------------------- the model ----

def test_it_satisfies_the_league_contract():
    m = _model()
    assert m.league == "nfl"
    assert "spread" in m.primary_markets


def test_totals_are_withheld_from_the_market_list():
    """data/totals_validation.json records supported=false on 1,039
    walk-forward games. Withheld in code, not published with a caveat."""
    assert "total" not in _model().primary_markets
    art = json.loads((ROOT / "data" / "totals_validation.json").read_text())
    assert art["supported"] is False, (
        "totals became supported; if that is real, add 'total' to "
        "primary_markets deliberately and update this test"
    )


def test_predict_passes_asof_through_to_the_feature_source():
    """The point-in-time contract. A model that ignores asof cannot be
    walk-forward tested honestly."""
    src = StubSource(rating_diff=1.0)
    nfl.NFLModel(src).predict("game-1", "2026-09-21T00:00:00Z")
    assert src.calls == [("game-1", "2026-09-21T00:00:00Z")]


def test_the_distribution_conforms_to_the_shared_battery():
    """The same assertions every other league will face."""
    d = _model(rating_diff=2.0, elo_diff=35.0).predict("g", "2026-09-21T00:00:00Z")
    assert_distribution_conforms(d, "nfl")


def test_key_number_weights_are_loaded_and_applied():
    m = _model()
    assert m.has_key_number_correction is True
    d = m.predict("g", "2026-09-21T00:00:00Z")
    assert d.has_key_number_correction is True
    assert d.margin_pmf(3) > 2.5 * d.margin_pmf(4), (
        "the key-number weights are not reaching the distribution"
    )


def test_weights_that_failed_their_gate_would_not_be_applied(tmp_path, monkeypatch):
    """The artifact carries its own grade, and the loader honours it. A
    regenerated table that stopped clearing its gate must not ship silently."""
    bad = tmp_path / "nfl_key_numbers.json"
    art = json.loads((ROOT / "data" / "nfl_key_numbers.json").read_text())
    art["holdout_grade"]["supported"] = False
    bad.write_text(json.dumps(art))
    monkeypatch.setattr(nfl, "KEY_NUMBERS_PATH", bad)
    assert nfl._load_key_number_weights() is None


def test_a_missing_weights_file_degrades_rather_than_crashes(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl, "KEY_NUMBERS_PATH", tmp_path / "absent.json")
    m = nfl.NFLModel(StubSource(rating_diff=1.0))
    assert m.has_key_number_correction is False
    assert_distribution_conforms(m.predict("g", "2026-09-21T00:00:00Z"), "nfl-nokeys")


# ------------------------------------------------------------ end to end ----

def test_a_prediction_prices_a_spread_against_a_real_market():
    """Model to distribution to cover probability to expected value, through
    the shared pricing code -- no league-specific pricing anywhere."""
    d = _model(rating_diff=3.0, elo_diff=60.0).predict("g", "2026-09-21T00:00:00Z")

    line = -3.0  # home favoured by 3
    p_cover = 1.0 - d.margin_cdf(-line)          # home wins by more than 3
    p_push = d.margin_pmf(-line)
    assert 0.0 < p_cover < 1.0
    assert p_push > 0.05, "a push on 3 should be a material probability"

    # price it against a -110 market using only core.pricing
    market = [P.american_to_decimal(-110), P.american_to_decimal(-110)]
    fair = P.devig(market, "power")[0]
    assert fair == pytest.approx(0.5)
    ev = P.expected_value(p_cover / (1 - p_push), P.american_to_decimal(-110))
    assert np.isfinite(ev)


def test_the_push_probability_is_not_the_normal_approximation():
    """The whole reason the key-number work was done: a plain normal would
    price this push at roughly a third of its true value."""
    from coverline.core.distributions import NormalMarginDistribution
    d = _model(rating_diff=3.0).predict("g", "2026-09-21T00:00:00Z")
    plain = NormalMarginDistribution(d.margin_mean(), nfl.MARGIN_SD,
                                     d.total_mean(), nfl.TOTAL_SD_UNVALIDATED)
    assert d.margin_pmf(3) > 2.5 * plain.margin_pmf(3)
