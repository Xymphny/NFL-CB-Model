"""NHL and NBA: structure without fitted coefficients.

Both packages exist so the seam is proven for all five leagues. Neither ships
a number. These tests exist to make sure that stays true -- a constant
appearing in either file is the failure the whole evidence apparatus is built
around, and it would be easy to add one while "finishing" the package.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.leagues.nba import model as nba  # noqa: E402
from coverline.leagues.nhl import model as nhl  # noqa: E402
from tests.core.test_conformance import assert_distribution_conforms  # noqa: E402

ASOF = "2026-09-21T00:00:00Z"


class Stub:
    def __init__(self, f):
        self.f = f

    def features(self, game_id, asof):
        return self.f


# ------------------------------------------------------------------ nhl ----

def test_nhl_conforms_when_given_rates():
    m = nhl.NHLModel(Stub(nhl.GameFeatures(lam_home=3.1, lam_away=2.8, lam_shared=0.3)))
    assert m.league == "nhl"
    assert_distribution_conforms(m.predict("g", ASOF), "nhl")


def test_nhl_does_not_offer_the_puck_line():
    """Hockey's most distinctive market, and the one most exposed to the
    empty-net problem. Offering it without a state-conditional layer would
    mean pricing the market whose bias is best understood and least
    corrected."""
    m = nhl.NHLModel(Stub(nhl.GameFeatures(3.1, 2.8)))
    assert "puckline" not in m.primary_markets
    assert "spread" not in m.primary_markets


def test_nhl_reports_that_empty_net_is_unmodelled():
    m = nhl.NHLModel(Stub(nhl.GameFeatures(3.1, 2.8)))
    assert m.models_empty_net is False


def test_nhl_shared_rate_defaults_to_zero_and_must_be_set_deliberately():
    """Hockey plausibly HAS the correlation baseball does not, but nobody has
    measured it here."""
    assert nhl.GameFeatures(3.1, 2.8).lam_shared == 0.0


def test_nhl_refuses_non_positive_rates():
    m = nhl.NHLModel(Stub(nhl.GameFeatures(lam_home=0.0, lam_away=2.8)))
    with pytest.raises(ValueError, match="must be positive"):
        m.predict("g", ASOF)


# ------------------------------------------------------------------ nba ----

def test_nba_requires_a_sigma_function_and_has_no_default():
    """The specific mistake a ported football model makes. NBA margin
    variance correlates ~0.60 with spread magnitude against ~-0.06 in the
    NFL, so a constant sigma is wrong in a way that looks like a calibration
    problem."""
    with pytest.raises(TypeError):
        nba.NBAModel(Stub(nba.GameFeatures(0.05, 99.0)))  # type: ignore[call-arg]


def test_nba_sigma_rises_with_the_spread():
    m = nba.NBAModel(Stub(nba.GameFeatures(0.2, 99.0)),
                     sigma=lambda mu: 11.0 + 0.10 * abs(mu),
                     margin_model=lambda f: 45.0 * f.rating_diff)
    big = m.predict("g", ASOF)
    m2 = nba.NBAModel(Stub(nba.GameFeatures(0.01, 99.0)),
                      sigma=lambda mu: 11.0 + 0.10 * abs(mu),
                      margin_model=lambda f: 45.0 * f.rating_diff)
    small = m2.predict("g", ASOF)
    assert big.margin_sd() > small.margin_sd()


def test_nba_conforms_when_given_fitted_pieces():
    m = nba.NBAModel(Stub(nba.GameFeatures(0.08, 99.0)),
                     sigma=lambda mu: 11.0 + 0.10 * abs(mu),
                     margin_model=lambda f: 45.0 * f.rating_diff)
    assert_distribution_conforms(m.predict("g", ASOF), "nba")


def test_nba_margin_is_continuous_unlike_football():
    m = nba.NBAModel(Stub(nba.GameFeatures(0.08, 99.0)),
                     sigma=lambda mu: 11.0, margin_model=lambda f: 4.0)
    assert m.predict("g", ASOF).is_discrete is False


def test_nba_records_whether_a_minutes_projection_was_applied():
    """The published work that beats this market is player-impact weighted by
    projected minutes. A team-level prediction is a weaker claim and should be
    recorded as one rather than presented as an NBA model."""
    assert nba.GameFeatures(0.08, 99.0).minutes_projected is False


def test_nba_does_not_offer_props():
    m = nba.NBAModel(Stub(nba.GameFeatures(0.08, 99.0)),
                     sigma=lambda mu: 11.0, margin_model=lambda f: 4.0)
    assert not any("prop" in x or "points" in x for x in m.primary_markets)


# ---------------------------------------------- no coefficients anywhere ----

@pytest.mark.parametrize("mod", [nhl, nba], ids=["nhl", "nba"])
def test_neither_package_ships_a_fitted_constant(mod):
    """The guard that matters. A float literal at module level in either file
    is a coefficient nobody measured."""
    src = Path(mod.__file__).read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    code = re.sub(r'""".*?"""', "", code, flags=re.S)
    literals = re.findall(r"^[A-Z_]+\s*(?:[:=])\s*[-+]?\d+\.\d+", code, re.M)
    assert not literals, (
        f"{mod.__name__} ships module-level numeric constants {literals}. "
        "Neither league has been fitted or graded; a plausible constant here "
        "is indistinguishable from a measured one six months from now."
    )


@pytest.mark.parametrize("mod", [nhl, nba], ids=["nhl", "nba"])
def test_both_packages_say_they_are_unfitted(mod):
    assert hasattr(mod, "NotFitted")
    assert "STRUCTURE ONLY" in (mod.__doc__ or "")
