"""NHL and NBA: fitted, graded, and still shipping no unmeasured constant.

Both were structure-only until 2026-09-21, when within-season walk-forward
fits cleared held-out gates -- NBA t = +5.96 on a season never graded before,
NHL t = +3.02 with zero hyperparameter trials. Static cross-season fits had
failed first (t = -6.96 and -1.33), which is why these walk forward.

The parameters live in data/{nhl,nba}_fitted.json, NOT in the modules. Each
artifact carries its own grade and the loader refuses one that did not clear
it. So the original property holds: a number in these packages has earned its
place and travels with the evidence for it.
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
def test_both_packages_still_refuse_unfitted_parameters(mod):
    """NotFitted survives the transition. The leagues are fitted now; the
    refusal path is what keeps an ungraded artifact from being used later."""
    assert hasattr(mod, "NotFitted")
    assert hasattr(mod, "load_fitted")


@pytest.mark.parametrize("mod", [nhl, nba], ids=["nhl", "nba"])
def test_an_artifact_that_failed_its_gate_is_refused(mod, tmp_path):
    """The property that makes shipping these numbers acceptable. A
    regenerated fit that stopped clearing its gate must not load."""
    import json
    good = json.loads(mod.FITTED_PATH.read_text())
    good["holdout_grade"]["supported"] = False
    bad = tmp_path / "failed.json"
    bad.write_text(json.dumps(good))
    with pytest.raises(mod.NotFitted, match="gate"):
        mod.load_fitted(bad)


@pytest.mark.parametrize("mod", [nhl, nba], ids=["nhl", "nba"])
def test_a_missing_artifact_raises_rather_than_defaulting(mod, tmp_path):
    with pytest.raises(mod.NotFitted):
        mod.load_fitted(tmp_path / "absent.json")


def test_nhl_parameters_are_graded():
    art = nhl.load_fitted()
    assert art["holdout_grade"]["supported"] is True
    assert art["holdout_grade"]["t"] >= 2.0
    assert art["_provenance"]["hyperparameter_trials"] == 0, (
        "NHL had no season left to tune on; a non-zero trial count means "
        "something was searched on the only ungraded season"
    )
    assert len(art["attack"]) == 32


def test_nba_parameters_are_graded_and_clear_their_noise_ceiling():
    art = nba.load_fitted()
    assert art["holdout_grade"]["supported"] is True
    mt = art["multiple_testing"]
    assert mt["clears_noise_ceiling"] is True
    assert mt["observed_holdout_t"] > 2 * mt["expected_best_t_from_pure_noise"]
    assert len(art["ratings"]) == 30


def test_nba_sigma_is_constant_because_varying_was_rejected():
    art = nba.load_fitted()
    assert art["sigma_varies_with_spread"] is False
    assert "REJECTED" in art["sigma_note"]
    sigma = nba.constant_sigma(art)
    assert sigma(0.0) == sigma(20.0), "sigma should not vary; that was tested"


def test_nhl_records_that_its_joint_distribution_is_still_wrong():
    """The rates are graded. The joint is not, and the artifact says so --
    otherwise a future reader sees a supported grade and assumes the whole
    model is sound."""
    art = nhl.load_fitted()
    assert "joint_distribution_warning" in art
    c = art["closeness_check"]
    assert c["actual_one_goal_rate"] - c["predicted_one_goal_rate"] > 0.05
    assert "puck line" in art["joint_distribution_warning"]


def test_an_unrated_team_is_refused_by_both(tmp_path):
    with pytest.raises(KeyError, match="refusing"):
        nhl.rates_from_fit(nhl.load_fitted(), "XXX", "BOS")
    with pytest.raises(KeyError, match="refusing"):
        nba.margin_from_fit(nba.load_fitted(), "XXX", "BOS")
