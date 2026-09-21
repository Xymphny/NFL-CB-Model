"""CFB: the port, and parity against the legacy model.

The structural differences from NFL are what get tested hardest, because the
two leagues share a distribution family and nothing else. A coefficient copied
from the wrong league would still produce football-shaped numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.leagues.cfb import model as cfb  # noqa: E402
from coverline.leagues.cfb.sources import (  # noqa: E402
    CachedWalkForwardSource, GameNotInCache, game_id,
)
from coverline.leagues.nfl import model as nfl  # noqa: E402
from tests.core.test_conformance import assert_distribution_conforms  # noqa: E402

ASOF = "2024-01-15T00:00:00Z"


@pytest.fixture(scope="module")
def source():
    return CachedWalkForwardSource.load()


# ------------------------------------------------------- the coefficients ----

def test_coefficients_match_the_shipped_vectors():
    assert cfb.MARGIN_COEFFICIENTS["rating_diff"] == 15.6918
    assert cfb.MARGIN_COEFFICIENTS["elo_diff"] == 0.0673
    assert cfb.MARGIN_COEFFICIENTS_DVOA_ONLY["rating_diff"] == 39.1897


def test_cfb_has_no_home_field_term_and_nfl_does():
    """A structural difference, not an oversight. Pinned so nobody 'fixes'
    CFB by adding one."""
    assert "home_field" not in cfb.MARGIN_COEFFICIENTS
    assert "home_field" in nfl.MARGIN_COEFFICIENTS
    assert cfb.home_edge_for_equal_teams() == pytest.approx(-1.9871)


def test_the_rating_scales_are_not_comparable_between_leagues():
    """15.69 against 0.1078 is a difference of scale, not of importance.
    Comparing them as if they measured the same thing is a mistake this
    project already made once."""
    assert cfb.MARGIN_COEFFICIENTS["rating_diff"] > 100 * nfl.MARGIN_COEFFICIENTS["rating_diff"]


def test_the_elo_absent_vector_is_a_separate_fit_not_a_zero():
    """Treating 'Elo unavailable' as 'Elo present and zero' would understate
    rating_diff, which is why the fallback has its own fitted coefficient --
    more than twice the ensemble's, because it carries alone."""
    with_elo = cfb.predict_margin(cfb.GameFeatures(rating_diff=0.1, elo_diff=0.0))
    without = cfb.predict_margin(cfb.GameFeatures(rating_diff=0.1, elo_present=False))
    assert with_elo != pytest.approx(without)
    assert (cfb.MARGIN_COEFFICIENTS_DVOA_ONLY["rating_diff"]
            > 2 * cfb.MARGIN_COEFFICIENTS["rating_diff"])


def test_elo_only_moves_the_margin_when_declared_present():
    a = cfb.predict_margin(cfb.GameFeatures(0.1, elo_diff=0.0, elo_present=False))
    b = cfb.predict_margin(cfb.GameFeatures(0.1, elo_diff=500.0, elo_present=False))
    assert a == pytest.approx(b), "elo leaked into the DVOA-only path"


# ------------------------------------------------------------ dispersion ----

def test_the_margin_sd_is_the_measured_one_not_a_borrowed_one():
    """17.54 against NFL's 13.30. Using NFL's would make every CFB
    probability too confident."""
    assert cfb.MARGIN_SD == pytest.approx(17.5401, abs=1e-4)
    assert cfb.MARGIN_SD > nfl.MARGIN_SD * 1.25


def test_the_margin_sd_still_matches_the_cache_it_was_measured_from(source):
    """Recomputes it rather than trusting the constant. If the cache is
    regenerated and dispersion moves, this fails instead of the constant
    quietly describing old data."""
    pred = np.array([cfb.predict_margin(cfb.GameFeatures(
        rating_diff=float(r), elo_present=False)) for r in source.frame.rating_diff])
    resid = source.frame.actual_margin.values - pred
    assert float(resid.std(ddof=1)) == pytest.approx(cfb.MARGIN_SD, abs=1e-3)


def test_the_known_bias_is_recorded_and_not_silently_corrected():
    """The DVOA-only path under-predicts the home margin by 2.16 points
    systematically. Subtracting it would be a model change."""
    assert cfb.DVOA_ONLY_MEAN_RESIDUAL == pytest.approx(2.1644, abs=1e-4)


def test_key_numbers_are_absent_so_integer_pushes_are_withheld():
    m = cfb.CFBModel(CachedWalkForwardSource.load())
    assert m.has_key_number_correction is False
    d = m.predict(m._source.game_ids()[0], ASOF)
    assert d.has_key_number_correction is False


# ---------------------------------------------------------------- model ----

def test_it_satisfies_the_league_contract(source):
    m = cfb.CFBModel(source)
    assert m.league == "cfb"
    assert "spread" in m.primary_markets and "total" not in m.primary_markets


def test_the_distribution_conforms_to_the_shared_battery(source):
    m = cfb.CFBModel(source)
    assert_distribution_conforms(m.predict(source.game_ids()[0], ASOF), "cfb")


def test_an_unknown_game_is_refused(source):
    with pytest.raises(GameNotInCache, match="Refusing to price"):
        source.features("1999-W01-XXX-YYY", ASOF)


def test_the_cache_reports_elo_absent_because_it_is(source):
    f = source.features(source.game_ids()[0], ASOF)
    assert f.elo_present is False


# --------------------------------------------------------------- parity ----

def test_margins_match_the_legacy_cfb_model_on_every_cached_game(source):
    """1,731 real games, both implementations, one tolerance."""
    try:
        from model.cfb_prediction import predict_margin as legacy
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"legacy CFB model not importable: {exc}")

    m = cfb.CFBModel(source)
    diffs = []
    for gid in source.game_ids():
        rd = float(source.frame.loc[gid].rating_diff)
        diffs.append(m.predict(gid, ASOF).margin_mean() - legacy(rd))
    diffs = np.asarray(diffs)
    assert len(diffs) == 1731
    worst = float(np.max(np.abs(diffs)))
    assert worst < 1e-9, f"new core diverges from legacy CFB by up to {worst:.2e}"


def test_the_parity_harness_would_notice_a_coefficient_drift(source):
    """Proves the comparison exercises the coefficients rather than passing
    because both sides call the same function."""
    from model.cfb_prediction import predict_margin as legacy
    rd = float(source.frame.loc[source.game_ids()[0]].rating_diff)
    honest = legacy(rd)
    drifted = (cfb.MARGIN_COEFFICIENTS_DVOA_ONLY["rating_diff"] * 1.01 * rd
               + cfb.MARGIN_COEFFICIENTS_DVOA_ONLY["intercept"])
    assert abs(drifted - honest) > 1e-9


def test_game_ids_are_canonical():
    assert game_id(2023, 7, "GA", "BAMA") == "2023-W07-GA-BAMA"
