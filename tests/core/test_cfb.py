"""CFB: the port, and parity against the legacy model.

The structural differences from NFL are what get tested hardest, because the
two leagues share a distribution family and nothing else. A coefficient copied
from the wrong league would still produce football-shaped numbers.
"""

import json
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
    """17.88 against NFL's 13.30. Using NFL's would make every CFB
    probability too confident.

    Was 17.5401 until 76 impossible tied rows were found in the cache it is
    measured from, then 17.8780 with those dropped, then 17.8047 once a
    second source showed the corruption was 159 rows rather than 76 and the
    scores could be REPAIRED rather than merely excluded.
    """
    assert cfb.MARGIN_SD == pytest.approx(17.8047, abs=1e-4)
    assert cfb.MARGIN_SD > nfl.MARGIN_SD * 1.25


def test_the_margin_sd_still_matches_the_cache_it_was_measured_from(source):
    """Recomputes it rather than trusting the constant. If the cache is
    regenerated and dispersion moves, this fails instead of the constant
    quietly describing old data."""
    # No exclusion needed any more: the scores are repaired and there are no
    # impossible rows left. The filter stays so that the day one reappears,
    # this test measures what cfb_margin_sd.py measures rather than diverging.
    frame = source.frame[source.frame.actual_margin != 0]
    pred = np.array([cfb.predict_margin(cfb.GameFeatures(
        rating_diff=float(r), elo_present=False)) for r in frame.rating_diff])
    resid = frame.actual_margin.values - pred
    assert float(resid.std(ddof=1)) == pytest.approx(cfb.MARGIN_SD, abs=1e-3)


def test_the_cache_is_repaired_and_the_repairs_are_recorded():
    """The fault IS fixed now, and the reason it was not before still holds.

    ADRs 0019 and 0020 refused to repair these rows because rewriting a
    result by hand is inventing one. That was right while there was no second
    source. There is one now -- ESPN, same event ids, with a completion flag
    -- so replacing a frozen score with what an independent pipeline recorded
    is adjudication, and every change is written to
    model/cfb_score_repairs.csv so the claim is auditable rather than trusted.
    """
    import pandas as pd

    from model.fit_data_checks import ImpossibleOutcome, check_no_impossible_ties

    g = pd.read_csv(ROOT / "model" / "cfb_full_walk_forward_cache.csv")
    clean = check_no_impossible_ties(g, "cfb", label="cfb cache",
                                     margin="actual_margin")
    assert clean["tied"] == 0, (
        "a tie is back in the constants cache; it was repaired from a second "
        "source and a reappearance means the repair was undone or the "
        "regenerator dropped the fix"
    )
    repairs = pd.read_csv(ROOT / "model" / "cfb_score_repairs.csv")
    assert len(repairs) > 200, "the repair record has shrunk"
    assert (repairs.was_home != repairs.now_home).any()


def test_the_known_bias_is_recorded_and_not_silently_corrected():
    """The DVOA-only path under-predicts the home margin by 2.16 points
    systematically. Subtracting it would be a model change."""
    assert cfb.DVOA_ONLY_MEAN_RESIDUAL == pytest.approx(2.0540, abs=1e-4)


def test_the_key_number_table_is_loaded_because_it_cleared_its_gate():
    art = json.loads((ROOT / "data" / "cfb_key_numbers.json").read_text())
    g = art["holdout_grade"]
    assert g["supported"] and g["at_fit_sigma"]["supported"] and g["at_shipped_sigma"]["supported"]
    assert g["at_shipped_sigma"]["sigma"] == pytest.approx(cfb.MARGIN_SD, abs=1e-4), (
        "graded at a width the model no longer prices with")
    m = cfb.CFBModel(CachedWalkForwardSource.load())
    assert m.has_key_number_correction is True
    d = m.predict(m._source.game_ids()[0], ASOF)
    assert d.has_key_number_correction is True
    assert d.margin_pmf(0) == 0.0, "a CFB game cannot end tied"
    assert d.margin_pmf(3) > 2 * d.margin_pmf(4)


def test_the_key_number_artifact_reproduces_from_its_script():
    """The committed weights are exactly what the script computes from the
    committed lines. A table nobody can regenerate is the ledger's oldest
    failure."""
    sys.path.insert(0, str(ROOT / "model"))
    import cfb_key_numbers as K
    art = json.loads((ROOT / "data" / "cfb_key_numbers.json").read_text())
    again = K.build(K.load_games(), art["_provenance"]["generated"])
    assert again == art


def test_an_ungraded_table_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "k.json"
    f.write_text(json.dumps({"weights": {"3": 3.0}, "holdout_grade": {"supported": False}}))
    monkeypatch.setattr(cfb, "KEY_NUMBERS_PATH", f)
    assert cfb._load_key_number_weights() is None


def test_a_whole_number_cfb_spread_now_prices():
    from coverline.execution.recommend import _can_price_push
    m = cfb.CFBModel(CachedWalkForwardSource.load())
    d = m.predict(m._source.game_ids()[0], ASOF)
    assert _can_price_push(d, -7.0) and _can_price_push(d, 3.0)


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
    """1,731 real games, both implementations, one tolerance.

    Compares the linear predictor (mu_margin), not margin_mean(): with the
    key-number table loaded, margin_mean() is the reweighted pmf's own mean,
    which the renormalisation pulls toward zero (see NormalMarginDistribution).
    The legacy model has no such table, so only the predictor is comparable."""
    try:
        from model.cfb_prediction import predict_margin as legacy
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"legacy CFB model not importable: {exc}")

    m = cfb.CFBModel(source)
    diffs = []
    for gid in source.game_ids():
        rd = float(source.frame.loc[gid].rating_diff)
        diffs.append(m.predict(gid, ASOF).mu_margin - legacy(rd))
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
