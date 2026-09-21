"""Parity: does the new core price what the legacy model priced?

THE PROPERTY THAT MAKES THE MIGRATION SAFE
Nothing in the legacy system can be retired until the new core is known to
reproduce it. Not "looks similar" -- reproduces it, on real data, to a stated
tolerance, with the comparison run by code rather than by reading.

This file runs both implementations over the whole committed walk-forward
cache (1,945 games, 2014-2023) and compares margins. The legacy path is
model/prediction.predict_margin; the new path is the NFL package driven by
CachedWalkForwardSource. Both use MARGIN_COEFFICIENTS_V1_RATING_ONLY, because
that is the vector the cache's features select.

WHAT PARITY DOES NOT PROVE
That either model is any good. data/spread_validation.json already records
supported=false for the shipped thresholds. Parity says the port is faithful,
which is a claim about the migration, not about edge. Conflating the two would
be how a faithful port of a losing model gets mistaken for progress.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.leagues.nfl import model as newnfl  # noqa: E402
from coverline.leagues.nfl.sources import (  # noqa: E402
    CachedWalkForwardSource, GameNotInCache, game_id,
)

ASOF = "2024-01-15T00:00:00Z"


@pytest.fixture(scope="module")
def source():
    return CachedWalkForwardSource.load()


@pytest.fixture(scope="module")
def legacy():
    """The legacy predictor, imported directly. Skips rather than fails if the
    legacy tree is unavailable -- a parity test that cannot see one side has
    nothing to say, and saying nothing loudly beats a false pass."""
    try:
        from model.prediction import (  # type: ignore
            MARGIN_COEFFICIENTS_V1_RATING_ONLY, predict_margin,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"legacy model not importable: {exc}")
    return predict_margin, MARGIN_COEFFICIENTS_V1_RATING_ONLY


# ------------------------------------------------------------- the source ----

def test_the_cache_loads_and_covers_what_it_claims(source):
    assert len(source) == 1945
    assert source.seasons == list(range(2014, 2024))


def test_an_unknown_game_is_refused_not_neutralised(source):
    """Returning rating_diff=0 for a game it does not know would produce a
    confident number with no information in it."""
    with pytest.raises(GameNotInCache, match="Refusing to price"):
        source.features("1999-W01-XXX-YYY", ASOF)


def test_features_say_ngs_is_absent_because_it_is(source):
    """The cache has no NGS columns. Claiming otherwise would select the full
    ensemble and price every game with the wrong coefficients -- which is
    exactly what ran unnoticed for weeks before the audit."""
    f = source.features(source.game_ids(2023)[0], ASOF)
    assert f.ngs_present is False
    assert f.elo_diff == 0.0


def test_asof_before_the_season_is_rejected(source):
    gid = source.game_ids(2023)[0]
    with pytest.raises(ValueError, match="backtest indexing bug"):
        source.features(gid, "2022-01-01T00:00:00Z")
    source.features(gid, "2023-12-01T00:00:00Z")  # in-season is fine


def test_asof_must_be_a_timestamp(source):
    with pytest.raises(ValueError, match="ISO-8601"):
        source.features(source.game_ids(2023)[0], "week 4")


def test_game_ids_are_unique_and_canonical():
    assert game_id(2023, 7, "KC", "DEN") == "2023-W07-KC-DEN"


def row_features(row):
    """The cached row as the new core's GameFeatures.

    Mirrors the source adapter so the parity comparison uses the same inputs
    the model would receive, without going through predict().
    """
    return newnfl.GameFeatures(
        rating_diff=float(row.rating_diff),
        rest_diff=float(row.rest_diff),
        is_neutral_site=not bool(row.home_field),
        ngs_present=False,
    )


# ------------------------------------------------------------- parity ----

def test_margins_match_the_legacy_model_on_every_cached_game(source, legacy):
    """The whole point. 1,945 real games, both implementations, one tolerance."""
    predict_margin, coeffs = legacy
    model = newnfl.NFLModel(source)

    diffs = []
    for gid in source.game_ids():
        row = source.frame.loc[gid]
        old = predict_margin(
            rating_diff=float(row.rating_diff),
            is_neutral_site=not bool(row.home_field),
            rest_diff=float(row.rest_diff),
            coefficients=coeffs,
        )
        # predict_margin, NOT the distribution's mean.
        #
        # They were the same number until margin_mean stopped reporting
        # mu_margin and started reporting the mean of the distribution it
        # actually represents. With the shipped key-number table those differ
        # by up to 0.52 points, because renormalising a multiplicative
        # reweighting fixes the total mass and not the first moment.
        #
        # Parity is a claim about the COEFFICIENTS reproducing the legacy
        # model, so it compares the coefficient output. The gap between that
        # and the distribution's mean is a separate, real finding and is
        # asserted in test_money_path_properties.py rather than hidden by
        # widening this tolerance.
        new = newnfl.predict_margin(row_features(row))
        diffs.append(new - old)

    diffs = np.asarray(diffs)
    assert len(diffs) == 1945
    worst = float(np.max(np.abs(diffs)))
    assert worst < 1e-9, (
        f"new core diverges from the legacy model by up to {worst:.2e} points. "
        "Parity is what makes retiring the old path safe; investigate before "
        "relaxing this tolerance."
    )


def test_the_rating_only_vector_is_the_one_being_compared(legacy):
    """Guard on the comparison itself: if the new package silently switched to
    the full ensemble, margins would differ and the test above would fail --
    but for the wrong reason. Pin which vector is in play."""
    _, coeffs = legacy
    assert coeffs == dict(newnfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY)
    assert "elo_diff" not in coeffs


def test_parity_would_fail_if_a_coefficient_drifted(source, legacy):
    """Proves the harness can detect divergence, rather than passing because
    both sides call the same function."""
    predict_margin, coeffs = legacy
    tampered = dict(coeffs)
    tampered["rating_diff"] = coeffs["rating_diff"] * 1.01

    gid = source.game_ids(2023)[0]
    row = source.frame.loc[gid]
    honest = predict_margin(rating_diff=float(row.rating_diff),
                            is_neutral_site=not bool(row.home_field),
                            rest_diff=float(row.rest_diff), coefficients=coeffs)
    drifted = predict_margin(rating_diff=float(row.rating_diff),
                             is_neutral_site=not bool(row.home_field),
                             rest_diff=float(row.rest_diff), coefficients=tampered)
    assert abs(drifted - honest) > 1e-9, (
        "a 1% coefficient change produced no margin difference; the harness "
        "is not actually exercising the coefficients"
    )


# --------------------------------------------------------- sanity on data ----

def test_predictions_are_in_a_plausible_range(source):
    """Not a parity check -- a smoke test that the ported model produces
    football numbers rather than, say, thousands."""
    model = newnfl.NFLModel(source)
    margins = [model.predict(g, ASOF).margin_mean()
               for g in source.game_ids(2023)]
    assert all(-40 < m < 40 for m in margins)
    assert 0.5 < float(np.std(margins)) < 15.0


def test_the_model_is_not_better_than_the_market_and_nothing_here_claims_it_is(source):
    """Records the honest state: parity is a migration property. The shipped
    spread thresholds are already graded supported=false."""
    import json
    art = json.loads((ROOT / "data" / "spread_validation.json").read_text())
    assert art["supported"] is False
    assert art["action"] == "DISCLOSED, NOT WITHHELD"
