"""The MLB live source, and the staleness it refuses to hide.

An in-season cache is refreshed by a scheduled job, so it always lags. A
source that quietly served three-day-old expected runs for tonight's game
would be confidently wrong, which is worse than being unavailable -- so the
lag is reported and a game outside the cache is refused rather than
approximated.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.leagues.mlb.model import MLBModel  # noqa: E402
from coverline.leagues.mlb.sources import (  # noqa: E402
    GameNotPriceable, WalkForwardSource,
)
from tests.core.test_conformance import assert_distribution_conforms  # noqa: E402

ASOF = "2026-09-21T00:00:00Z"


@pytest.fixture(scope="module")
def source():
    return WalkForwardSource.load()


def test_it_loads_a_full_season(source):
    assert len(source) > 2000
    assert source.as_of_date.startswith("2026-")


def test_rows_without_an_expectation_are_dropped_not_served(source):
    """A NaN expected-runs value reaching the distribution would fail far from
    its cause. The walk-forward refuses to ingest postponed games; this
    refuses to serve rows it produced nothing for."""
    import math
    for gid in source.game_ids()[:200]:
        f = source.features(gid, ASOF)
        assert math.isfinite(f.exp_home) and math.isfinite(f.exp_away)


def test_a_game_outside_the_cache_is_refused(source):
    with pytest.raises(GameNotPriceable, match="Refusing to approximate"):
        source.features("NOT_A_GAME_KEY", ASOF)


def test_staleness_is_reported_rather_than_hidden(source):
    """The number a caller needs to decide whether to trust tonight's price."""
    assert source.staleness_days(source.as_of_date) == 0
    assert source.staleness_days("2026-09-25") > 0


def test_the_distribution_conforms(source):
    m = MLBModel(source)
    assert_distribution_conforms(m.predict(source.game_ids()[-1], ASOF), "mlb")


def test_predictions_are_baseball_shaped(source):
    m = MLBModel(source)
    for gid in source.game_ids()[-40:]:
        d = m.predict(gid, ASOF)
        assert 5.0 < d.total_mean() < 14.0, gid
        assert abs(d.margin_mean()) < 4.0, gid


def test_a_favourite_reads_as_a_favourite(source):
    """The invariant the moneyline bug violated: a positive expected margin
    must imply a win probability above 0.5 once the tie is conditioned out."""
    from coverline.execution.recommend import cover_probability
    m = MLBModel(source)
    checked = 0
    for gid in source.game_ids()[-200:]:
        d = m.predict(gid, ASOF)
        if abs(d.margin_mean()) < 0.15:
            continue
        p, _ = cover_probability(d, 0.0)
        assert (p > 0.5) == (d.margin_mean() > 0), (
            f"{gid}: margin {d.margin_mean():+.3f} but P(home) {p:.4f}"
        )
        checked += 1
    assert checked > 50, "too few decisive games to prove anything"
