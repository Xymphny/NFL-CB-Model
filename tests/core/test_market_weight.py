"""The market-relative staking weight: estimator, artifact, and its use.

The estimator is checked on data where the right answer is known, because a
weight that always came back zero would look exactly like the NFL result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core import market_weight as MW  # noqa: E402


def _synthetic(true_w, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    pk = rng.uniform(0.3, 0.7, n)
    pm = np.clip(pk + rng.normal(0, 0.06, n), 0.02, 0.98)
    q = np.clip(true_w * pm + (1 - true_w) * pk, 0.01, 0.99)
    return pm, pk, (rng.uniform(size=n) < q).astype(int)


@pytest.mark.parametrize("true_w", [0.0, 0.5, 1.0, -0.5])
def test_the_estimator_recovers_a_known_weight(true_w):
    r = MW.estimate(*_synthetic(true_w))
    assert abs(r.w_hat - true_w) < 3 * r.se
    assert r.se < 0.2


def test_a_model_with_real_information_gets_a_positive_stake():
    r = MW.estimate(*_synthetic(1.0))
    assert 0.5 < r.staking_weight <= 1.0
    assert r.staking_weight == pytest.approx(r.w_hat - MW.Z_ONE_SIDED * r.se)


@pytest.mark.parametrize("true_w", [0.0, -0.5])
def test_noise_or_worse_stakes_nothing(true_w):
    assert MW.estimate(*_synthetic(true_w)).staking_weight == 0.0


def test_the_maximum_is_the_maximum():
    pm, pk, y = _synthetic(0.4, seed=3)
    r = MW.estimate(pm, pk, y)

    def ll(w):
        q = pk + w * (pm - pk)
        return np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    assert ll(r.w_hat) >= max(ll(r.w_hat - 0.01), ll(r.w_hat + 0.01))


def test_bad_input_is_refused():
    pm, pk, y = _synthetic(0.0, n=100)
    with pytest.raises(ValueError, match="30"):
        MW.estimate(pm[:10], pk[:10], y[:10])
    with pytest.raises(ValueError, match="pushes"):
        MW.estimate(pm, pk, np.where(y == 1, 0.5, 0))
    with pytest.raises(ValueError, match="inside"):
        MW.estimate(np.where(pm > 0.5, 1.0, pm), pk, y)


# ----------------------------------------------------------- the artifact --

def test_a_league_with_no_grade_stakes_nothing(tmp_path):
    assert MW.staking_weight("nba") == (0.0, None)
    assert MW.staking_weight("nfl", tmp_path / "absent.json") == (0.0, None)


def test_a_hand_edited_weight_is_refused(tmp_path):
    art = json.loads(MW.WEIGHTS_PATH.read_text())
    art["leagues"]["nfl"]["staking_weight"] = 0.9
    p = tmp_path / "w.json"
    p.write_text(json.dumps(art))
    with pytest.raises(ValueError, match="lower bound"):
        MW.staking_weight("nfl", p)


def test_the_committed_weights_reproduce_from_source():
    """The artifact is regenerated here and compared, so a weight nobody can
    rebuild cannot be what sizes a bet."""
    from model import grade_market_weight as G
    art = json.loads(MW.WEIGHTS_PATH.read_text())
    assert set(art["leagues"]) == set(G.LEAGUES)
    for name, build in G.LEAGUES.items():
        rows, _ = build()
        fresh = G.grade(rows)
        for k in ("n", "w_hat", "se", "staking_weight"):
            assert fresh[k] == pytest.approx(art["leagues"][name][k], abs=1e-6), (name, k)


def test_football_has_no_measured_edge_over_the_close():
    """The finding this module exists for, pinned so a change to it is a
    deliberate event. If a refit moves either league off zero, this fails and
    the ADR gets revisited -- it does not quietly start staking."""
    for league in ("nfl", "cfb"):
        w, grade = MW.staking_weight(league)
        assert grade is not None and w == 0.0, league
        assert grade["contamination"].startswith("UPPER BOUND")
