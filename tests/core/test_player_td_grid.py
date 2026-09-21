"""The grid that was lost, reconstructed -- and the confound that nearly ruined it.

WHAT THIS PINS
migration/ledger.yaml's frozen-threshold-grid row records shipped constants in
model/player_projection.py citing a grid search whose output and specification
are both gone. model/calibrate_player_td_lambda.py sweeps them on training
seasons alone and commits the surface, which is a reconstruction rather than a
recovery: the original artifact is still lost and the row stays open.

THE CONFOUND IS THE REASON THESE ASSERTIONS EXIST
Eligibility for the anytime_td sample is max(lambda, base) >= TD_MIN_LAMBDA,
and every constant in the sweep moves lambda. So a free-floating sample scores
each configuration on a different population, and admitting more low-lambda
players LOWERS log-loss for free because they are easy negatives.

The first run of the sweep did exactly that and concluded that eight of ten
shipped constants were not the training argmin -- with TIER_K = {12, 8, 4}
apparently beating the shipped value by 0.011 of log-loss while admitting
1,124 more player-weeks. Evaluated on a FIXED population that same
configuration is WORSE than shipped. The conclusion reversed completely.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

GRID = ROOT / "model" / "player_td_grid.json"


@pytest.fixture(scope="module")
def grid() -> dict:
    assert GRID.exists(), (
        f"{GRID.relative_to(ROOT)} is missing -- run "
        "model/calibrate_player_td_lambda.py"
    )
    return json.loads(GRID.read_text())


def test_every_configuration_is_scored_on_one_fixed_population(grid) -> None:
    """The correction that flipped the answer.

    Without it the sweep rewards permissiveness rather than accuracy.
    """
    assert grid["fixed_evaluation_set"]["n"] > 20_000
    assert grid["shipped_configuration"]["coverage"] == 1.0
    for name, s in grid["sweeps"].items():
        assert s["evaluated_on_fixed_population"], name


def test_no_held_out_metric_was_computed(grid) -> None:
    """There is nothing to leak because the sealed numbers do not exist.

    2024-2025 are spent on the shipped configuration's grade in
    model/player_projection_results.json, and model/ratings.py is the worked
    example of a test column consulted while a knob was being turned.
    """
    prov = grid["_provenance"]
    assert prov["no_holdout_metric_computed"]
    assert prov["selects_nothing"]
    blob = json.dumps(grid).lower()
    for sealed in ("test_log_loss", "holdout_log_loss", "held_out_log_loss"):
        assert sealed not in blob, f"a sealed metric appears in the grid: {sealed}"


def test_the_shipped_constants_hold_up(grid) -> None:
    """Seven of ten sit on the training argmin.

    The lost grid was run by someone who picked defensible values, which is
    the outcome worth recording: the wound was that nobody could CHECK, not
    that the numbers were wrong.
    """
    sweeps = grid["sweeps"]
    on_argmin = [n for n, s in sweeps.items() if s["shipped_is_train_argmin"]]
    assert len(on_argmin) >= 7, (
        f"only {len(on_argmin)} of {len(sweeps)} shipped constants are the "
        "training argmin; if that has fallen, something moved"
    )


def test_the_constants_that_miss_the_argmin_are_the_flat_ones(grid) -> None:
    """The other three, and why nothing was re-selected.

    ENV_DAMP, LONG_CRED and ENV_CLAMP are not at their training argmin and
    have the three smallest spreads in the sweep -- ENV_CLAMP moves log-loss
    by 0.000022 across its whole range. A knob that does nothing is not a
    knob worth turning, and turning it would be a new selection the next
    holdout has to pay for.
    """
    sweeps = grid["sweeps"]
    off = {n: s["spread_of_train_log_loss"]
           for n, s in sweeps.items() if not s["shipped_is_train_argmin"]}
    on = {n: s["spread_of_train_log_loss"]
          for n, s in sweeps.items() if s["shipped_is_train_argmin"]}
    assert off, "every constant is now the argmin, which would be suspicious"
    assert max(off.values()) < 0.001, (
        f"a constant that misses its argmin now has a spread of "
        f"{max(off.values())} -- it has stopped being flat and the decision "
        "not to move it needs re-reading"
    )
    assert max(on.values()) > max(off.values()), (
        "the load-bearing constants are no longer the ones sitting on their "
        "argmin"
    )


def test_coverage_is_reported_where_a_value_cannot_cover_the_set(grid) -> None:
    """A configuration that cannot price a shipped-eligible player-week says so.

    TD_MIN_LAMBDA above the shipped 0.15 covers 83% and then 56% of the fixed
    set, which is why its log-loss column must be read after its coverage
    column and not instead of it.
    """
    s = grid["sweeps"]["TD_MIN_LAMBDA"]
    assert s["values_with_incomplete_coverage"], (
        "raising the eligibility floor no longer drops anything, which "
        "cannot be right"
    )
    assert s["note"]
