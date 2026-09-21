"""The ninth-inning layer, and the invariant that caught a subtle leak.

WHAT THIS PINS
ADR 0017 withheld the MLB moneyline for a systematic 2.5 point understatement
of the home team's win probability, caused by two league rules a symmetric
pair of count distributions cannot express. This layer is the repair, graded
once on 2024-2025.

THE GATE IS THE BIAS GATE, deliberately. The market was closed for a bias, so
the matched question is whether the residual bias is distinguishable from
zero. It is not: t = 0.35 against the baseline's 3.61. Binary log-loss on the
realised winner improves by only t = 1.79 and does NOT clear, which is also
asserted here -- removing a bias is enough to reopen a market closed for a
bias and not enough to claim an edge.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.leagues.mlb import model as mlb  # noqa: E402
from coverline.leagues.mlb.rules import (  # noqa: E402
    MLBFinalScoreDistribution, NinthInningLayer,
)

ART = ROOT / "data" / "mlb_rules.json"


class _Src:
    def features(self, game_id, asof):
        return mlb.GameFeatures(exp_home=4.6, exp_away=4.4)


@pytest.fixture(scope="module")
def rules():
    return mlb.load_rules()


@pytest.fixture(scope="module")
def dist(rules):
    layer, hs, as_ = rules
    return MLBFinalScoreDistribution(4.6, 4.4, mlb.R_HOME, mlb.R_AWAY, layer,
                                     home_eight_scale=hs, away_nine_scale=as_)


def test_a_tie_is_impossible_at_every_state(rules) -> None:
    """Extra innings resolve every game, so no state may finish level.

    THE SUBTLE ONE. Rows are measured per state and validated per state, but
    a deficit beyond the measured range borrows the edge row -- and a cell
    that took a four-run deficit to a one-run WIN takes a five-run deficit to
    a TIE. The first version leaked 0.0004 of mass onto an impossible
    outcome: small, and exactly the kind of small this layer exists to
    remove.
    """
    layer, _, _ = rules
    for state in range(-15, 16):
        row = layer.row(state)
        tied = sum(v for (i, j), v in row.items() if state + i - j == 0)
        assert tied == 0.0, f"state {state} finishes level with mass {tied}"
        assert np.isclose(sum(row.values()), 1.0, atol=1e-6)


def test_the_composed_distribution_never_ties(dist) -> None:
    assert dist.ties() == 0.0
    assert dist.margin_pmf(0.0) == 0.0


def test_a_lead_after_the_top_of_the_ninth_ends_the_game(rules) -> None:
    """Rows the league fixes, which the layer must not fit."""
    layer, _, _ = rules
    for state in (1, 2, 3, 4):
        assert layer.row(state) == {(0, 0): 1.0}


def test_the_layer_reproduces_the_one_run_asymmetry(dist) -> None:
    """The fingerprint that started this, back out of the composed model.

    Home wins by exactly one far more often than the away side does, because
    a walk-off ends play the instant the home team leads. Measured 0.1725
    against 0.1111; a symmetric model says they are equal.
    """
    plus, minus = dist.margin_pmf(1), dist.margin_pmf(-1)
    assert plus > minus * 1.3, (
        f"P(home by 1)={plus:.4f} is no longer well above P(away by 1)="
        f"{minus:.4f}; the walk-off asymmetry has gone out of the layer"
    )


def test_a_row_that_would_end_level_is_refused() -> None:
    """The validation, aimed at a table that breaks the rule."""
    with pytest.raises(ValueError, match="finish level"):
        NinthInningLayer(table={-1: {(1, 0): 1.0}})
    with pytest.raises(ValueError, match="deterministically"):
        NinthInningLayer(table={1: {(1, 0): 1.0}})


def test_the_scales_must_shrink_the_observed_rates() -> None:
    """The trap: exp_home is fitted to truncated runs.

    A scale of 1.0 means the truncation is about to be counted twice, and the
    resulting model looks BETTER calibrated than it is.
    """
    layer, hs, as_ = mlb.load_rules()
    assert 0.85 < hs < 0.97, f"home eight-inning scale {hs} is implausible"
    assert 0.9 < as_ <= 1.0
    with pytest.raises(ValueError, match="counted twice"):
        MLBFinalScoreDistribution(4.6, 4.4, mlb.R_HOME, mlb.R_AWAY, layer,
                                  home_eight_scale=1.0)


def test_the_moneyline_returns_only_with_the_layer(rules) -> None:
    without = mlb.MLBModel(_Src())
    assert "moneyline" not in without.primary_markets
    assert "runline" in without.primary_markets

    with_layer = mlb.MLBModel(_Src(), rules=rules)
    assert "moneyline" in with_layer.primary_markets


def test_the_bias_gate_is_what_reopened_the_market() -> None:
    """Both numbers, so the headline cannot be read alone."""
    art = json.loads(ART.read_text())
    gate = art["moneyline_bias_gate"]
    assert not gate["baseline"]["unbiased"]
    assert abs(gate["baseline"]["t"]) > 2
    assert gate["rules_model"]["unbiased"]
    assert abs(gate["rules_model"]["t"]) < 2

    assert not art["moneyline_grade"]["supported"], (
        "binary log-loss now clears too; that is a stronger claim than this "
        "market was reopened on and deserves its own record"
    )
    assert art["what_this_does_not_show"]


def test_most_of_the_headline_gain_is_just_removing_ties() -> None:
    """A t of 33 is a reason for suspicion, as it was in hockey."""
    art = json.loads(ART.read_text())
    d = art["decomposition"]
    assert d["share_that_is_just_removing_ties"] > 0.8
    assert d["measured_layer_over_and_above"]["t"] > 2, (
        "the measured layer no longer clears on its own; the headline is "
        "then the impossible-outcome fix and must be reported as that"
    )


def test_the_loader_refuses_a_layer_that_is_still_biased(tmp_path) -> None:
    art = json.loads(ART.read_text())
    art["moneyline_bias_gate"]["rules_model"]["unbiased"] = False
    art["moneyline_bias_gate"]["rules_model"]["t"] = 4.1
    bad = tmp_path / "mlb_rules.json"
    bad.write_text(json.dumps(art))
    with pytest.raises(mlb.NotFitted, match="bias gate"):
        mlb.load_rules(bad)
    with pytest.raises(mlb.NotFitted, match="no graded"):
        mlb.load_rules(tmp_path / "absent.json")
