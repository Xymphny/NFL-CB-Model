"""MLB, where the obvious distribution turned out to be wrong.

The tests that matter most here are the ones pinning WHY it is not Poisson,
because "baseball is Poisson" is the kind of assumption that gets restored by
a well-meaning simplification.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core.distributions import (  # noqa: E402
    BivariatePoissonDistribution, NegativeBinomialScoreDistribution,
)
from coverline.leagues.mlb import model as mlb  # noqa: E402
from tests.core.test_conformance import assert_distribution_conforms  # noqa: E402

RESULTS = ROOT / "model" / "mlb_dispersion_results.json"
ASOF = "2026-09-21T00:00:00Z"


class Stub:
    def __init__(self, **kw):
        self.f = mlb.GameFeatures(**kw)

    def features(self, game_id, asof):
        return self.f


def _model(exp_home=4.48, exp_away=4.43):
    return mlb.MLBModel(Stub(exp_home=exp_home, exp_away=exp_away))


# ------------------------------------------------------ the measurement ----

def test_the_dispersion_was_measured_not_assumed():
    art = json.loads(RESULTS.read_text())
    assert art["_provenance"]["n_games"] > 10_000
    for side in ("home", "away"):
        assert art["sides"][side]["variance_over_mean"] > 2.0, (
            "runs are no longer overdispersed; if the data changed, the "
            "distribution family should be re-read, not the constant nudged"
        )


def test_baseball_is_not_poisson_and_the_file_says_so():
    art = json.loads(RESULTS.read_text())
    assert "NOT Poisson" in art["verdict"]
    assert mlb.R_HOME == pytest.approx(3.8880, abs=1e-4)
    assert mlb.R_AWAY == pytest.approx(3.2426, abs=1e-4)


def test_the_two_sides_are_independent_and_that_is_why_no_shared_component():
    """The shared-component Poisson was the natural fix and is wrong: it buys
    total variance by inventing correlation that measures at +0.0006."""
    ind = json.loads(RESULTS.read_text())["independence"]
    assert abs(ind["correlation"]) < 0.01
    assert ind["independent"] is True
    assert abs(ind["total_variance"] - ind["sum_of_side_variances"]) < 0.5


def test_a_poisson_would_understate_run_variance_by_more_than_half():
    """The cost of the wrong family, as a number."""
    nb = NegativeBinomialScoreDistribution(4.48, 4.43, mlb.R_HOME, mlb.R_AWAY)
    po = BivariatePoissonDistribution(lam_home=4.48, lam_away=4.43)
    assert nb.total_sd() > 1.4 * po.total_sd()
    assert po.total_sd() ** 2 < 0.55 * nb.total_sd() ** 2


# ------------------------------------------------------------- the model ----

def test_it_satisfies_the_league_contract():
    m = _model()
    assert m.league == "mlb"
    assert "runline" in m.primary_markets


def test_the_moneyline_is_withheld_for_a_measured_reason():
    """Withheld on 2026-09-21, and the reason is a number.

    The model's conditioned P(home) averages 0.5063 over 12,148 games against
    an actual home win rate of 0.5315: a 2.5 point understatement, systematic,
    on every game. Two league rules cause it -- extra innings resolve every
    game, and the home team stops batting when it leads -- and conditioning
    the tie out redistributes that mass proportionally when the walk-off rule
    gives it overwhelmingly to the home side.

    The RUNLINE is not withheld, which is the surprising half: the fictitious
    tie mass and the missing one-run wins sit on the same side of 1.5, so the
    errors cancel where that market is priced.
    """
    art = json.loads((ROOT / "model" / "mlb_rules_structure.json").read_text())
    ml = art["market_impact"]["moneyline"]
    assert abs(ml["error"]) > 0.015, (
        "the moneyline error has shrunk below a point and a half; if that "
        "holds up the market can come back, but it comes back with a grade"
    )
    assert "moneyline" not in _model().primary_markets

    rl = art["market_impact"]["runline"]
    assert abs(rl["error"]) < 0.01
    assert "runline" in _model().primary_markets


def test_the_distribution_conforms_to_the_shared_battery():
    assert_distribution_conforms(_model().predict("g", ASOF), "mlb")


def test_the_distribution_reproduces_the_measured_dispersion():
    d = _model().predict("g", ASOF)
    ind = json.loads(RESULTS.read_text())["independence"]
    assert d.margin_sd() ** 2 == pytest.approx(ind["sum_of_side_variances"], rel=0.02)


def test_absurd_expected_runs_are_refused_not_priced():
    """A walk-forward row outside sane bounds is an upstream bug, and pricing
    it would produce confident nonsense."""
    with pytest.raises(ValueError, match="upstream bug"):
        _model(exp_home=0.2).predict("g", ASOF)
    with pytest.raises(ValueError, match="upstream bug"):
        _model(exp_away=40.0).predict("g", ASOF)


def test_a_non_positive_dispersion_is_refused():
    with pytest.raises(ValueError, match="not an overdispersed"):
        NegativeBinomialScoreDistribution(4.5, 4.5, r_home=0.0, r_away=3.0)


def test_totals_are_offered_and_marked_untested():
    """Unlike football, where totals are withheld on a GRADED failure. Here
    nothing has been graded either way, which is a weaker position -- so the
    market is offered and the absence of evidence is recorded rather than
    dressed up as either confidence or caution."""
    assert "total" in _model().primary_markets


# ---------------------------------------------------------------------------
# The rule, measured from linescores rather than inferred from fingerprints.
# ---------------------------------------------------------------------------

def _rules():
    return json.loads((ROOT / "model" / "mlb_rules_structure.json").read_text())


def test_the_home_team_never_bats_while_leading():
    """The rule is deterministic, and that is what makes it a rule.

    Across 12,146 games, ZERO had the home team leading after the top of the
    ninth and still batting. The converse has 74 exceptions -- shortened and
    called games -- which is a real 0.6% and is stated rather than modelled.
    """
    lt = _rules()["linescore_transition"]
    assert lt["measured"], "run model/ingest/mlb_linescores.py"
    det = lt["rule_is_deterministic"]
    assert det["home_led_after_top_nine_and_still_batted"] == 0
    assert det["home_did_not_lead_and_did_not_bat"] < 0.02 * lt["n_games"]


def test_a_lead_after_the_top_of_the_ninth_ends_the_game():
    """Rows the layer does not have to fit, because the league fixes them.

    Every state at +1 or better maps to that exact final margin with
    probability one. A layer that produced anything else there would be
    modelling a game that cannot happen.
    """
    table = _rules()["linescore_transition"]["transition_table"]
    for state in ("1", "2", "3", "4"):
        row = table[state]["final_margin"]
        assert row == {state: 1.0}, (
            f"state {state} no longer ends the game deterministically: {row}"
        )


def test_batting_last_is_worth_thirteen_points_when_tied():
    """The number the shipped model cannot express.

    Tied after the top of the ninth, the home team wins 62.8% of the time --
    on 1,149 games -- because it bats last. A symmetric distribution says
    50%, and conditioning a tie out proportionally keeps saying 50%. That gap
    is where the moneyline's 2.5 point bias comes from.
    """
    w = _rules()["linescore_transition"]["walk_off_advantage"]
    assert w["n"] > 800
    assert w["p_home_wins_when_tied_after_top_nine"] > 0.58, (
        "the walk-off advantage has shrunk; if it approaches 50% the "
        "symmetric model stops being wrong and the moneyline can return"
    )


def test_the_rates_are_fitted_to_truncated_runs_and_that_is_recorded():
    """The trap in front of whoever builds the layer.

    exp_home is fitted to OBSERVED home runs, 4.4766, which are truncated in
    45% of games. Untruncated nine-inning home scoring is about 4.68.
    Applying the rule on top of the observed rates would count the truncation
    twice, and the resulting model would look better calibrated than it is.
    """
    u = _rules()["linescore_transition"]["untruncated_scoring"]
    assert u["implied_untruncated_home_nine_innings"] > u["home_runs_observed"]
    assert u["why"]
