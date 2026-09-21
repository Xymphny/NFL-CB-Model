"""The goalie-pull and overtime layers, and the trap they open.

WHAT THESE PIN
The layers are graded (t = 39.99 on 2022-2023), and the grade is not the
interesting part -- 82% of it is the overtime rule, which is a fact anyone can
look up and which the shipped model was simply ignoring by assigning about a
sixth of its probability to a tied final score the league does not permit. The
goalie-pull layer is the modelled part and clears separately at t = 6.75.

What needs pinning is the SHAPE the layers produce and the DOUBLE-COUNTING
TRAP they open. The rates in data/nhl_fitted.json are fitted to final scores,
so they already contain every empty-net goal the pull layer then adds. Feeding
them into the composition is the one mistake this design makes easy, it
produces numbers that look entirely reasonable, and nothing but an explicit
refusal catches it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core.interfaces import ScoreDistribution  # noqa: E402
from coverline.leagues.nhl.model import (  # noqa: E402
    GameFeatures, NHLModel, NotFitted, load_rules,
)
from coverline.leagues.nhl.rules import (  # noqa: E402
    GoaliePullLayer, NHLFinalScoreDistribution, OvertimeLayer,
)

ARTIFACT = ROOT / "data" / "nhl_rules.json"


@pytest.fixture(scope="module")
def layers():
    return load_rules()


@pytest.fixture(scope="module")
def dist(layers):
    pull, overtime = layers
    return NHLFinalScoreDistribution(2.95, 2.65, pull, overtime)


class _Source:
    def __init__(self, **kw):
        self._kw = kw

    def features(self, game_id: str, asof: str) -> GameFeatures:
        return GameFeatures(**self._kw)


def test_a_tie_is_impossible_not_merely_unlikely(dist) -> None:
    """The league does not permit it, so the model must not price it.

    This is the cheapest check that the overtime layer is wired in at all, and
    it must be exactly zero rather than small: a three-way market priced off a
    small draw probability is a different bet from one priced off none.
    """
    assert dist.margin_pmf(0.0) == 0.0
    assert dist.ties() == 0.0


def test_the_margin_distribution_is_non_monotone(dist) -> None:
    """More three-goal games than two-goal games, as the league actually is.

    This is the property no independent Poisson produces at any rates, and it
    sits across exactly the 1.5 line the puck line is priced on. If it ever
    goes away, the layer has stopped doing the one thing it was built for.
    """
    two = dist.margin_pmf(2) + dist.margin_pmf(-2)
    three = dist.margin_pmf(3) + dist.margin_pmf(-3)
    assert three > two, f"P(|m|=3)={three:.4f} is not above P(|m|=2)={two:.4f}"


def test_the_composed_mass_is_a_distribution(dist) -> None:
    j = dist.joint()
    assert np.isclose(j.sum(), 1.0)
    assert (j >= 0).all()
    margins = sum(dist.margin_pmf(m) for m in range(-25, 26))
    totals = sum(dist.total_pmf(t) for t in range(0, 51))
    assert np.isclose(margins, 1.0, atol=1e-9)
    assert np.isclose(totals, 1.0, atol=1e-9)


def test_sampling_agrees_with_the_analytic_pmf(dist) -> None:
    """A distribution that cannot simulate itself cannot be checked."""
    rng = np.random.default_rng(20260921)
    draws = dist.sample(60_000, rng)
    margin = draws[:, 0] - draws[:, 1]
    assert (margin != 0).all(), "a sampled tie escaped the overtime layer"
    for m in (-2, -1, 1, 2, 3):
        assert (margin == m).mean() == pytest.approx(
            dist.margin_pmf(m), abs=0.01
        )
    assert draws.sum(axis=1).mean() == pytest.approx(dist.total_mean(), abs=0.05)


def test_composing_with_final_score_rates_is_refused(layers) -> None:
    """The one mistake this design makes easy.

    Final-score rates already contain the empty-net goals the pull layer adds.
    Composing with them double-counts, and the result looks entirely plausible
    -- slightly more scoring, slightly wider margins -- which is why this has
    to be a refusal rather than a comment.
    """
    model = NHLModel(_Source(lam_home=3.23, lam_away=3.00), rules=layers)
    with pytest.raises(ValueError, match="NO-PULL"):
        model.predict("game", "2026-01-01T00:00:00Z")


def test_the_puck_line_appears_only_with_the_layers(layers) -> None:
    """Withheld without them, offered with them. Not a flag anyone can set."""
    without = NHLModel(_Source(lam_home=3.2, lam_away=3.0))
    assert "puck_line" not in without.primary_markets

    with_layers = NHLModel(
        _Source(lam_home=3.2, lam_away=3.0,
                lam_home_nopull=2.95, lam_away_nopull=2.65),
        rules=layers,
    )
    assert "puck_line" in with_layers.primary_markets
    assert with_layers.models_empty_net is True


def test_the_composed_distribution_satisfies_the_seam(dist) -> None:
    """Everything downstream imports only core. This must fit through it."""
    for name in ("is_discrete", "margin_mean", "margin_sd", "margin_cdf",
                 "margin_pmf", "total_mean", "total_cdf", "total_pmf",
                 "sample"):
        assert hasattr(dist, name), f"missing {name}"
    assert dist.is_discrete is True
    # cdf and pmf must agree, which is where an off-by-one in a discrete
    # distribution shows up.
    for x in (-3, -1, 0, 1, 2, 4):
        below = sum(dist.margin_pmf(m) for m in range(-30, x + 1))
        assert dist.margin_cdf(x) == pytest.approx(below, abs=1e-9)


def test_home_advantage_does_not_survive_the_tie_break() -> None:
    """The finding worth carrying, asserted where it can be seen.

    Home teams win about 55% of games decided in regulation and about 50% of
    those decided after it. Carrying the regulation edge through the tie-break
    overprices every home moneyline by roughly the tie probability times the
    difference.
    """
    art = json.loads(ARTIFACT.read_text())
    p = art["overtime_home_win_prob"]
    assert 0.47 < p < 0.53, (
        f"overtime home win probability is {p}, which is no longer close to a "
        "coin flip -- if this is real it changes every moneyline price and "
        "deserves its own record"
    )


def test_the_grade_is_decomposed_and_both_parts_clear() -> None:
    """A t of 40 is a reason for suspicion, not celebration.

    Most of it is knowing the rules of hockey. The decomposition is asserted
    so that nobody reads the headline number as the layer's own achievement.
    """
    art = json.loads(ARTIFACT.read_text())
    d = art["decomposition"]
    assert d["overtime_rule_alone"]["t"] > 2
    assert d["goalie_pull_layer_over_and_above"]["t"] > 2, (
        "the goalie-pull layer no longer clears on its own; the headline "
        "grade is then the overtime rule and should be reported as that"
    )
    assert d["share_of_gain_that_is_the_overtime_rule"] > 0.5, (
        "if the pull layer is now the majority of the gain that is a genuine "
        "improvement and this test should be re-read, not deleted"
    )


def test_a_pull_table_that_does_not_sum_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="sums to"):
        GoaliePullLayer(table={0: {(0, 0): 0.9}})
    with pytest.raises(ValueError, match="no measured table"):
        GoaliePullLayer(table={})


def test_an_overtime_layer_outside_zero_and_one_is_refused() -> None:
    with pytest.raises(ValueError):
        OvertimeLayer(home_win_prob=0.0)
    with pytest.raises(ValueError):
        OvertimeLayer(home_win_prob=1.5)


def test_the_loader_refuses_an_artifact_that_did_not_clear(tmp_path) -> None:
    art = json.loads(ARTIFACT.read_text())
    art["holdout_grade"]["supported"] = False
    art["holdout_grade"]["t"] = 0.4
    bad = tmp_path / "nhl_rules.json"
    bad.write_text(json.dumps(art))
    with pytest.raises(NotFitted, match="did not clear"):
        load_rules(bad)
    with pytest.raises(NotFitted, match="no fitted"):
        load_rules(tmp_path / "absent.json")


def test_the_puck_line_prices_end_to_end(layers) -> None:
    """The market enabled today, run through the actual recommender.

    Adding "puck_line" to primary_markets changed nothing operationally until
    core/markets.py existed, because nothing in production read that list and
    "puck_line" is not the vendor's word for it. This is the first test that
    takes the league's own market name, finds the quotes, prices both sides
    and checks the numbers are sane.

    The push assertion is the one that matters. A puck line sits at 1.5, so
    there is no push -- but the same code path prices a moneyline as a spread
    of zero, where the tie IS voiding, and getting that wrong was a 5.9 point
    error in MLB.
    """
    from coverline.core.markets import vendor_key
    from coverline.execution import recommend as Rc
    from coverline.execution.normalize import Quote

    model = NHLModel(
        _Source(lam_home=3.2, lam_away=3.0,
                lam_home_nopull=2.95, lam_away_nopull=2.65),
        rules=layers,
    )
    dist = model.predict("e1", "2026-01-01T00:00:00Z")

    def _q(outcome, price, point):
        return Quote(event_id="e1", sport="icehockey_nhl",
                     commence_time="2026-01-02T00:00:00Z", home_team="H",
                     away_team="A", bookmaker="pinnacle",
                     market=vendor_key("puck_line"), outcome=outcome,
                     price_decimal=price, point=point,
                     last_update=None, captured_at="2026-01-01T00:00:00Z")

    quotes = [_q("H", 2.35, -1.5), _q("A", 1.63, 1.5)]
    signals = Rc.recommend(
        dist=dist, quotes=quotes, event_id="e1", market="puck_line",
        league="nhl", bankroll=10_000.0, shrinkage=0.92,
        primary_markets=model.primary_markets,
    )

    assert len(signals) == 2, "both sides of the puck line should be priced"
    for s in signals:
        assert 0.0 < s.p_model < 1.0
        assert s.push_probability == 0.0, (
            "a 1.5 line cannot push; a non-zero push here means the line was "
            "rounded to an integer somewhere"
        )
    assert sum(s.p_model for s in signals) == pytest.approx(1.0, abs=1e-9), (
        "the two sides of a no-push market must be complementary"
    )
