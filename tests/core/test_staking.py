"""Staking, with the growth-rate claims verified numerically.

Every quantitative claim the design rests on is re-derived here from the growth
rate g(f) = p*ln(1+fb) + (1-p)*ln(1-f) rather than asserted from a remembered
formula. The retention figures (75% at half Kelly, 43.7% at quarter) and the
overbetting result (growth ~0 at twice Kelly) come out of that function, so if
someone later changes the sizing code these tests fail on the economics, not on
a magic number.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coverline.core.staking import (  # noqa: E402
    growth_rate,
    kelly_fraction,
    shrink_probability,
    shrinkage_weight,
    size_bet,
)

P, DEC = 0.55, 2.0  # even money, 55% -> full Kelly is exactly 10% of bankroll


# ------------------------------------------------------------- shrinkage ----

def test_shrinkage_weight_reproduces_the_published_table():
    assert shrinkage_weight([1.3] * 40) == pytest.approx(0.4083, abs=1e-4)
    assert shrinkage_weight([1.6] * 40) == pytest.approx(0.6094, abs=1e-4)


def test_shrinkage_weight_is_zero_when_the_pipeline_carries_no_signal():
    """RMS t of 1.0 is what pure noise produces. The arithmetic must say so
    rather than returning a small positive weight that ships it anyway."""
    assert shrinkage_weight([1.0] * 40) == 0.0
    assert shrinkage_weight([0.4, -0.7, 0.9, -0.2]) == 0.0


def test_shrinkage_weight_refuses_an_empty_log():
    with pytest.raises(ValueError, match="every candidate tested"):
        shrinkage_weight([])


def test_shrinkage_weight_uses_squares_so_sign_does_not_matter():
    """A candidate that came back strongly negative is as informative about
    the pipeline's noise level as one that came back strongly positive."""
    assert shrinkage_weight([2.0, -2.0]) == pytest.approx(shrinkage_weight([2.0, 2.0]))


def test_selecting_only_winners_inflates_the_weight():
    """The failure mode the docstring warns about, made concrete: dropping the
    losing attempts roughly doubles the weight and ships noise at full size."""
    honest = [2.1, 0.3, -0.5, 1.1, 0.2, -0.9, 1.8]
    cherry_picked = [t for t in honest if t > 1.0]
    assert shrinkage_weight(cherry_picked) > shrinkage_weight(honest) + 0.25


def test_shrink_probability_interpolates():
    assert shrink_probability(0.60, 0.50, 1.0) == pytest.approx(0.60)
    assert shrink_probability(0.60, 0.50, 0.0) == pytest.approx(0.50)
    assert shrink_probability(0.60, 0.50, 0.4) == pytest.approx(0.54)


def test_shrink_probability_rejects_bad_input():
    with pytest.raises(ValueError):
        shrink_probability(0.6, 0.5, 1.5)
    with pytest.raises(ValueError):
        shrink_probability(1.0, 0.5, 0.5)


# ----------------------------------------------------------------- kelly ----

def test_full_kelly_is_the_growth_maximum():
    """Verify f* maximises g by checking it beats its neighbours, rather than
    trusting the closed form."""
    f_star = kelly_fraction(P, DEC)
    assert f_star == pytest.approx(0.10)
    best = growth_rate(f_star, P, DEC)
    for delta in (-0.02, -0.01, 0.01, 0.02):
        assert growth_rate(f_star + delta, P, DEC) < best


def test_half_kelly_retains_about_three_quarters_of_growth():
    f_star = kelly_fraction(P, DEC)
    ratio = growth_rate(f_star / 2, P, DEC) / growth_rate(f_star, P, DEC)
    assert ratio == pytest.approx(0.75, abs=0.01)


def test_quarter_kelly_retains_about_forty_four_percent():
    f_star = kelly_fraction(P, DEC)
    ratio = growth_rate(f_star / 4, P, DEC) / growth_rate(f_star, P, DEC)
    assert ratio == pytest.approx(0.437, abs=0.01)


def test_double_kelly_destroys_the_entire_edge():
    """The asymmetry the quarter-Kelly default exists to survive. Every bet is
    still +EV at 2f*; the bankroll still does not grow."""
    f_star = kelly_fraction(P, DEC)
    g_star = growth_rate(f_star, P, DEC)
    g_double = growth_rate(2 * f_star, P, DEC)
    assert abs(g_double) < 0.05 * g_star
    assert g_double < 0.0
    # And underbetting by the same factor costs only a quarter of growth.
    assert growth_rate(f_star / 2, P, DEC) > 0.7 * g_star


def test_kelly_is_negative_when_the_price_is_bad():
    assert kelly_fraction(0.45, DEC) < 0.0


def test_kelly_rejects_impossible_input():
    with pytest.raises(ValueError):
        kelly_fraction(0.0, DEC)
    with pytest.raises(ValueError):
        kelly_fraction(0.55, 1.0)


# -------------------------------------------------------------- sizing ----

def test_the_two_orderings_agree_only_at_even_money():
    """Shrink-then-Kelly and Kelly-then-scale cross at decimal 2.0. Pinned
    because the crossing point is what makes the ordering safe at the prices
    this project actually bets."""
    for dec, direction in ((1.8, "scale"), (2.0, "equal"), (2.5, "shrink")):
        shrink_first = kelly_fraction(shrink_probability(0.60, 0.50, 0.40), dec)
        scale_after = kelly_fraction(0.60, dec) * 0.40
        if direction == "equal":
            assert shrink_first == pytest.approx(scale_after), dec
        elif direction == "scale":
            assert scale_after > shrink_first, dec
        else:
            assert shrink_first > scale_after, dec


def test_shrinking_first_can_veto_a_bet_that_stake_scaling_would_place():
    """The safety case, at a price close to the -110 this project lives on.

    At decimal 1.8 a 60% model against a 50% market, shrunk at weight 0.40,
    has NO positive-Kelly stake at all -- the shrunk probability does not clear
    the price. Scaling the stake instead would place 4% of bankroll on it.
    This is the concrete reason stage 1 precedes stage 2."""
    plan = size_bet(
        p_model=0.60, p_market=0.50, decimal_price=1.8,
        bankroll=100_000, shrinkage=0.40, kelly_multiple=1.0,
        max_bankroll_fraction=1.0,
    )
    assert plan.placed is False
    assert plan.stake == 0.0
    assert kelly_fraction(0.60, 1.8) * 0.40 > 0.0  # the ordering that would bet


def test_plan_records_both_the_claimed_and_the_used_edge():
    plan = size_bet(
        p_model=0.56, p_market=0.52, decimal_price=1.9091,
        bankroll=100_000, shrinkage=0.41,
    )
    assert plan.edge_claimed > plan.edge_used > 0.0
    assert plan.p_used == pytest.approx(shrink_probability(0.56, 0.52, 0.41))
    assert plan.placed is True
    assert plan.kelly_multiple == pytest.approx(0.25)


def test_zero_shrinkage_collapses_to_the_market_and_refuses_to_bet():
    """b = 0 means the attempt log says the pipeline has no signal. The engine
    must then decline every bet rather than betting the market price."""
    plan = size_bet(
        p_model=0.60, p_market=0.52, decimal_price=1.9091,
        bankroll=100_000, shrinkage=0.0,
    )
    assert plan.placed is False
    assert plan.stake == 0.0
    assert "no edge" in plan.reason


def test_bankroll_cap_binds_and_says_so():
    plan = size_bet(
        p_model=0.80, p_market=0.55, decimal_price=DEC,
        bankroll=100_000, shrinkage=1.0, kelly_multiple=0.25,
        max_bankroll_fraction=0.02,
    )
    assert plan.bankroll_fraction == pytest.approx(0.02)
    assert "capped" in plan.reason
    assert plan.kelly_multiple < 0.25


def test_minimum_edge_floor_declines_marginal_bets():
    plan = size_bet(
        p_model=0.530, p_market=0.525, decimal_price=1.9091,
        bankroll=100_000, shrinkage=0.40, min_edge=0.02,
    )
    assert plan.placed is False
    assert "below floor" in plan.reason


def test_sizing_rejects_impossible_configuration():
    with pytest.raises(ValueError):
        size_bet(p_model=0.55, p_market=0.52, decimal_price=DEC,
                 bankroll=0.0, shrinkage=0.5)
    with pytest.raises(ValueError):
        size_bet(p_model=0.55, p_market=0.52, decimal_price=DEC,
                 bankroll=1000, shrinkage=0.5, kelly_multiple=1.5)


def test_no_default_shrinkage_is_importable():
    """The withholding pillar, enforced. There is no attempt log yet, so there
    is no defensible default weight, so the module must not offer one."""
    import coverline.core.staking as s
    assert not hasattr(s, "DEFAULT_SHRINKAGE")
    with pytest.raises(TypeError):
        size_bet(p_model=0.55, p_market=0.52, decimal_price=DEC, bankroll=1000)  # type: ignore[call-arg]
