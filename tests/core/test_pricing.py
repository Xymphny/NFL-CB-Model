"""Devig, EV and CLV, checked against defining equations rather than against
quoted figures.

This file exists in this shape because of what happened while it was written.
The research that specified this module quoted four devig results for a
-300/+240 market: 71.84 multiplicative, 72.47 power, 72.80 additive, 73.05
shin. Two of those four are wrong, and the only reason that is known is that
these tests check the defining equation instead of the quoted number:

  * POWER is defined by sum(p_i^k) = 1. At the solved k = 1.0793 the
    implementation gives 73.31, and sum(p^k) = 1.0 to twelve decimals. No k
    exists that produces 72.47 while satisfying the constraint.
  * SHIN is defined by a single z satisfying pi_i^2/S = (1-z)p_i^2 + z*p_i for
    EVERY outcome. The implementation recovers one consistent z to machine
    precision across 2-, 3- and 4-way markets. It gives 72.79, not 73.05.

Had these tests asserted the quoted numbers, the code would have been bent to
match a secondhand figure. Annotate, do not adjust.

The same exercise resolved a source conflict the research flagged and could not
settle -- whether Shin reduces to the additive method for two-outcome markets.
It does, exactly, to 1e-16, across every two-way market tested. It does NOT for
three or more outcomes. The source asserting the equivalence was right; the
worked example contradicting it was wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coverline.core import pricing as P  # noqa: E402

FAV_DOG = [P.american_to_decimal(-300), P.american_to_decimal(240)]
BALANCED = [P.american_to_decimal(-110), P.american_to_decimal(-110)]
LOPSIDED = [P.american_to_decimal(-800), P.american_to_decimal(550)]
THREE_WAY = [2.10, 3.40, 3.90]


# ----------------------------------------------------------- conversion ----

def test_american_decimal_roundtrip():
    for a in (-300, -110, -101, 100, 240, 550):
        assert P.decimal_to_american(P.american_to_decimal(a)) == pytest.approx(a)


def test_known_conversions():
    assert P.american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-6)
    assert P.american_to_decimal(240) == pytest.approx(3.40)
    assert P.american_to_implied(-110) == pytest.approx(0.5238095, abs=1e-6)


def test_hold_is_not_the_overround():
    """A -110/-110 market is 4.76% overround but holds 4.55% of handle.
    Conflating them overstates the cost of vig by about 5% of itself."""
    assert P.booksum(BALANCED) == pytest.approx(1.0476190, abs=1e-6)
    assert P.hold(BALANCED) == pytest.approx(0.0454545, abs=1e-6)


def test_conversion_rejects_impossible_prices():
    with pytest.raises(ValueError):
        P.american_to_decimal(0)
    with pytest.raises(ValueError):
        P.decimal_to_implied(1.0)


# ---------------------------------------------- defining-equation checks ----

def test_every_method_normalises():
    for market in (FAV_DOG, BALANCED, LOPSIDED, THREE_WAY):
        for m in P.METHODS:
            try:
                p = P.devig(market, m)
            except ValueError:
                continue
            assert p.sum() == pytest.approx(1.0, abs=1e-10), (m, market)
            assert np.all(p > 0.0), (m, market)


def test_power_satisfies_its_defining_equation():
    """sum(raw ** k) == 1 at the solved k. This is what 'power method' MEANS."""
    raw = np.array([P.decimal_to_implied(d) for d in FAV_DOG])
    p = P.devig(FAV_DOG, "power")
    k = float(np.log(p[0]) / np.log(raw[0]))
    assert float(np.sum(raw**k)) == pytest.approx(1.0, abs=1e-12)
    assert k == pytest.approx(1.0793, abs=1e-4)
    assert p[0] == pytest.approx(0.733084, abs=1e-6)


def test_shin_recovers_one_consistent_z():
    """Shin's inverse is only correct if a SINGLE z satisfies the forward
    relation pi^2/S = (1-z)p^2 + z*p for every outcome simultaneously.
    Per-outcome z values that disagree mean the inverse is wrong."""
    for market in (FAV_DOG, THREE_WAY, [1.55, 4.20, 6.00, 9.50]):
        raw = np.array([P.decimal_to_implied(d) for d in market])
        s = raw.sum()
        p = P.devig(market, "shin")
        zs = [(raw[i] ** 2 / s - p[i] ** 2) / (p[i] - p[i] ** 2) for i in range(len(p))]
        assert max(zs) - min(zs) < 1e-12, (market, zs)
        assert 0.0 < zs[0] < 1.0, zs[0]


def test_shin_equals_additive_for_two_outcomes_only():
    """Resolves a conflict the research could not settle. The equivalence is
    real and exact for n=2, and false for n>2."""
    for market in (FAV_DOG, BALANCED, LOPSIDED):
        shin = P.devig(market, "shin")
        additive = P.devig(market, "additive")
        assert np.max(np.abs(shin - additive)) < 1e-14, market

    shin3 = P.devig(THREE_WAY, "shin")
    add3 = P.devig(THREE_WAY, "additive")
    assert np.max(np.abs(shin3 - add3)) > 1e-5, "shin collapsed to additive on a 3-way"


def test_balanced_market_is_a_coin_flip_under_every_method():
    for m in P.METHODS:
        assert P.devig(BALANCED, m)[0] == pytest.approx(0.5, abs=1e-12), m


def test_method_ordering_on_a_favourite():
    """Power keeps the most on the favourite because it shrinks the longshot
    hardest; multiplicative keeps the least. If this order ever flips, a devig
    implementation has changed behaviour."""
    fav = {m: P.devig(FAV_DOG, m)[0] for m in P.METHODS}
    assert fav["multiplicative"] < fav["additive"] <= fav["shin"] < fav["power"]


# ------------------------------------------------------------ guardrail ----

def test_devig_all_reports_the_disagreement():
    cmp = P.devig_all(FAV_DOG, index=0)
    assert set(cmp.fair_by_method) == set(P.METHODS)
    assert cmp.spread == pytest.approx(0.0147, abs=1e-3)
    # The guardrail's whole purpose: 1.5 points of method disagreement swamps
    # a 1% claimed edge and does not swamp a 5% one.
    assert cmp.exceeds(0.01) is True
    assert cmp.exceeds(0.05) is False


#: A three-way with a heavy favourite and a 50-1 longshot. The excess split
#: three ways exceeds the longshot's own implied probability, so additive would
#: hand back a negative number. Found by construction, not guessed: a two-way
#: market has to be far more extreme than -5000/+1200 to break additive, and
#: that one does NOT break it.
ADDITIVE_BREAKS = [1.0 / 0.75, 1.0 / 0.30, 1.0 / 0.02]


def test_devig_all_omits_methods_that_cannot_run():
    """A market too lopsided for additive must drop it, not substitute
    something and quietly narrow the reported spread."""
    cmp = P.devig_all(ADDITIVE_BREAKS, index=2)
    assert "additive" not in cmp.fair_by_method
    assert "power" in cmp.fair_by_method and "shin" in cmp.fair_by_method


def test_additive_refuses_rather_than_returning_a_negative_probability():
    with pytest.raises(ValueError, match="too lopsided"):
        P.devig_additive(ADDITIVE_BREAKS)


def test_devig_rejects_a_one_sided_market():
    with pytest.raises(ValueError, match="at least two"):
        P.devig([1.91])


def test_unknown_method_names_itself():
    with pytest.raises(ValueError, match="unknown devig method"):
        P.devig(FAV_DOG, "bayesian")  # type: ignore[arg-type]


# ------------------------------------------------------------ ev and clv ----

def test_expected_value_is_zero_at_the_fair_price():
    assert P.expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert P.expected_value(0.52381, 1.9090909) == pytest.approx(0.0, abs=1e-5)


def test_expected_value_matches_the_clv_ratio_form():
    """EV and CLV must share an axis, or they cannot be plotted against
    realised ROI together."""
    fair, price = 0.55, 1.9090909
    p_bet = P.decimal_to_implied(price)
    assert P.expected_value(fair, price) == pytest.approx((fair - p_bet) / p_bet)


def test_clv_devigged_is_far_below_naive():
    """The mistake this module exists to prevent.

    Betting -110 into a -110/-110 close scores EXACTLY ZERO against the posted
    close -- by construction, since implied(-110) * decimal(-110) = 1 -- while
    its true value against the devigged fair price is -4.55%, the full hold.
    So the naive figure does not merely overstate the edge here; it reports a
    break-even bet that is in fact losing at the house rate. Every bet in a
    log scored the naive way carries this same shift."""
    clv = P.closing_line_value(
        bet_decimal=P.american_to_decimal(-110),
        close_decimals=BALANCED,
        outcome_index=0,
    )
    assert clv.ev_pct == pytest.approx(-0.0454545, abs=1e-6)
    assert clv.naive_ev_pct == pytest.approx(0.0, abs=1e-9)
    assert clv.devig_overstatement == pytest.approx(0.0454545, abs=1e-6)
    assert clv.devig_overstatement == pytest.approx(P.hold(BALANCED), abs=1e-9)


def test_beating_the_close_on_price_is_not_the_same_as_having_an_edge():
    """This reproduces, in our own arithmetic, the finding that positive CLV
    deciles can still lose money.

    Betting -105 into a -110/-110 close beats the posted price by 5 cents and
    beats the number by half a point -- unambiguously good execution -- and is
    STILL -2.38% EV, because the devigged fair price is even money and -105 is
    worse than even money. Good execution inside a market you have no edge in
    loses more slowly; it does not win. Any dashboard that reports price CLV
    without EV will show this bet as a success."""
    clv = P.closing_line_value(
        bet_decimal=P.american_to_decimal(-105),
        close_decimals=BALANCED,
        outcome_index=0,
        bet_line=3.5,
        close_line=3.0,
    )
    assert clv.price_cents == pytest.approx(5.0)
    assert clv.line_points == pytest.approx(0.5)
    assert clv.ev_pct == pytest.approx(-0.0238095, abs=1e-6)
    assert clv.ev_pct > -0.0454545  # better than betting -110, still negative


def test_clv_line_points_are_none_without_a_line():
    clv = P.closing_line_value(bet_decimal=2.0, close_decimals=BALANCED)
    assert clv.line_points is None


def test_props_clv_is_marked_invalid_not_zero():
    """Prop closes are not made by market-making books, so CLV against them is
    an informed guess. Marking it invalid is not the same as recording zero,
    and averaging code must be able to tell the difference."""
    clv = P.closing_line_value(
        bet_decimal=P.american_to_decimal(-115),
        close_decimals=[P.american_to_decimal(-120), P.american_to_decimal(100)],
        market_has_sharp_close=False,
    )
    assert clv.valid is False
    assert clv.ev_pct != 0.0


def test_clv_uses_the_full_market_not_one_side():
    """Passing a single price is the commonest way to get CLV wrong; the
    signature makes it impossible."""
    with pytest.raises(ValueError, match="at least two"):
        P.closing_line_value(bet_decimal=1.91, close_decimals=[1.91])
