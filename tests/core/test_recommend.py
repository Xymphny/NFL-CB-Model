"""The integration layer, with the sign convention pinned first.

A spread's sign is the easiest thing in this project to get backwards, and
getting it backwards produces numbers that look entirely reasonable while
recommending the wrong side of every game. So the convention is tested before
anything else, in both directions, against a distribution whose answer is
known by construction.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.core.distributions import NormalMarginDistribution  # noqa: E402
from coverline.execution import recommend as Rc  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402
from coverline.execution.normalize import Quote  # noqa: E402

WEIGHTS = {int(k): float(v) for k, v in
           json.loads((ROOT / "data" / "nfl_key_numbers.json").read_text())["weights"].items()}


def _dist(mu=-3.0, keyed=True):
    return NormalMarginDistribution(
        mu_margin=mu, sd_margin=13.3, mu_total=44.0, sd_total=10.0,
        key_number_weights=WEIGHTS if keyed else None)


def _q(outcome, price, point, book="pinnacle", market="spreads", event="e1"):
    return Quote(event_id=event, sport="nfl", commence_time="", home_team="H",
                 away_team="A", bookmaker=book, market=market, outcome=outcome,
                 price_decimal=price, point=point, last_update=None,
                 captured_at="2026-09-21T18:00:00Z")


def _market(home_price=1.9091, away_price=1.9091, point=-3.0, book="pinnacle"):
    return [_q("H", home_price, point, book), _q("A", away_price, -point, book)]


# ------------------------------------------------------- sign convention ----

def test_a_home_favourite_covers_more_often_than_a_home_underdog():
    """The most basic sanity check, and the one that catches a flipped sign."""
    fav = _dist(mu=+7.0)      # home expected to win by 7
    dog = _dist(mu=-7.0)      # home expected to lose by 7
    p_fav, _ = Rc.cover_probability(fav, -3.0)
    p_dog, _ = Rc.cover_probability(dog, -3.0)
    assert p_fav > 0.5 > p_dog, (
        f"home favourite covers {p_fav:.3f}, underdog {p_dog:.3f} -- the "
        "margin or line sign is inverted"
    )


def test_a_bigger_line_is_harder_to_cover():
    d = _dist(mu=+7.0)
    easy, _ = Rc.cover_probability(d, -1.5)
    hard, _ = Rc.cover_probability(d, -10.5)
    assert easy > hard


def test_the_two_sides_of_a_market_are_complementary():
    """P(home covers -L) + P(away covers +L) must be 1 once pushes are
    conditioned out, or the two sides are being priced inconsistently.

    Away covers when the home margin falls BELOW the threshold, so its
    unconditioned probability is cdf(t) - pmf(t): the push is excluded from
    both sides, not handed to one of them. The first version of this test
    subtracted the home side from 1 and got the push sign wrong, which is the
    same slip the conditioning exists to prevent.
    """
    d = _dist(mu=+2.0)
    t = 3.0
    home, push = Rc.cover_probability(d, -t)
    away = (d.margin_cdf(t) - d.margin_pmf(t)) / (1.0 - push)
    assert push > 0, "pick a threshold with mass on it or this proves nothing"
    assert home + away == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------- pushes ----

def test_the_push_is_conditioned_out_not_ignored():
    """On a key number the push is worth 7-8%. Comparing raw P(cover) with a
    devigged two-way price prices a two-way bet with three-way numbers."""
    d = _dist(mu=+3.0)
    conditioned, push = Rc.cover_probability(d, -3.0)
    raw = 1.0 - d.margin_cdf(3.0)
    # centred ON the key number, so the atom is larger than the 0.078 the
    # distribution tests pin at mu = -3
    assert push == pytest.approx(0.0871, abs=0.002)
    assert conditioned > raw
    assert conditioned == pytest.approx(raw / (1 - push), abs=1e-12)


def test_a_half_point_line_has_no_push():
    _, push = Rc.cover_probability(_dist(), -3.5)
    assert push == 0.0


def test_an_integer_line_is_withheld_without_measured_key_numbers():
    """A plain rounded normal understates P(margin=3) by ~3x, so the edge
    would be wrong in the flattering direction."""
    with pytest.raises(Rc.PushPriceWithheld, match="flattering direction"):
        Rc.price_candidate(dist=_dist(keyed=False), quotes=_market(point=-3.0),
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=0.5)


def test_a_half_point_line_is_priced_even_without_key_numbers():
    """Nothing to get wrong when no mass sits on the line."""
    c = Rc.price_candidate(dist=_dist(keyed=False), quotes=_market(point=-3.5),
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=0.5)
    assert c.push_probability == 0.0


# ------------------------------------------------------------ pricing ----

def test_a_one_sided_market_is_refused():
    with pytest.raises(ValueError, match="devigging one side"):
        Rc.price_candidate(dist=_dist(), quotes=[_q("H", 1.91, -3.0)],
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=0.5)


def test_an_unknown_selection_is_refused():
    with pytest.raises(ValueError, match="not an outcome"):
        Rc.price_candidate(dist=_dist(), quotes=_market(), event_id="e1",
                           market="spreads", bookmaker="pinnacle",
                           selection="NOPE", bankroll=100_000, shrinkage=0.5)


def test_a_balanced_market_devigs_to_a_coin_flip():
    c = Rc.price_candidate(dist=_dist(mu=+3.0), quotes=_market(point=-3.0),
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=0.5)
    assert c.p_market == pytest.approx(0.5, abs=1e-9)


# ----------------------------------------------------------- signals ----

def test_a_declined_candidate_still_becomes_a_ledger_row():
    """The whole reason the ledger exists. Dropping it makes execution cost
    unmeasurable."""
    c = Rc.price_candidate(dist=_dist(mu=-20.0), quotes=_market(point=-3.0),
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=0.5)
    s = Rc.to_signal(c, league="nfl", bankroll=100_000)
    assert s.placed is False
    assert s.not_placed_reason == "below_threshold"
    assert s.p_model > 0 and s.edge_claimed < 0


def test_a_placed_signal_carries_everything_clv_will_need():
    c = Rc.price_candidate(dist=_dist(mu=+14.0), quotes=_market(point=-3.0),
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=1.0)
    s = Rc.to_signal(c, league="nfl", bankroll=100_000)
    assert s.placed is True
    assert s.book and s.price_decimal and s.stake and s.line is not None


def test_the_recorded_shrinkage_reproduces_the_probability_actually_bet():
    c = Rc.price_candidate(dist=_dist(mu=+14.0), quotes=_market(point=-3.0),
                           event_id="e1", market="spreads", bookmaker="pinnacle",
                           selection="H", bankroll=100_000, shrinkage=0.6)
    s = Rc.to_signal(c, league="nfl", bankroll=100_000)
    rebuilt = s.shrinkage * s.p_model + (1 - s.shrinkage) * s.p_market
    assert rebuilt == pytest.approx(s.p_used, abs=1e-9)


# ------------------------------------------------------------ slate ----

def test_every_book_and_side_is_priced_and_recorded(tmp_path):
    quotes = (_market(1.9091, 1.9091, -3.0, "pinnacle")
              + _market(1.9524, 1.8696, -3.0, "draftkings"))
    led = BetLedger(tmp_path)
    sigs = Rc.recommend(dist=_dist(mu=+10.0), quotes=quotes, event_id="e1",
                        market="spreads", league="nfl", bankroll=100_000,
                        shrinkage=0.6, ledger=led)
    assert len(sigs) == 4                       # 2 books x 2 sides
    assert len(led.signals()) == 4
    assert {s.book or "unplaced" for s in sigs} >= {"pinnacle", "draftkings"} | {"unplaced"} - {"unplaced"}


def test_the_best_available_price_is_surveyed_across_books(tmp_path):
    quotes = (_market(1.9091, 1.9091, -3.0, "pinnacle")
              + _market(1.9524, 1.8696, -3.0, "draftkings"))
    sigs = Rc.recommend(dist=_dist(mu=+14.0), quotes=quotes, event_id="e1",
                        market="spreads", league="nfl", bankroll=100_000,
                        shrinkage=1.0)
    placed_home = [s for s in sigs if s.placed and s.selection == "H"]
    assert placed_home, "no home side was recommended to test shopping on"
    for s in placed_home:
        assert s.books_surveyed == 2
        assert s.best_available_decimal == pytest.approx(1.9524)
    pin = next(s for s in placed_home if s.book == "pinnacle")
    assert pin.price_slippage_cents > 0, "taking the worse price shows no slippage"


def test_withheld_candidates_produce_no_signal_at_all(tmp_path):
    """Withheld is not a recommendation of any kind, including a negative one."""
    sigs = Rc.recommend(dist=_dist(keyed=False), quotes=_market(point=-3.0),
                        event_id="e1", market="spreads", league="nfl",
                        bankroll=100_000, shrinkage=0.5)
    assert sigs == []


def test_recommending_writes_nothing_unless_given_a_ledger(tmp_path):
    """A slate can be inspected before anything is committed to an
    append-only record."""
    led = BetLedger(tmp_path)
    Rc.recommend(dist=_dist(mu=+10.0), quotes=_market(point=-3.0), event_id="e1",
                 market="spreads", league="nfl", bankroll=100_000, shrinkage=0.6)
    assert led.signals() == []


# ------------------------------------------------- moneylines and ties ----

def test_a_positive_expected_margin_implies_a_favourite():
    """The invariant that catches an unconditioned voiding outcome.

    This is the test that would have caught the moneyline bug on the day it
    was written, and did not exist. A distribution with a positive expected
    margin whose win probability reads below 0.5 has mass somewhere it should
    not be counting.
    """
    from coverline.core.distributions import NegativeBinomialScoreDistribution
    d = NegativeBinomialScoreDistribution(mu_home=4.6, mu_away=4.2,
                                          r_home=3.9, r_away=3.2)
    assert d.margin_mean() > 0
    p, tie = Rc.cover_probability(d, 0.0)
    assert tie > 0.05, "pick a distribution with real tie mass or this proves nothing"
    assert p > 0.5, (
        f"expected margin is {d.margin_mean():+.3f} but P(win) reads {p:.4f}; "
        "a voiding outcome is being counted as a loss"
    )


def test_the_moneyline_conditions_the_tie_out_like_a_push():
    """A moneyline is a spread of zero. The tie voids the bet exactly as a
    push does, and conditioning it out is the same operation."""
    from coverline.core.distributions import NegativeBinomialScoreDistribution
    d = NegativeBinomialScoreDistribution(4.6, 4.2, 3.9, 3.2)
    conditioned, tie = Rc.cover_probability(d, 0.0)
    raw = 1.0 - d.margin_cdf(0.0)
    assert conditioned == pytest.approx(raw / (1 - tie), abs=1e-12)
    assert conditioned - raw > 0.03, "the correction should be material here"


def test_the_two_moneyline_sides_are_complementary():
    from coverline.core.distributions import NegativeBinomialScoreDistribution
    d = NegativeBinomialScoreDistribution(4.6, 4.2, 3.9, 3.2)
    home, tie = Rc.cover_probability(d, 0.0)
    away = (d.margin_cdf(0.0) - d.margin_pmf(0.0)) / (1 - tie)
    assert home + away == pytest.approx(1.0, abs=1e-12)


def test_football_moneylines_condition_out_their_rarer_tie_too():
    """Ties are rare in football, not impossible, and a rare voiding outcome
    is still a voiding outcome."""
    d = _dist(mu=+0.5)
    conditioned, tie = Rc.cover_probability(d, 0.0)
    assert 0.0 < tie < 0.05
    assert conditioned > 1.0 - d.margin_cdf(0.0)


# ---------------------------------------------------------------------------
# The two pricing bugs found on 2026-09-21, pinned as properties.
#
# Both were invisible to every test here because every test priced ONE side
# and checked it against a number computed the same way the code computes it.
# A property that relates the two sides catches what agreement with yourself
# cannot.
# ---------------------------------------------------------------------------


def _pair(dist, point, market="spreads", outcomes=("H", "A")):
    quotes = [_q(outcomes[0], 1.91, point, market=market),
              _q(outcomes[1], 1.91, None if point is None else -point,
                 market=market)]
    if market == "totals":
        quotes = [_q("Over", 1.91, point, market=market),
                  _q("Under", 1.91, point, market=market)]
    return [Rc.price_candidate(dist=dist, quotes=quotes, event_id="e1",
                               market=market, bookmaker="pinnacle",
                               selection=q.outcome, bankroll=100_000,
                               shrinkage=0.9)
            for q in quotes]


@pytest.mark.parametrize("point", [-6.5, -3.5, 0.5, 2.5, 7.5])
def test_the_two_sides_of_a_handicap_are_complementary(point):
    """THE BUG: the away side was priced with the home team's answer.

    cover_probability always answers for the HOME team at the line it is
    given, and price_candidate used to hand it whichever line was on the
    quote being priced. For a home favourite at -6.5 the away side came back
    0.5314 against a truth of 0.8214, and the two sides summed to 1.0849.

    The error flips sign with the line, so it was not a constant bias that
    might have shown up as a losing record -- it inflated the away side on
    some games and deflated it on others, and every away-side handicap price
    in every league was wrong.
    """
    cands = _pair(_dist(mu=-6.0), point)
    assert sum(c.p_model for c in cands) == pytest.approx(1.0, abs=1e-9)


def test_the_two_sides_of_a_moneyline_are_complementary_after_the_tie():
    """Same fix, and the tie must still be conditioned out of both sides."""
    cands = _pair(_dist(mu=-6.0), None, market="h2h")
    assert sum(c.p_model for c in cands) == pytest.approx(1.0, abs=1e-9)
    assert cands[0].push_probability == cands[1].push_probability > 0


def test_a_total_is_priced_off_the_total_not_the_margin():
    """THE OTHER BUG, and the worse one.

    Every market went through cover_probability, which reads margin_cdf. An
    Over 44.5 was therefore priced as P(margin > -44.5) and came back 0.9982
    against a truth of 0.4801. MLB is the only league that offers totals
    today, and it would have bet every Over at maximum stake.
    """
    d = _dist(mu=-6.0)
    truth = 1.0 - d.total_cdf(44.5)
    over, under = _pair(d, 44.5, market="totals")
    assert over.p_model == pytest.approx(truth, abs=1e-9)
    assert over.p_model + under.p_model == pytest.approx(1.0, abs=1e-9)
    assert over.p_model < 0.9, (
        "an Over priced above 0.9 on a line near the mean is the margin "
        "distribution answering a question about the total"
    )


def test_an_unknown_outcome_name_is_refused_rather_than_assumed():
    """The side is read off the quote's own team names.

    Guessing it -- by position, or by assuming the first side is home -- is
    how the away side was wrong before, and a renamed team would reintroduce
    it silently.
    """
    quotes = [_q("Home Team", 1.91, -3.5), _q("A", 1.91, 3.5)]
    with pytest.raises(ValueError, match="neither"):
        Rc.price_candidate(dist=_dist(mu=-6.0), quotes=quotes, event_id="e1",
                           market="spreads", bookmaker="pinnacle",
                           selection="Home Team", bankroll=100_000,
                           shrinkage=0.9)


def test_a_count_distribution_may_price_an_integer_line():
    """A pmf that IS the atom needs no key-number table.

    The push guard refused every integer total in baseball and hockey,
    because it treated an exact count distribution as if it were a rounded
    normal. That withheld markets that price correctly -- the conservative
    direction, and still wrong.
    """
    from coverline.core.distributions import NegativeBinomialScoreDistribution

    d = NegativeBinomialScoreDistribution(4.6, 4.2, 3.888, 3.2426)
    over, under = _pair(d, 9.0, market="totals")
    assert over.push_probability == pytest.approx(d.total_pmf(9), abs=1e-9)
    assert over.push_probability > 0.05
    assert over.p_model + under.p_model == pytest.approx(1.0, abs=1e-9)
