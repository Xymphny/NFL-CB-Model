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
