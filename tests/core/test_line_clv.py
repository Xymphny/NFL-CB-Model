"""CLV when the line moved: valued along the league's margin distribution."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core import pricing as P  # noqa: E402
from coverline.execution import grade as G  # noqa: E402
from coverline.execution.ledger import BetLedger, Close, Signal  # noqa: E402
from coverline.execution.line_clv import fair_at_bet_line  # noqa: E402

F = fair_at_bet_line


def test_an_unmoved_line_or_a_lineless_league_is_left_to_the_price():
    assert F(league="nfl", side="home", fair_close=0.5, close_line=-3.0, bet_line=-3.0) is None
    assert F(league="mlb", side="home", fair_close=0.5, close_line=-1.5, bet_line=-2.5) is None
    assert F(league="nfl", side="home", fair_close=0.5, close_line=None, bet_line=-3.0) is None


def test_the_half_point_off_three_is_worth_what_the_key_number_says():
    """+3.5 against a +3 close turns every push on 3 into a win. With NFL's
    measured P(margin = 3) that is worth about four points; a plain normal
    would say about one and a half."""
    got = F(league="nfl", side="away", fair_close=0.5, close_line=3.0, bet_line=3.5)
    assert 0.535 < got < 0.56
    off_key = F(league="nfl", side="away", fair_close=0.5, close_line=5.0, bet_line=5.5)
    assert got - 0.5 > 2 * (off_key - 0.5)


def test_a_cfb_pickem_half_point_is_worth_nothing_because_ties_cannot_happen():
    got = F(league="cfb", side="home", fair_close=0.52, close_line=-0.5, bet_line=0.5)
    assert got == pytest.approx(0.52, abs=1e-6)


def test_moving_there_and_back_returns_the_close():
    there = F(league="cfb", side="home", fair_close=0.47, close_line=-7.0, bet_line=-6.5)
    back = F(league="cfb", side="home", fair_close=there, close_line=-6.5, bet_line=-7.0)
    assert back == pytest.approx(0.47, abs=1e-6)
    assert there > 0.47                              # the better number


def test_nba_is_continuous_and_moves_smoothly():
    a = F(league="nba", side="home", fair_close=0.5, close_line=-4.5, bet_line=-3.5)
    b = F(league="nba", side="home", fair_close=0.5, close_line=-4.5, bet_line=-2.5)
    assert 0.5 < a < b < 0.6


def test_the_ledger_summary_no_longer_reports_a_beaten_line_as_the_vig(tmp_path):
    """The failure: a paper trade at -110 that beat the close by a point read
    -2.4 probability points -- exactly the vig -- whatever the line did."""
    led = BetLedger(tmp_path / "ledger")
    price = P.american_to_decimal(-110)
    led.record(Signal(signal_id="s", at="2026-09-22T14:00:00Z", league="nfl", event_id="e",
                      market="spreads", selection="A", line=3.5, p_model=0.56, p_market=0.5,
                      p_used=0.56, shrinkage=0.0, edge_claimed=0.07, edge_used=0.07,
                      disposition="not_placed", not_placed_reason="below_threshold", book="pinnacle",
                      price_decimal=price, side="away", commence_time="2026-09-27T17:00:00Z"))
    led.record_close(Close(signal_id="s", at="2026-09-27T16:55:00Z", close_decimals=[price, price],
                           outcome_index=1, close_line=3.0, market_has_sharp_close=True,
                           devig_method="power"))
    plain = led.clv("s").prob_points
    assert plain == pytest.approx(0.5 - 1 / price)           # the vig, and nothing else
    summ = G.clv_summary(led, paper=True)
    assert summ["line_adjusted"] == 1
    assert summ["mean_clv_probability_points"] > 0 > plain


def test_an_unmoved_market_is_minus_the_vig_after_it_and_zero_before_it(tmp_path):
    """The first real CLV read -2.9 and -1.7 points, which looks like losing
    and is mostly the hold. The vig-free market move separates the two."""
    led = BetLedger(tmp_path / "ledger")
    price = P.american_to_decimal(-110)
    common = dict(league="nfl", market="spreads", p_model=0.56, p_used=0.56, shrinkage=0.0,
                  edge_claimed=0.07, edge_used=0.07, disposition="not_placed",
                  not_placed_reason="below_threshold", book="pinnacle", price_decimal=price,
                  side="away", commence_time="2026-09-27T17:00:00Z")
    led.record(Signal(signal_id="flat", at="2026-09-22T14:00:00Z", event_id="e1",
                      selection="A", line=3.5, p_market=0.5, **common))
    led.record(Signal(signal_id="moved", at="2026-09-22T14:00:00Z", event_id="e2",
                      selection="A", line=3.5, p_market=0.5, **common))
    for sid, line in (("flat", 3.5), ("moved", 3.0)):
        led.record_close(Close(signal_id=sid, at="2026-09-27T16:55:00Z",
                               close_decimals=[price, price], outcome_index=1,
                               close_line=line, market_has_sharp_close=True,
                               devig_method="power"))
    closes = {c.signal_id: c for c in led.closes()}
    sigs = {s.signal_id: s for s in led.signals()}
    pts, _, _, move = G.line_aware_clv(sigs["flat"], closes["flat"], led.clv_of(sigs["flat"], closes["flat"]))
    assert move == pytest.approx(0.0) and pts == pytest.approx(0.5 - 1 / price)
    _, _, adjusted, move = G.line_aware_clv(sigs["moved"], closes["moved"],
                                            led.clv_of(sigs["moved"], closes["moved"]))
    assert adjusted and move > 0.035                  # the hook off 3, vig-free
    assert G.clv_summary(led, paper=True)["mean_market_move_points"] > 0
