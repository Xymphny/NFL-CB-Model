"""Grading: the loop from a recorded bet back to what the market closed at.

The tests that matter are the refusals. Grading a bet against a post-kickoff
snapshot, or inventing a close for one nobody captured, both produce a CLV
series that looks complete and is biased -- and a biased CLV series is worse
than a short one, because the whole reason CLV was adopted is that it needs
few observations to mean something.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.execution import grade as G  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger, Signal  # noqa: E402

KICKOFF = "2026-09-21T20:00:00Z"


#: Exact, not rounded. An earlier version used 1.9091 and the CLV assertion
#: below -- which assumes a true -110 -- missed by 4.5e-6. The fixture was
#: wrong, not the tolerance.
EVEN_JUICE = P.american_to_decimal(-110)


def _payload(home_price=EVEN_JUICE, away_price=EVEN_JUICE, point=-3.0, event="e1"):
    return [{"id": event, "sport_key": "nfl", "home_team": "H", "away_team": "A",
             "commence_time": KICKOFF,
             "bookmakers": [{"key": "pinnacle", "markets": [
                 {"key": "spreads", "outcomes": [
                     {"name": "H", "price": home_price, "point": point},
                     {"name": "A", "price": away_price, "point": -point}]}]}]}]


def _signal(sid="s1", market="spreads", selection="H", price=EVEN_JUICE):
    return Signal(signal_id=sid, at="2026-09-21T12:00:00Z", league="nfl",
                  event_id="e1", market=market, selection=selection, line=-3.0,
                  p_model=0.56, p_market=0.52, p_used=0.54, shrinkage=0.5,
                  edge_claimed=0.07, edge_used=0.03, disposition="placed",
                  book="pinnacle", price_decimal=price, stake=500.0,
                  bankroll_at_placement=100_000.0)


@pytest.fixture
def setup(tmp_path):
    led = BetLedger(tmp_path / "ledger")
    store = BronzeStore(tmp_path / "bronze")
    return led, store


# ------------------------------------------------------- choosing a close ----

def test_the_close_is_the_last_snapshot_before_kickoff(setup):
    led, store = setup
    for at, price in (("2026-09-21T14:00:00Z", 1.85),
                      ("2026-09-21T19:55:00Z", 1.95),
                      ("2026-09-21T20:30:00Z", 2.50)):   # after kickoff
        store.write_snapshot(sport="nfl", captured_at=at,
                             payload=_payload(home_price=price), cost=3,
                             source_url="u")
    cands = G.snapshots_for(store, "nfl")
    close = G.close_for(cands, event_id="e1", commence_time=KICKOFF)
    assert close is not None
    assert close.captured_at == "2026-09-21T19:55:00Z"


def test_a_post_kickoff_snapshot_is_never_used(setup):
    """It carries an in-play price. Using it would move every CLV figure in
    whichever direction the game went -- a bias that looks like skill."""
    led, store = setup
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T21:00:00Z",
                         payload=_payload(home_price=3.0), cost=3, source_url="u")
    cands = G.snapshots_for(store, "nfl")
    assert G.close_for(cands, event_id="e1", commence_time=KICKOFF) is None


def test_an_event_absent_from_every_snapshot_has_no_close(setup):
    led, store = setup
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:00:00Z",
                         payload=_payload(event="other"), cost=3, source_url="u")
    cands = G.snapshots_for(store, "nfl")
    assert G.close_for(cands, event_id="e1", commence_time=KICKOFF) is None


# ------------------------------------------------------------- grading ----

def test_a_placed_bet_gets_a_close_and_a_clv(setup):
    led, store = setup
    led.record(_signal())
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:55:00Z",
                         payload=_payload(), cost=3, source_url="u")
    rep = G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    assert rep.graded == ["s1"]
    clv = led.clv("s1")
    assert clv is not None and clv.valid is True
    assert clv.ev_pct == pytest.approx(-0.0454545, abs=1e-6)


def test_an_ungradeable_bet_is_reported_not_invented(setup):
    """An ungraded bet and a break-even one are different facts."""
    led, store = setup
    led.record(_signal())
    rep = G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    assert rep.graded == []
    assert rep.reasons() == {"no_pregame_snapshot": 1}
    assert led.clv("s1") is None


def test_unplaced_signals_are_not_graded(setup):
    led, store = setup
    led.record(Signal(signal_id="s9", at="x", league="nfl", event_id="e1",
                      market="spreads", selection="H", line=-3.0,
                      p_model=0.55, p_market=0.53, p_used=0.54, shrinkage=0.5,
                      edge_claimed=0.02, edge_used=0.01,
                      disposition="not_placed", not_placed_reason="line_moved"))
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:55:00Z",
                         payload=_payload(), cost=3, source_url="u")
    rep = G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    assert rep.graded == [] and rep.ungraded == []


def test_grading_twice_does_not_record_a_second_close(setup):
    """Two closes for one signal would be two versions of what the market
    did, and picking one silently is how a record stops being a record."""
    led, store = setup
    led.record(_signal())
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:55:00Z",
                         payload=_payload(), cost=3, source_url="u")
    G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    rep = G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    assert rep.graded == [] and rep.already_had == ["s1"]
    assert len(led.closes()) == 1


def test_a_market_missing_from_the_snapshot_is_named(setup):
    led, store = setup
    led.record(_signal(market="totals"))
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:55:00Z",
                         payload=_payload(), cost=3, source_url="u")
    rep = G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    assert rep.reasons() == {"market_not_in_snapshot": 1}


# ---------------------------------------------------------- prop closes ----

def test_prop_markets_are_marked_as_having_no_sharp_close():
    assert G.has_sharp_close("spreads") is True
    assert G.has_sharp_close("h2h") is True
    assert G.has_sharp_close("player_points") is False
    assert G.has_sharp_close("batter_props") is False


def test_prop_clv_is_counted_separately_not_averaged_in(setup):
    """Averaging a prop close in with real ones produces a number that is
    neither."""
    led, store = setup
    led.record(_signal(sid="real", market="spreads"))
    led.record(_signal(sid="prop", market="player_points"))
    payload = _payload()
    payload[0]["bookmakers"][0]["markets"].append(
        {"key": "player_points", "outcomes": [
            {"name": "H", "price": 1.87, "point": 22.5},
            {"name": "A", "price": 1.95, "point": 22.5}]})
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:55:00Z",
                         payload=payload, cost=3, source_url="u")
    G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})

    s = G.clv_summary(led)
    assert s["graded_valid"] == 1
    assert s["graded_invalid_no_sharp_close"] == 1
    assert s["mean_clv_devigged"] is not None


# ------------------------------------------------------------- summary ----

def test_the_summary_reports_the_naive_overstatement(setup):
    """The size of the mistake a CLV number carries when nobody says which
    close it was computed against."""
    led, store = setup
    led.record(_signal())
    store.write_snapshot(sport="nfl", captured_at="2026-09-21T19:55:00Z",
                         payload=_payload(), cost=3, source_url="u")
    G.grade(led, store, sport="nfl", commence_times={"e1": KICKOFF})
    s = G.clv_summary(led)
    # clv_summary rounds to 5 places for readability, so the tolerance
    # matches that rather than demanding a precision the function does not
    # claim to return.
    assert s["mean_overstatement_if_naive"] == pytest.approx(
        P.hold([EVEN_JUICE, EVEN_JUICE]), abs=1e-5)
    assert "which one" in s["note"]


def test_an_empty_ledger_summarises_honestly(setup):
    led, _ = setup
    s = G.clv_summary(led)
    assert s["graded_valid"] == 0 and s["mean_clv_devigged"] is None
