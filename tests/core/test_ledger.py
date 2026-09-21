"""The bet ledger, with the refusals tested harder than the recording.

The easy half is writing rows. The half that makes the ledger worth having is
what it will not accept: a bet missing the fields CLV needs, a free-text
reason for a signal that did not convert, an edit to a row already written, a
close given as one side of a market.

Each of those, allowed once, makes some later question unanswerable -- and the
question usually only gets asked months after the row was written.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.execution.ledger import (  # noqa: E402
    BetLedger, Close, Settlement, Signal,
)


def _placed(**over):
    base = dict(signal_id="s1", at="2026-09-21T18:00:00Z", league="nfl",
                event_id="e1", market="spread", selection="KC", line=-3.0,
                p_model=0.56, p_market=0.52, p_used=0.5364, shrinkage=0.52,
                edge_claimed=0.069, edge_used=0.024, disposition="placed",
                book="pinnacle", price_decimal=1.9091, stake=661.13,
                bankroll_at_placement=100_000.0)
    base.update(over)
    return Signal(**base)


def _unplaced(**over):
    base = dict(signal_id="s2", at="2026-09-21T18:05:00Z", league="nfl",
                event_id="e2", market="spread", selection="BUF", line=-6.5,
                p_model=0.55, p_market=0.53, p_used=0.54, shrinkage=0.52,
                edge_claimed=0.05, edge_used=0.03, disposition="not_placed",
                not_placed_reason="line_moved")
    base.update(over)
    return Signal(**base)


@pytest.fixture
def ledger(tmp_path):
    return BetLedger(tmp_path)


# ------------------------------------------------------------ recording ----

def test_a_placed_bet_round_trips(ledger):
    ledger.record(_placed())
    back = ledger.signals()
    assert len(back) == 1 and back[0].placed is True
    assert back[0].stake == pytest.approx(661.13)


def test_an_unconverted_signal_is_recorded_too(ledger):
    """The design decision this module exists around: bets you did not get
    leave no trace unless something writes them down."""
    ledger.record(_unplaced())
    assert ledger.signals()[0].placed is False


def test_conversion_reports_why_signals_did_not_become_bets(ledger):
    ledger.record(_placed())
    ledger.record(_unplaced())
    ledger.record(_unplaced(signal_id="s3", not_placed_reason="limited"))
    c = ledger.conversion()
    assert c == {"signals": 3, "placed": 1, "rate": pytest.approx(1 / 3, abs=1e-4),
                 "reasons": {"line_moved": 1, "limited": 1}}


# ------------------------------------------------------------- refusals ----

def test_a_placed_bet_without_the_fields_clv_needs_is_refused(ledger):
    for missing in ("book", "price_decimal", "stake"):
        with pytest.raises(ValueError, match="never be graded"):
            ledger.record(_placed(**{missing: None}))


def test_a_free_text_reason_is_refused(ledger):
    """Free text is how the commonest reason becomes unqueryable."""
    with pytest.raises(ValueError, match="unqueryable"):
        ledger.record(_unplaced(not_placed_reason="was busy"))
    with pytest.raises(ValueError, match="must be one of"):
        ledger.record(_unplaced(not_placed_reason=None))


def test_an_unplaced_signal_cannot_carry_a_stake(ledger):
    with pytest.raises(ValueError, match="cannot carry a stake"):
        ledger.record(_unplaced(stake=100.0))


def test_the_ledger_refuses_to_rewrite_a_row(ledger):
    """Rewriting history in a betting ledger is how a losing week becomes a
    break-even one without anybody deciding to lie."""
    ledger.record(_placed())
    with pytest.raises(ValueError, match="append-only"):
        ledger.record(_placed(stake=1.0))


def test_impossible_probabilities_are_refused(ledger):
    with pytest.raises(ValueError, match="strictly between"):
        ledger.record(_placed(p_model=1.0))
    with pytest.raises(ValueError, match="shrinkage"):
        ledger.record(_placed(shrinkage=1.5))


def test_a_close_needs_the_whole_market(ledger):
    """Passing one side is the commonest way to get CLV wrong."""
    ledger.record(_placed())
    with pytest.raises(ValueError, match="cannot be removed"):
        ledger.record_close(Close(signal_id="s1", at="x", close_decimals=[1.91],
                                  outcome_index=0, close_line=-3.0))


def test_a_close_for_an_unknown_signal_is_refused(ledger):
    with pytest.raises(KeyError, match="no signal"):
        ledger.record_close(Close(signal_id="ghost", at="x",
                                  close_decimals=[1.91, 1.91],
                                  outcome_index=0, close_line=-3.0))


def test_only_placed_bets_settle(ledger):
    ledger.record(_unplaced())
    with pytest.raises(KeyError, match="not a PLACED signal"):
        ledger.record_settlement(Settlement("s2", "x", "win", 100.0))


def test_an_unknown_settlement_result_is_refused(ledger):
    ledger.record(_placed())
    with pytest.raises(ValueError, match="must be one of"):
        ledger.record_settlement(Settlement("s1", "x", "cashed_out", 10.0))


# ----------------------------------------------------------------- clv ----

def test_clv_is_computed_against_the_devigged_close(ledger):
    """Betting -110 into a -110/-110 close is break-even against the posted
    price and -4.55% against the fair one."""
    ledger.record(_placed(price_decimal=P.american_to_decimal(-110)))
    ledger.record_close(Close(signal_id="s1", at="x",
                              close_decimals=[P.american_to_decimal(-110)] * 2,
                              outcome_index=0, close_line=-3.0))
    clv = ledger.clv("s1")
    assert clv is not None
    assert clv.ev_pct == pytest.approx(-0.0454545, abs=1e-6)
    assert clv.naive_ev_pct == pytest.approx(0.0, abs=1e-9)


def test_clv_is_none_until_a_close_is_recorded(ledger):
    ledger.record(_placed())
    assert ledger.clv("s1") is None


def test_props_closes_carry_their_invalidity_through(ledger):
    """A prop close is not a sharp forecast, and averaging code must be able
    to tell that apart from a CLV of zero."""
    ledger.record(_placed(market="player_points"))
    ledger.record_close(Close(signal_id="s1", at="x",
                              close_decimals=[1.87, 1.95], outcome_index=0,
                              close_line=None, market_has_sharp_close=False))
    assert ledger.clv("s1").valid is False


# ---------------------------------------------------- execution cost ----

def test_slippage_measures_cents_given_up_against_the_best_price(ledger):
    s = _placed(price_decimal=P.american_to_decimal(-110),
                best_available_decimal=P.american_to_decimal(-105),
                books_surveyed=7)
    ledger.record(s)
    assert ledger.signals()[0].price_slippage_cents == pytest.approx(5.0, abs=0.01)
    assert ledger.execution_cost()["mean_cents_given_up"] == pytest.approx(5.0, abs=0.01)


def test_unsurveyed_bets_are_counted_as_unmeasured_not_as_zero(ledger):
    """Recording no shopping as zero slippage would flatter execution exactly
    where nothing was checked."""
    ledger.record(_placed())                       # no best_available recorded
    cost = ledger.execution_cost()
    assert cost == {"placed": 1, "measurable": 0, "unmeasured": 1,
                    "mean_cents_given_up": None}


def test_an_empty_ledger_answers_honestly(ledger):
    assert ledger.conversion()["rate"] is None
    assert ledger.execution_cost()["mean_cents_given_up"] is None
    assert ledger.signals() == []


def test_no_credentials_or_account_fields_exist():
    """This file sits in a git repo. Book names yes, account references never.

    Inspects the actual dataclass FIELDS rather than grepping the source. The
    first version scanned the text and failed on this module's own docstring,
    which says it stores no book logins -- a check that cannot tell a schema
    from a sentence about a schema is not checking the schema.
    """
    import dataclasses
    from coverline.execution import ledger as L

    banned = {"account_number", "username", "password", "api_key", "account_id",
              "login", "token", "secret", "credential"}
    for cls in (L.Signal, L.Close, L.Settlement):
        names = {f.name.lower() for f in dataclasses.fields(cls)}
        hits = {n for n in names if any(b in n for b in banned)}
        assert not hits, f"{cls.__name__} carries {sorted(hits)}"


def test_the_only_book_reference_is_a_name(ledger):
    """A book name identifies a market. An account reference identifies you."""
    import dataclasses
    from coverline.execution import ledger as L
    names = {f.name for f in dataclasses.fields(L.Signal)}
    assert "book" in names
    assert not any(n.startswith("book_") and n != "books_surveyed" for n in names)
