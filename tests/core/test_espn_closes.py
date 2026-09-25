"""ESPN's public odds records -> closing lines for the market grade."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from model.ingest import espn_closes as E  # noqa: E402


def _am(v):
    return {"american": v}


def _item(name, home_ml="+105", away_ml="-125", spread="+1.5", hp="-105", ap="-115",
          close=True):
    side = lambda ml, pts, pr: {"moneyLine": 105, "open": {"moneyLine": _am(ml)},   # noqa: E731
                                **({"close": {"moneyLine": _am(ml), "pointSpread": _am(pts),
                                              "spread": _am(pr)}} if close else {})}
    return {"provider": {"name": name}, "overUnder": 222.5,
            "homeTeamOdds": side(home_ml, spread, hp),
            "awayTeamOdds": side(away_ml, "-" + spread[1:], ap)}


def test_a_close_block_parses_with_signs_from_the_home_side():
    r = E.parse_close(_item("ESPN BET"))
    assert (r["home_ml"], r["away_ml"], r["home_spread"]) == (105.0, -125.0, 1.5)
    assert (r["home_spread_price"], r["away_spread_price"]) == (-105.0, -115.0)
    assert r["total"] == 222.5 and r["provider"] == "ESPN BET"


def test_a_live_feed_is_never_a_close_and_no_close_block_means_no_row():
    assert E.parse_close(_item("ESPN Bet - Live Odds")) is None
    assert E.parse_close(_item("ESPN BET", close=False)) is None


def test_the_preferred_book_wins_else_any_book_with_a_close():
    items = [_item("Caesars Sportsbook", home_ml="+110"), _item("ESPN BET", home_ml="+105")]
    assert E.pick(items)["provider"] == "ESPN BET"
    assert E.pick([_item("Some Book", home_ml="+120")])["home_ml"] == 120.0
    assert E.pick([_item("ESPN Bet - Live Odds")]) is None


def test_even_money_reads_as_plus_one_hundred():
    assert E._american({"moneyLine": {"american": "EVEN"}}, "moneyLine") == 100.0


def test_the_season_window_spans_the_new_year_for_winter_leagues():
    days = E.season_days("nhl", 2025)
    assert days[0] == "20240925" and days[-1] == "20250430"
    assert E.season_days("mlb", 2024)[0] == "20240315"


def test_a_book_with_both_closes_beats_the_preferred_book_with_one():
    """2023's first-choice books often closed a spread and no moneyline."""
    no_ml = _item("ESPN BET")
    no_ml["homeTeamOdds"]["close"].pop("moneyLine")
    full = _item("Bet365", home_ml="+115")
    assert E.pick([no_ml, full])["provider"] == "Bet365"
