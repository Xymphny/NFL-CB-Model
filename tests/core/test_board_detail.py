"""Matchup detail in the board export (dashboard v2 items 9 and 10)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import board_detail as D  # noqa: E402
from coverline.core.distributions import NormalMarginDistribution  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from tests.core.test_board_export import make_nba_night  # noqa: E402


def _dist(**kw):
    return NormalMarginDistribution(mu_margin=kw.pop("mu", 2.5), sd_margin=13.3,
                                    mu_total=45, sd_total=10, **kw)


def test_the_pmf_is_the_distributions_own_mass_with_tails_kept():
    d = _dist(discrete=True)
    pm = D.margin_pmf("nfl", d)
    assert (pm["from"], pm["to"], len(pm["mass"])) == (-28, 28, 57)
    assert sum(pm["mass"]) + pm["tail_below"] + pm["tail_above"] == pytest.approx(1.0, abs=1e-4)
    assert pm["mass"][28 + 3] == pytest.approx(d.margin_pmf(3), abs=1e-5)
    assert pm["binned"] is False


def test_nba_is_continuous_so_its_mass_is_binned_and_says_so():
    pm = D.margin_pmf("nba", _dist(discrete=False))
    assert pm["binned"] is True and sum(pm["mass"]) > 0.99


def test_key_numbers_come_from_the_measured_table():
    keys = D.key_numbers("nfl")
    assert 3 in keys and 7 in keys and -3 in keys
    assert D.key_numbers("nba") == [] and D.key_numbers("mlb") == []


def test_a_withheld_total_says_why():
    assert D.totals_view("nfl", _dist(total_validated=False)) == {
        "priced": False, "reason": "Not priced: the totals model failed its grade."}
    assert D.totals_view("cfb", _dist(total_validated=False))["reason"].endswith("graded.")


def _event(eid, home_pt=-3.0, prices=((1.91, 1.91), (1.95, 1.87))):
    return {"id": eid, "home_team": "H", "away_team": "A", "commence_time": "2030-01-01T18:00:00Z",
            "bookmakers": [{"key": f"b{i}", "markets": [{"key": "spreads", "outcomes": [
                {"name": "H", "price": h, "point": home_pt},
                {"name": "A", "price": a, "point": -home_pt}]}]}
                for i, (h, a) in enumerate(prices)]}


def test_one_capture_is_the_median_across_books():
    s = D.summarise_event(_event("e"), "spreads")
    assert s["home_line"] == -3.0 and s["books"] == 2
    assert s["home_price"] == pytest.approx(1.93) and 0.49 < s["p_home"] < 0.51


def test_history_reads_every_capture_and_never_fills_a_gap(tmp_path):
    store = BronzeStore(tmp_path)
    for at, pt in (("2029-12-30T14:00:00Z", -3.0), ("2029-12-31T14:00:00Z", -3.5)):
        store.write_snapshot(sport="nfl", captured_at=at, payload=[_event("e", pt)], cost=3,
                             source_url="u", kind="early")
    import pandas as pd
    hist = D.history(store, "nfl", {"e": "spreads"}, pd.Timestamp("2030-01-01T12:00:00Z"))
    assert [p["home_line"] for p in hist["e"]] == [-3.0, -3.5]
    one = D.for_side(hist["e"][:1], "home", "2030-01-01T18:00:00Z", [])
    assert one["captures"] == 1 and one["points"][0]["line"] == -3.0   # a valid state
    away = D.for_side(hist["e"], "away", "2030-01-01T18:00:00Z", [])
    assert [p["line"] for p in away["points"]] == [3.0, 3.5]


def test_the_close_is_marked_only_once_the_game_has_started():
    pts = [{"captured_at": "2020-01-01T14:00:00Z", "kind": "early", "home_line": -3.0,
            "home_price": 1.9, "away_price": 1.9, "p_home": 0.5, "books": 3},
           {"captured_at": "2020-01-01T17:50:00Z", "kind": "current", "home_line": -3.5,
            "home_price": 1.9, "away_price": 1.9, "p_home": 0.5, "books": 3}]
    done = D.for_side(pts, "home", "2020-01-01T18:00:00Z", [])
    assert done["frozen"] and done["close_at"] == "2020-01-01T17:50:00Z"
    ahead = D.for_side(pts, "home", "2099-01-01T18:00:00Z", [])
    assert not ahead["frozen"] and ahead["close_at"] is None


def test_only_this_games_missed_windows_are_listed():
    pts = [{"captured_at": "2020-01-01T14:00:00Z", "kind": "early", "home_line": -3.0,
            "home_price": 1.9, "away_price": 1.9, "p_home": 0.5, "books": 3}]
    gaps = [{"intended_at": "2020-01-01T17:55:00Z", "reason": "window_missed"},   # its close
            {"intended_at": "2020-01-01T16:00:00Z", "reason": "window_missed"},   # another game
            {"intended_at": "2020-01-01T14:00:00Z", "reason": "early_window_missed"}]
    got = D.for_side(pts, "home", "2020-01-01T18:00:00Z", gaps)["missed_windows"]
    assert [g["intended_at"] for g in got] == ["2020-01-01T17:55:00Z", "2020-01-01T14:00:00Z"]


def test_the_board_carries_the_detail(tmp_path):
    board, _, _ = make_nba_night(tmp_path)
    g = board["games"][0]
    assert g["model"]["margin_pmf"]["binned"] is True
    assert g["line_history"]["captures"] == 2 and not g["line_history"]["frozen"]
    assert isinstance(g["fair_line"], float)
