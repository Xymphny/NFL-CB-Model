"""Team pages, outlook and per-team record (dashboard v2 items 7, 12-14)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import export_teams as T  # noqa: E402


def test_ranks_are_one_for_best_nulls_unranked_ties_share():
    assert T.ranks({"a": 3, "b": 1, "c": None, "d": 3}, True) == {"a": 1, "b": 3, "c": None, "d": 1}
    assert T.ranks({"a": 3, "b": 1}, False) == {"a": 2, "b": 1}


def test_division_place_is_win_pct_then_point_differential():
    teams = {"A": {"division": "X", "record": {"w": 2, "l": 0, "t": 0, "pf": 40, "pa": 30}},
             "B": {"division": "X", "record": {"w": 2, "l": 0, "t": 0, "pf": 60, "pa": 30}},
             "C": {"division": "X", "record": {"w": 1, "l": 1, "t": 0, "pf": 90, "pa": 10}}}
    assert T.division_places(teams) == {"B": 1, "A": 2, "C": 3}


def test_the_nfl_page_ranks_every_stat_in_its_direction():
    art = T.build_nfl(2026)
    assert len(art["teams"]) == 32 and art["footnote"] == T.NFL_FOOTNOTE
    t = {x["abbr"]: x for x in art["teams"]}
    best_d = min(t, key=lambda a: t[a]["model"]["defense_voa"]["value"])
    assert t[best_d]["model"]["defense_voa"]["rank"] == 1, "lower defence VOA is better"
    hardest = max(t, key=lambda a: t[a]["schedule"]["remaining"]["value"] or -9)
    assert t[hardest]["schedule"]["remaining"]["rank"] == 1, "schedule rank 1 is the hardest"
    assert all(len(x["next"]) <= 3 for x in art["teams"])


def test_the_nba_page_comes_from_the_season_file_the_model_reads():
    art = T.build_nba(2026)
    assert art["of"] == 30 and art["outlook"] == "Not simulated yet."
    t = art["teams"][0]
    assert t["record"]["rank"] and t["rating"]["value"] is not None and t["first_game"]
    assert {x["conference"] for x in art["teams"]} == {"East", "West"}


def test_the_nhl_page_says_the_model_does_not_use_it():
    art = T.build_nhl()
    assert art["banner"].startswith("NOT USED BY THE MODEL") and len(art["teams"]) == 32
    assert all(x["gp"] == 82 for x in art["teams"])


def test_mlb_bullpen_workload_is_marked_not_in_the_price():
    art = T.build_mlb("2026-09-26")
    assert len(art["teams"]) == 30 and "58%" in art["note"]
    assert all(t["bullpen_workload"]["in_price"] is False for t in art["teams"])
    assert all(0 <= t["pen_ra27"]["credibility"] <= 1 for t in art["teams"])


def test_the_outlook_is_playoff_odds_only_and_labelled(monkeypatch):
    import export_outlook as O
    art = O.build(2026, n=4)
    assert set(art) >= {"playoff_pct", "label", "n", "ratings_week"}
    assert not any("division" in k for k in art), "division odds are a proxy; not published"
    assert art["label"].endswith("QB changes aren't in it.")
    assert art["residual_sd_used"] == 13.0 and 0 <= min(art["playoff_pct"].values())


def test_the_record_counts_a_game_for_both_teams(tmp_path):
    import export_record as XR
    from coverline.execution.ledger import BetLedger, Outcome, Signal
    led = BetLedger(tmp_path)
    for side, sel, pm in (("home", "Boston Celtics", 0.6), ("away", "New York Knicks", 0.4)):
        led.record(Signal(signal_id=f"s{side}", at="2030-01-15T20:00:00Z", league="nba",
                          event_id="e1", market="spreads", selection=sel, line=-2.5,
                          p_model=pm, p_market=0.5, p_used=0.5, shrinkage=0.0,
                          edge_claimed=pm - 0.5, edge_used=0.0, disposition="not_placed",
                          not_placed_reason="below_threshold", book="pinnacle",
                          price_decimal=1.91, side=side, game_id="g1",
                          commence_time="2030-01-15T23:30:00Z"))
    led.record_outcome(Outcome(signal_id="shome", at="2030-01-16T03:00:00Z", result="win",
                               home_score=110, away_score=100))
    got = XR.record(tmp_path)["leagues"]["nba"]
    assert set(got["teams"]) == {"BOS", "NY"} and got["team_floor"] == 10
    assert got["teams"]["BOS"] == {"settled": 1, "wins": 1, "mean_clv": None}
