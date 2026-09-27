"""Grades by point in the season (accuracy plan 2b), and the CFB hold."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from model import grade_by_week as G  # noqa: E402

ART = ROOT / "data" / "grade_by_week.json"


def test_every_league_is_graded_by_segment():
    a = json.loads(ART.read_text())
    assert set(a["leagues"]) == {"nfl", "cfb", "nba", "nhl", "mlb"}
    for league, g in a["leagues"].items():
        assert [s["label"] for s in g["segments"]] == [x[0] for x in G.SEGMENTS[league]]
        assert sum(s["n"] for s in g["segments"]) == g["all"]["n"], league


def test_a_segment_says_which_weeks_it_actually_holds():
    """Neither football cache has weeks 1-3: the '1-4' segment is week 4
    alone and must say so rather than stand for a month it never saw."""
    a = json.loads(ART.read_text())
    for league in ("nfl", "cfb"):
        first = a["leagues"][league]["segments"][0]
        assert first["weeks_covered"] == [4, 4], league


def test_a_small_segment_gets_no_w_hat_rather_than_a_noisy_one():
    rows = pd.DataFrame({"week": [1] * 10, "p_model": [0.6] * 10,
                         "p_market": [0.5] * 10, "y": [1, 0] * 5})
    s = G.segment(rows, 1, 4)
    assert (s["n"], s["w_hat"], s["preferred_side"]) == (10, None, 0.5)


def test_the_record_export_carries_the_artifact_verbatim(tmp_path):
    import export_record as X
    rec = X.record(ledger_dir=tmp_path)
    assert rec["by_week"] == json.loads(ART.read_text())


def _entry():
    return {"markets": {"spread": {"status": "priced", "tier": "lean", "stake_fraction": 0.0}}}


def test_the_cfb_early_hold_is_off_and_does_nothing():
    import export_board as X
    e = _entry()
    X.apply_cfb_early_hold(e, 3)
    assert e == _entry()


def test_switched_on_it_holds_weeks_1_to_4_and_not_5():
    import export_board as X
    e = _entry()
    X.apply_cfb_early_hold(e, 4, on=True)
    assert e["markets"]["spread"]["tier"] == "coin_flip"
    assert e["markets"]["spread"]["cap"]["rule"] == "cfb_early"
    e = _entry()
    X.apply_cfb_early_hold(e, 5, on=True)
    assert e == _entry()
