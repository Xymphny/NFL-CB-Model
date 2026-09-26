"""The CFB board and the outlier flag (dashboard v2 item 8)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import export_board as X  # noqa: E402


@pytest.mark.parametrize("start,want", [
    ("2026-09-25T23:30:00Z", "Friday night"),           # 7:30 pm ET Friday
    ("2026-09-26T16:00:00Z", "Saturday noon"),          # noon
    ("2026-09-26T18:59:00Z", "Saturday noon"),          # 2:59 pm
    ("2026-09-26T19:00:00Z", "Saturday afternoon"),     # 3 pm
    ("2026-09-26T22:59:00Z", "Saturday afternoon"),     # 6:59 pm
    ("2026-09-26T23:00:00Z", "Saturday prime time"),    # 7 pm
    ("2026-09-27T02:00:00Z", "Saturday late"),          # 10 pm
    ("2026-09-27T03:30:00Z", "Saturday late"),          # 11:30 pm
    ("2026-10-01T00:00:00Z", "Wednesday"),
])
def test_kickoff_windows_are_eastern(start, want):
    assert X.cfb_window(start) == want


def _game(side, line, mu, status="priced"):
    return {"away": "A", "home": "H", "headline": "spread", "model": {"margin_mean": mu},
            "markets": {"spread": {"status": status, "side": side, "line": line}}}


def test_the_check_flag_follows_the_leagues_threshold_and_not_the_tier():
    board = {"tiers": {"outlier_points": 28.24}, "games": [
        _game("away", 34.5, 1.6),       # the brief's example: SMU by 1.6 vs by 34.5
        _game("home", -10.0, 8.0)]}
    X.attach_check_flags(board, "spread")
    f = board["games"][0]["check_flag"]
    assert f["gap_points"] == 32.9 and f["market_margin"] == 34.5 and f["threshold"] == 28.24
    assert "check_flag" not in board["games"][1]
    assert "tier" not in f, "display only: the tier is untouched"
    none = {"tiers": {}, "games": [_game("away", 34.5, 1.6)]}
    X.attach_check_flags(none, "spread")
    assert "check_flag" not in none["games"][0], "no threshold, no flag"


def test_every_not_priced_reason_is_listed_zero_included():
    games = [dict(_game("home", -3, 1, "refused"), refusal="GameNotPriceable: a team is not rated"),
             dict(_game("home", -3, 1), started=True),
             dict(_game("home", -3, 1, "refused"), refusal="no market price captured for this game")]
    got = {x["key"]: x["games"] for x in X.not_priced({"games": games})}
    assert got == {"no_line": 1, "unrated": 1, "whole_number": 0, "started": 1, "neutral": 0, "other": 0}


def test_outlier_points_is_the_99th_percentile_of_the_backtest_gap():
    from model import derive_tier_thresholds as D
    rows = pd.DataFrame({"model_margin": list(range(100)) + [0.0], "market_margin": [0.0] * 101})
    assert D.outlier_points(rows) == pytest.approx((rows.model_margin - rows.market_margin).abs().quantile(0.99), abs=0.01)
    assert D.outlier_points(pd.DataFrame({"p_model": [0.5]})) is None     # moneyline leagues
    art = json.loads((ROOT / "data" / "tier_thresholds.json").read_text())
    assert art["leagues"]["cfb"]["outlier_points"] > art["leagues"]["nfl"]["outlier_points"]
    assert "outlier_points" not in art["leagues"]["mlb"]
