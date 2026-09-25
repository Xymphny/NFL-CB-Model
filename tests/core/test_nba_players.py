"""The NBA Players tab: availability and minutes, and the promise that the
model never reads them.

The ingest's parsers run on captured response shapes; the export is built
from synthetic box scores so the minutes arithmetic is checked exactly.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))

import export_players as E  # noqa: E402
from model.ingest import nba_players as NP  # noqa: E402

NAMES = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS"}


def _inj(name, player, status, pos="G", **det):
    return {"displayName": name, "injuries": [{
        "status": status, "date": "2026-10-20T15:00Z", "shortComment": "c",
        "details": det, "athlete": {"displayName": player,
                                    "position": {"abbreviation": pos}}}]}


def test_the_injury_report_is_keyed_by_code_and_sorted_worst_first():
    payload = {"injuries": [
        {"displayName": "Boston Celtics", "injuries": [
            {"status": "Probable", "athlete": {"displayName": "B One"}},
            {"status": "Out", "athlete": {"displayName": "A Two"},
             "details": {"side": "Left", "type": "Ankle", "returnDate": "2026-11-01"}},
            {"status": "Active", "athlete": {"displayName": "Ignored"}},
        ]},
    ]}
    rep = NP.parse_injuries(payload, NAMES)
    assert [r["status"] for r in rep["BOS"]] == ["Out", "Probable"]
    assert rep["BOS"][0]["injury"] == "Left Ankle"
    assert rep["BOS"][0]["return_date"] == "2026-11-01"
    assert "_unmatched" not in rep


def test_an_unspecified_side_is_left_out_of_the_injury():
    rep = NP.parse_injuries({"injuries": [_inj("Boston Celtics", "X", "Out",
                                               side="Not Specified", type="Back")]}, NAMES)
    assert rep["BOS"][0]["injury"] == "Back"


def test_an_unknown_team_name_is_named_not_dropped_silently():
    rep = NP.parse_injuries({"injuries": [_inj("Seattle SuperSonics", "X", "Out")]}, NAMES)
    assert rep["_unmatched"] == ["Seattle SuperSonics"]


def test_every_team_in_the_table_resolves():
    """The real table is what the live parser uses."""
    names = NP.name_to_code()
    assert len(set(names.values())) == 30


def test_an_unchanged_report_writes_nothing(tmp_path):
    f = tmp_path / "r.json"
    assert NP.write_if_changed(f, {"a": 1}) is True
    assert NP.write_if_changed(f, {"a": 1}) is False
    assert NP.write_if_changed(f, {"a": 2}) is True


def _summary(team="Boston Celtics", players=(("P1", "35", True), ("P2", None, False))):
    ath = []
    for i, (name, mins, starter) in enumerate(players):
        row = {"athlete": {"id": str(i), "displayName": name, "position": {"abbreviation": "F"}},
               "starter": starter}
        if mins is None:
            row.update(didNotPlay=True, stats=[])
        else:
            row["stats"] = [mins, "20"]
        ath.append(row)
    return {"boxscore": {"players": [{"team": {"displayName": team, "abbreviation": "BOS"},
                                      "statistics": [{"labels": ["MIN", "PTS"], "athletes": ath}]}]}}


def test_a_box_score_parses_minutes_and_did_not_play():
    rows = NP.parse_box(_summary(), {"season": 2027, "game_id": "g1", "date": "2026-10-21T23:30Z"},
                        NAMES)
    assert [(r["player"], r["minutes"], r["did_not_play"], r["starter"]) for r in rows] == [
        ("P1", 35, False, True), ("P2", 0, True, False)]
    assert all(r["team"] == "BOS" and r["game_id"] == "g1" for r in rows)


def test_box_scores_are_fetched_once(tmp_path, monkeypatch):
    sdv = tmp_path / "sdv"
    sdv.mkdir()
    pd.DataFrame({"id": [1, 2], "season": [2027, 2027], "season_type": [2, 2],
                  "type_abbreviation": ["STD", "STD"], "status_type_completed": [True, False],
                  "date": ["2026-10-21T23:30Z", "2026-10-22T23:30Z"]}).to_parquet(sdv / "nba_2027.parquet")
    monkeypatch.setattr(NP, "name_to_code", lambda: NAMES)
    calls = []
    fetch = lambda gid: calls.append(gid) or _summary()          # noqa: E731
    assert NP.update_box(2027, raw=tmp_path, sdv=sdv, fetch=fetch) == (1, 0)
    assert NP.update_box(2027, raw=tmp_path, sdv=sdv, fetch=fetch) == (0, 0)
    assert calls == ["1"], "a cached final was fetched again, or an unfinished game was fetched"


def _box(games):
    """games: [(game_id, date, {player: minutes or None})] for BOS."""
    rows = []
    for gid, d, players in games:
        for i, (p, m) in enumerate(players.items()):
            rows.append({"team": "BOS", "game_id": gid, "date": d, "athlete_id": p, "player": p,
                         "position": "G", "starter": i == 0, "minutes": m or 0,
                         "did_not_play": m is None})
    return pd.DataFrame(rows)


def test_the_rotation_is_the_last_ten_games_by_total_minutes():
    games = [(f"g{i:02d}", f"2026-11-{i + 1:02d}", {"Star": 36, "Bench": 12})
             for i in range(12)]
    games[-1][2]["Star"] = None                                   # sat the latest game
    rot, n = E.rotation(_box(games), "BOS")
    assert n == 10
    star = next(r for r in rot if r["player"] == "Star")
    assert (star["played"], star["missed"], star["min_per_game"]) == (9, 1, 36.0)
    assert rot[0]["player"] == "Star"                             # most total minutes first


def test_before_opening_night_there_are_no_rotations_and_it_says_why(tmp_path):
    (tmp_path / "injuries_current.json").write_text(json.dumps({"BOS": []}))
    a = E.build(today=date(2026, 9, 25), raw=tmp_path)
    assert a["season"] == 2027 and a["minutes_through"] is None
    assert "rosters change" in a["minutes_note"]
    assert all(t["rotation"] == [] for t in a["teams"])
    assert len(a["teams"]) == 30


def test_the_export_has_no_clock_of_its_own(tmp_path):
    """An unchanged day must write identical bytes, or the live-inputs job
    commits every run."""
    (tmp_path / "injuries_current.json").write_text(json.dumps({}))
    one = E.build(today=date(2026, 9, 25), raw=tmp_path)
    two = E.build(today=date(2026, 9, 25), raw=tmp_path)
    assert one == two
    assert not any(k in one for k in ("generated_at", "generated", "built_at"))


def test_it_is_labelled_and_never_read_by_a_pricing_path():
    """The whole point of the tab: the model is team-level, and this is the
    information it cannot see. If a pricing or staking path ever reads it,
    that is an ungraded player model -- ADR 0024's exact failure."""
    art = E.build(today=date(2026, 9, 25))
    assert art["not_a_model_input"] is True and "Not a model input" in art["note"]
    for f in [*(ROOT / "src" / "coverline").rglob("*.py"), ROOT / "scripts" / "export_board.py",
              ROOT / "scripts" / "recommend_slate.py", ROOT / "scripts" / "paper_trade.py"]:
        text = f.read_text()
        for needle in ("injuries_current", "player_box_", "players_nba", "nba_players"):
            assert needle not in text, f"{f.relative_to(ROOT)} reads {needle}"


def test_the_live_inputs_job_refreshes_and_commits_it():
    import deploy.live_inputs_job as J
    names = [n for n, _ in J.steps(date(2026, 10, 21))]
    assert names.index("nba_players") == names.index("nba") + 1
    assert "data/raw/nba" in J.PATHS and "data/site/players_nba.json" in J.PATHS
