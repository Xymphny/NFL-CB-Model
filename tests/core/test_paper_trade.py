"""Automated paper trading: every captured window becomes graded evidence.

Without this the ledger only grows when someone remembers to run the operator
command with --commit, and a grade that depends on remembering never arrives.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_trade as PT  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402


@pytest.fixture(autouse=True)
def _site_exports_go_to_tmp(monkeypatch, tmp_path):
    """The capture job re-exports the board and record after paper trading
    (scripts/capture.py), and both exporters write to data/site by default.
    Without this, every run of these tests rewrote the real, cron-published
    site files in the working tree -- changes nobody made, one `git commit
    -am` away from overwriting what the jobs publish."""
    import export_board
    import export_record
    monkeypatch.setattr(export_board, "SITE", tmp_path / "site")
    monkeypatch.setattr(export_record, "SITE", tmp_path / "site")

SDV = ROOT / "data" / "raw" / "sportsdataverse"


def _event(eid, sport, start, home, away, line=-2.5, books=("pinnacle", "fanduel")):
    return {"id": eid, "sport_key": sport, "commence_time": start,
            "home_team": home, "away_team": away,
            "bookmakers": [{"key": b, "markets": [{"key": "spreads", "outcomes": [
                {"name": home, "price": 1.91, "point": line},
                {"name": away, "price": 1.91, "point": -line}]}]} for b in books]}


def _quotes(events):
    return normalize(events, captured_at="x")


def test_only_the_captured_window_is_priced():
    ev = [_event("a", "basketball_nba", "2026-10-20T19:00:00Z", "H", "A"),   # started
          _event("b", "basketball_nba", "2026-10-20T23:00:00Z", "H", "A"),
          _event("c", "basketball_nba", "2026-10-21T02:00:00Z", "H", "A"),   # 10pm ET, same date
          _event("d", "basketball_nba", "2026-10-22T23:00:00Z", "H", "A")]   # days out
    runs, cutoff = PT.slates("basketball_nba", "2026-10-20T22:30:00Z", _quotes(ev))
    assert cutoff == "2026-10-21T01:30:00Z"
    assert runs == [["--league", "nba", "--date", "2026-10-20"]]
    runs, _ = PT.slates("basketball_nba", "2026-10-20T22:30:00Z", _quotes(ev), window_hours=4)
    assert runs == [["--league", "nba", "--date", "2026-10-20"]]    # 02:00Z is still the 20th ET


def test_a_window_spanning_midnight_runs_each_eastern_date():
    ev = [_event("a", "icehockey_nhl", "2026-10-21T02:30:00Z", "H", "A"),   # 22:30 ET on the 20th
          _event("b", "icehockey_nhl", "2026-10-21T04:30:00Z", "H", "A")]   # 00:30 ET on the 21st
    runs, _ = PT.slates("icehockey_nhl", "2026-10-21T02:00:00Z", _quotes(ev))
    assert [r[-1] for r in runs] == ["2026-10-20", "2026-10-21"]


def test_nfl_runs_by_week():
    sched = pd.DataFrame({"season": [2026, 2026], "week": [4, 5], "game_type": ["REG", "REG"],
                          "gameday": ["2026-09-27", "2026-10-04"]})
    ev = [_event("a", "americanfootball_nfl", "2026-09-27T17:00:00Z", "H", "A")]
    runs, _ = PT.slates("americanfootball_nfl", "2026-09-27T16:30:00Z", _quotes(ev),
                        schedule=sched)
    assert runs == [["--league", "nfl", "--season", "2026", "--week", "4"]]
    # CFB runs by date (tests/core/test_cfb_live.py).
    ev = [_event("a", "americanfootball_ncaaf", "2026-09-26T17:00:00Z", "H", "A")]
    assert PT.slates("americanfootball_ncaaf", "2026-09-26T16:30:00Z", _quotes(ev))[0] == [
        ["--league", "cfb", "--date", "2026-09-26"]]


def test_an_empty_window_prices_nothing():
    ev = [_event("d", "basketball_nba", "2026-10-22T23:00:00Z", "H", "A")]
    assert PT.slates("basketball_nba", "2026-10-20T22:30:00Z", _quotes(ev))[0] == []


@pytest.fixture()
def nba_upcoming(tmp_path, monkeypatch):
    """2021-2022 committed, 2023 with its last two games moved to 2030 and
    made upcoming, and the live loader pointed at that directory."""
    import shutil
    from coverline.leagues.nba import live
    for y in (2021, 2022):
        shutil.copy(SDV / f"nba_{y}.parquet", tmp_path / f"nba_{y}.parquet")
    d = pd.read_parquet(SDV / "nba_2023.parquet")
    std = d[(d.season_type == 2) & (d.type_abbreviation == "STD") & ~d.neutral_site]
    rows = std.sort_values("date").iloc[-2:]
    for tip, (_, r) in zip(("2030-01-15T23:30Z", "2030-01-16T00:30Z"), rows.iterrows()):
        m = d.id == r.id
        d.loc[m, "date"] = tip
        d.loc[m, ["home_score", "away_score"]] = 0
        d.loc[m, "status_type_completed"] = False
        d.loc[m, "status_type_name"] = "STATUS_SCHEDULED"
    d.to_parquet(tmp_path / "nba_2023.parquet")
    src = live.NBALiveSource.load(2023, data=str(tmp_path / "nba_{year}.parquet"))

    real = PT._runner
    import dataclasses

    def patched():
        mod = real()
        mod.LEAGUES["nba"] = dataclasses.replace(
            mod.LEAGUES["nba"], loader=lambda day: (live.build_model(src), src, src.slate(day)))
        return mod
    monkeypatch.setattr(PT, "_runner", patched)
    return rows


def test_a_capture_becomes_settleable_paper_trades(tmp_path, nba_upcoming):
    rows = nba_upcoming
    events = [_event(f"e{i}", "basketball_nba", t, r.home_display_name, r.away_display_name)
              for i, (t, (_, r)) in enumerate(zip(("2030-01-15T23:30:00Z", "2030-01-16T00:30:00Z"),
                                                  rows.iterrows()))]
    snap = BronzeStore(tmp_path / "bronze").write_snapshot(
        sport="basketball_nba", captured_at="2030-01-15T23:00:00Z", payload=events,
        cost=1, source_url="u").path
    res = PT.paper_trade(snap, tmp_path / "ledger")
    assert res == [(["--league", "nba", "--date", "2030-01-15"], 0)]
    sigs = BetLedger(tmp_path / "ledger").signals()
    assert len(sigs) == 8                            # 2 games x 2 books x 2 sides
    assert not any(s.placed for s in sigs)
    assert {s.game_id for s in sigs} == {str(r.id) for _, r in rows.iterrows()}
    assert all(s.side in ("home", "away") and s.book and s.price_decimal for s in sigs)


def test_the_capture_job_papers_what_it_captured_and_commits_both(monkeypatch, capsys, tmp_path):
    import capture as entry
    calls, commits = [], []
    monkeypatch.setenv("CAPTURE_ENABLED", "1")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(entry, "OddsAPIClient", lambda **kw: object())
    monkeypatch.setattr(entry, "plan", lambda *a, **k: [])
    monkeypatch.setattr(entry, "BRONZE_ROOT", tmp_path / "bronze")

    class Res:
        captured = ["basketball_nba|2030-01-15T23:00:00Z"]
        gapped: list = []
        credits_spent = 1

        def summary(self):
            return "1 captured"
    monkeypatch.setattr(entry, "run", lambda *a, **k: Res())
    monkeypatch.setattr(PT, "paper_trade", lambda p: calls.append(p) or [(["x"], 0)])
    monkeypatch.setitem(sys.modules, "paper_trade", PT)
    monkeypatch.setitem(sys.modules, "deploy.git_utils", type(sys)("deploy.git_utils"))
    sys.modules["deploy.git_utils"].git_commit_and_push = lambda p, m: commits.append((p, m))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / "bronze").mkdir()
    assert entry.main(["--run", "--paper", "--persist"]) == 0
    assert len(calls) == 1 and calls[0].name.startswith("current-2030-01-15")
    assert commits and "data/ledger" in commits[0][0] and "1 paper slate" in commits[0][1]


def test_a_paper_failure_never_loses_the_capture(monkeypatch, tmp_path):
    import capture as entry
    commits = []
    monkeypatch.setenv("CAPTURE_ENABLED", "1")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(entry, "OddsAPIClient", lambda **kw: object())
    monkeypatch.setattr(entry, "plan", lambda *a, **k: [])
    monkeypatch.setattr(entry, "BRONZE_ROOT", tmp_path / "bronze")

    class Res:
        captured = ["basketball_nba|2030-01-15T23:00:00Z"]
        gapped: list = []
        credits_spent = 1

        def summary(self):
            return "1 captured"
    monkeypatch.setattr(entry, "run", lambda *a, **k: Res())

    def boom(p):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(PT, "paper_trade", boom)
    monkeypatch.setitem(sys.modules, "paper_trade", PT)
    monkeypatch.setitem(sys.modules, "deploy.git_utils", type(sys)("deploy.git_utils"))
    sys.modules["deploy.git_utils"].git_commit_and_push = lambda p, m: commits.append((p, m))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / "bronze").mkdir()
    assert entry.main(["--run", "--paper", "--persist"]) == 0
    assert len(commits) == 1


def test_the_blueprint_papers_every_capture():
    import yaml
    job = next(s for s in yaml.safe_load((ROOT / "render.yaml").read_text())["services"]
               if s["name"] == "close-capture-job")
    assert "--paper" in job["startCommand"] and "--persist" in job["startCommand"]


def test_a_paper_run_is_never_placed_even_with_a_weight():
    R = PT._runner()
    with pytest.raises(SystemExit):
        R.main(["--league", "nba", "--date", "2030-01-15", "--bankroll", "1",
                "--paper", "--market-weight", "0.5"])
