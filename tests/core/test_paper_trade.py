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


def _capture_job(monkeypatch, tmp_path, codes):
    """Run the capture entrypoint with one captured snapshot, each child step
    answering from `codes` ({script name: return code}). Returns (calls,
    commits)."""
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
    Res.paths = {Res.captured[0]: str(BronzeStore(tmp_path / "bronze").snapshot_path(
        "basketball_nba", "2030-01-15T23:00:00Z"))}
    monkeypatch.setattr(entry, "run", lambda *a, **k: Res())

    def child(argv):
        calls.append(argv)
        return codes.get(Path(argv[0]).stem, 0)
    monkeypatch.setattr(entry, "_isolated", child)
    monkeypatch.setitem(sys.modules, "deploy.git_utils", type(sys)("deploy.git_utils"))
    sys.modules["deploy.git_utils"].git_commit_and_push = lambda p, m: commits.append((p, m))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / "bronze").mkdir()
    assert entry.main(["--run", "--paper", "--persist"]) == 0
    return calls, commits


def test_the_capture_job_papers_what_it_captured_and_commits_both(monkeypatch, tmp_path):
    calls, commits = _capture_job(monkeypatch, tmp_path, {})
    assert [Path(c[0]).stem for c in calls] == ["paper_trade", "export_board", "export_record"]
    assert Path(calls[0][calls[0].index("--snapshot") + 1]).name.startswith("current-2030-01-15")
    paths, msg = commits[0]
    assert {"data/ledger", "data/site"} <= set(paths) and "1 paper snapshot" in msg


def test_a_paper_failure_never_loses_the_capture(monkeypatch, tmp_path):
    _, commits = _capture_job(monkeypatch, tmp_path, {"paper_trade": 1})
    assert len(commits) == 1 and "bronze" in commits[0][0][0]


def test_an_out_of_memory_kill_still_commits_the_snapshot(monkeypatch, tmp_path):
    """The failure this replaced: in-process, an OOM kill took the whole job
    down before the commit, and Render's disk took the snapshot with it."""
    _, commits = _capture_job(monkeypatch, tmp_path, {"paper_trade": -9, "export_board": -9})
    paths, _ = commits[0]
    assert paths == ["bronze"], "a killed child's half-written ledger or site was committed"


def test_a_failed_export_does_not_publish_the_site(monkeypatch, tmp_path):
    _, commits = _capture_job(monkeypatch, tmp_path, {"export_record": 1})
    assert "data/site" not in commits[0][0] and "data/ledger" in commits[0][0]


def test_a_child_step_really_runs_in_its_own_process():
    import capture as entry
    assert entry._isolated(["-c", "import sys; sys.exit(3)"]) == 3
    assert entry._isolated(["-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]) == -9


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


def test_an_early_snapshot_paper_trades_the_next_day(monkeypatch, tmp_path):
    """From the daily early poll the window is a full day, so every game gets
    exactly one early trade; from a close it stays WINDOW_HOURS."""
    seen = []
    monkeypatch.setattr(PT, "slates", lambda sport, at, q, window_hours, cutoff=None: seen.append(window_hours) or ([], "x"))
    store = BronzeStore(tmp_path / "bronze")
    for kind in ("early", "current"):
        snap = store.write_snapshot(sport="basketball_nba", captured_at="2026-10-21T14:00:00Z",
                                    payload=[], cost=1, source_url="u", kind=kind).path
        PT.paper_trade(snap, tmp_path / "ledger")
    assert seen == [PT.EARLY_WINDOW_HOURS, PT.WINDOW_HOURS]


def test_a_cfb_early_price_runs_to_the_next_ratings_update():
    """cfb-weekly-job refreshes the ratings Sunday 10:00 UTC; an early trade
    must never price past it on the old ones."""
    c = PT.early_cutoff
    assert c("cfb", "2026-09-25T14:00:00Z", []) == "2026-09-27T10:00:00Z"     # Fri -> Sun
    assert c("cfb", "2026-09-27T14:00:00Z", []) == "2026-10-04T10:00:00Z"     # Sun after update
    assert c("cfb", "2026-09-27T09:00:00Z", []) == "2026-09-27T10:00:00Z"     # Sun before it
    assert c("mlb", "2026-09-25T14:00:00Z", []) is None                       # daily: 24h


def test_an_nfl_early_price_covers_the_week_of_the_next_game_and_no_further():
    sched = pd.DataFrame({"season": [2026] * 4, "week": [4, 4, 4, 5], "game_type": ["REG"] * 4,
                          "gameday": ["2026-10-01", "2026-10-04", "2026-10-05", "2026-10-08"]})
    ev = [_event("thu", "americanfootball_nfl", "2026-10-02T00:15:00Z", "H", "A"),
          _event("sun", "americanfootball_nfl", "2026-10-04T17:00:00Z", "H", "A"),
          _event("mon", "americanfootball_nfl", "2026-10-06T00:15:00Z", "H", "A"),   # MNF, ET 5th
          _event("nxt", "americanfootball_nfl", "2026-10-09T00:15:00Z", "H", "A")]   # week 5
    q = _quotes(ev)
    cut = PT.early_cutoff("nfl", "2026-09-29T14:00:00Z", q, schedule=sched)
    assert cut == "2026-10-06T00:15:01Z"
    runs, _ = PT.slates("americanfootball_nfl", "2026-09-29T14:00:00Z", q, schedule=sched, cutoff=cut)
    assert runs == [["--league", "nfl", "--season", "2026", "--week", "4"]]


def test_an_early_poll_skips_games_already_priced(monkeypatch, tmp_path):
    """One early trade per game -- its first. Without the skip, seven daily
    polls re-trade a CFB week seven times: rows in git, no information."""
    argvs = []

    class R:
        @staticmethod
        def main(argv):
            argvs.append(argv)
            return 0
    monkeypatch.setattr(PT, "_runner", lambda: R)
    monkeypatch.setattr(PT, "already_priced", lambda d, lg: {"old"})
    ev = [_event("old", "americanfootball_ncaaf", "2026-09-26T17:00:00Z", "H", "A"),
          _event("new", "americanfootball_ncaaf", "2026-09-26T20:00:00Z", "H", "A")]
    store = BronzeStore(tmp_path / "bronze")
    snap = store.write_snapshot(sport="americanfootball_ncaaf", captured_at="2026-09-25T14:00:00Z",
                                payload=ev, cost=1, source_url="u", kind="early").path
    assert PT.paper_trade(snap, tmp_path / "ledger") == [(["--league", "cfb", "--date", "2026-09-26"], 0)]
    argv = argvs[0]
    assert argv[argv.index("--skip-events") + 1] == "old"
    assert argv[argv.index("--events-before") + 1] == "2026-09-27T10:00:00Z"

    monkeypatch.setattr(PT, "already_priced", lambda d, lg: {"old", "new"})
    assert PT.paper_trade(snap, tmp_path / "ledger") == []          # nothing new: no run


def test_a_close_never_skips(monkeypatch, tmp_path):
    """The close is the trade results are graded on; it always prices."""
    argvs = []

    class R:
        @staticmethod
        def main(argv):
            argvs.append(argv)
            return 0
    monkeypatch.setattr(PT, "_runner", lambda: R)
    monkeypatch.setattr(PT, "already_priced", lambda d, lg: {"g"})
    ev = [_event("g", "americanfootball_ncaaf", "2026-09-26T17:00:00Z", "H", "A")]
    snap = BronzeStore(tmp_path / "bronze").write_snapshot(
        sport="americanfootball_ncaaf", captured_at="2026-09-26T16:55:00Z",
        payload=ev, cost=1, source_url="u").path
    assert len(PT.paper_trade(snap, tmp_path / "ledger")) == 1
    assert "--skip-events" not in argvs[0]


def test_a_game_already_in_play_is_never_paper_traded(tmp_path, nba_upcoming):
    """The odds feed lists games in progress. The runner priced them from
    their live line whenever the model slate still held them -- a paper
    trade taken after its own close."""
    rows = nba_upcoming
    events = [_event(f"e{i}", "basketball_nba", t, r.home_display_name, r.away_display_name)
              for i, (t, (_, r)) in enumerate(zip(("2030-01-15T23:30:00Z", "2030-01-16T00:30:00Z"),
                                                  rows.iterrows()))]
    snap = BronzeStore(tmp_path / "bronze").write_snapshot(
        sport="basketball_nba", captured_at="2030-01-16T00:00:00Z", payload=events,
        cost=1, source_url="u").path                  # the 23:30 game is 30 min in
    PT.paper_trade(snap, tmp_path / "ledger")
    sigs = BetLedger(tmp_path / "ledger").signals()
    assert {s.event_id for s in sigs} == {"e1"}
