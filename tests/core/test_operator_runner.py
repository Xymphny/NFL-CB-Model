"""The five-league operator command, and the ingests that feed it.

The runner is where a misconfiguration costs money, so these check what it
REFUSES as much as what it prints: a football week passed to a date league,
inputs that are missing or behind, a weight nothing supports. The ingest
tests run the pure parsers on captured response shapes, because the
endpoints are unreachable from the build sandbox -- the first live run
verifies the rest, and each ingest says so.
"""

import dataclasses
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.leagues.nba import live as nba  # noqa: E402

RUNNER = ROOT / "scripts" / "recommend_slate.py"
SDV = ROOT / "data" / "raw" / "sportsdataverse"


def _runner():
    spec = importlib.util.spec_from_file_location("recommend_slate", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recommend_slate"] = mod    # dataclasses resolve through it
    spec.loader.exec_module(mod)
    return mod


def _run(*args):
    return subprocess.run([sys.executable, str(RUNNER), *args],
                          capture_output=True, text=True, cwd=ROOT)


# ------------------------------------------------------------ arguments ----

@pytest.mark.parametrize("league", ["cfb", "mlb", "nhl", "nba"])
def test_a_date_league_refuses_a_week(league):
    r = _run("--league", league, "--week", "2", "--bankroll", "100")
    assert r.returncode == 2 and "use --date, not --week" in r.stderr


def test_the_nfl_refuses_a_date_and_requires_a_week():
    r = _run("--league", "nfl", "--date", "2026-09-22", "--bankroll", "100")
    assert r.returncode == 2 and "--date does not apply" in r.stderr
    r = _run("--league", "nfl", "--bankroll", "100")
    assert r.returncode == 2 and "--week is required" in r.stderr


@pytest.mark.parametrize("league,fix", [
    ("mlb", "mlb_slate.py --date"),
    ("nhl", "nhl_api.py --seasons 2099-2099"),
    ("nba", "nba_2099.parquet"),
    ("cfb", "cfb_espn.py --seasons 2098-2098"),
])
def test_missing_inputs_exit_2_with_the_command_that_fixes_them(league, fix):
    # A date whose inputs cannot exist. The first version used 2026-10-22 and
    # passed only until that season was pulled.
    r = _run("--league", league, "--date", "2098-11-02", "--bankroll", "100")
    assert r.returncode == 2, r.stderr
    assert "inputs are not ready" in r.stdout and fix in r.stdout
    assert "Traceback" not in r.stderr


# --------------------------------------------------------------- weight ----

def test_date_leagues_paper_trade_by_default():
    r = _run("--league", "nba", "--date", "2026-10-22", "--bankroll", "100")
    assert "market grade: none for NBA" in r.stdout
    assert "PAPER TRADING" in r.stdout and "weight 0" in r.stdout
    assert "Nothing will be staked" in r.stdout


def test_a_weight_override_says_it_is_a_guess():
    r = _run("--league", "nhl", "--date", "2026-10-22", "--bankroll", "100",
             "--market-weight", "0.3")
    assert "OVERRIDES" in r.stdout and "a guess with money on it" in r.stdout
    r = _run("--league", "nhl", "--bankroll", "100", "--market-weight", "1.5")
    assert r.returncode == 2 and "between 0 and 1" in r.stderr


def test_football_is_sized_by_its_market_grade_and_says_it_is_an_upper_bound():
    r = _run("--league", "nfl", "--week", "2", "--bankroll", "100")
    assert "market grade: w_hat" in r.stdout and "staking weight 0.000" in r.stdout
    assert "UPPER BOUND" in r.stdout
    assert "Nothing will be staked" in r.stdout


def test_every_league_has_a_team_source_and_a_vendor_key():
    R = _runner()
    assert set(R.LEAGUES) == {"nfl", "cfb", "mlb", "nhl", "nba"}
    for name, lg in R.LEAGUES.items():
        if lg.slate == "date":
            assert importlib.import_module(lg.team_table).TABLE.league == name
    from scripts.capture import SPORTS
    assert {lg.vendor_sport for lg in R.LEAGUES.values()} == set(SPORTS)


# ------------------------------------------------------------- snapshots ----

def test_a_snapshot_stored_under_the_vendor_key_is_found(tmp_path):
    """capture.py writes 'americanfootball_nfl'; the runner used to look for
    'nfl' and report no snapshot forever."""
    R = _runner()
    store = BronzeStore(tmp_path)
    store.write_snapshot(sport="americanfootball_nfl", captured_at="2026-09-24T12:00:00Z",
                         payload=[], cost=3, source_url="x")
    assert R.latest_snapshot(store, "nfl") is not None
    assert R.latest_snapshot(store, "nba") is None


# ------------------------------------------------------------ end to end ----

@pytest.fixture()
def nba_slate(tmp_path):
    """Committed NBA data with one late-2023 game made upcoming in 2030."""
    for y in (2021, 2022):
        shutil.copy(SDV / f"nba_{y}.parquet", tmp_path / f"nba_{y}.parquet")
    d = pd.read_parquet(SDV / "nba_2023.parquet")
    std = d[(d.season_type == 2) & (d.type_abbreviation == "STD") & ~d.neutral_site]
    row = std.sort_values("date").iloc[-1]
    m = d.id == row.id
    d.loc[m, "date"] = "2030-01-15T23:30Z"
    d.loc[m, ["home_score", "away_score"]] = 0
    d.loc[m, "status_type_completed"] = False
    d.loc[m, "status_type_name"] = "STATUS_SCHEDULED"
    d.to_parquet(tmp_path / "nba_2023.parquet")
    src = nba.NBALiveSource.load(2023, data=str(tmp_path / "nba_{year}.parquet"))
    mu = nba.build_model(src).predict(str(row.id), "now").mu_margin
    return src, row, mu


def _snapshot(tmp_path, row, home_line):
    store = BronzeStore(tmp_path / "bronze")
    event = {
        "id": "evt1", "sport_key": "basketball_nba",
        "commence_time": "2030-01-15T23:30:00Z",
        "home_team": row.home_display_name, "away_team": row.away_display_name,
        "bookmakers": [{"key": "pinnacle", "last_update": "2030-01-15T12:00:00Z",
                        "markets": [{"key": "spreads", "outcomes": [
                            {"name": row.home_display_name, "price": 1.95, "point": home_line},
                            {"name": row.away_display_name, "price": 1.95, "point": -home_line},
                        ]}]}],
    }
    return store.write_snapshot(sport="basketball_nba", captured_at="2030-01-15T12:00:00Z",
                                payload=[event], cost=1, source_url="x").path


def _main(monkeypatch, capsys, src, argv):
    R = _runner()
    monkeypatch.setitem(R.LEAGUES, "nba", dataclasses.replace(
        R.LEAGUES["nba"], loader=lambda day: (nba.build_model(src), src, src.slate(day))))
    code = R.main(argv)
    return code, capsys.readouterr().out


def test_an_nba_slate_runs_end_to_end_as_paper(tmp_path, monkeypatch, capsys, nba_slate):
    src, row, mu = nba_slate
    line = -(round(mu) + 0.5) + 4.0      # a half-point line well off the model
    snap = _snapshot(tmp_path, row, line)
    code, out = _main(monkeypatch, capsys, src,
                      ["--league", "nba", "--date", "2030-01-15", "--bankroll", "1000",
                       "--snapshot", str(snap)])
    assert code == 0, out
    assert "matched 1 events" in out
    assert "paper " in out and "0 recommended" in out
    assert "Dry run" in out


def test_the_same_slate_stakes_only_when_a_weight_is_forced(tmp_path, monkeypatch, capsys,
                                                             nba_slate):
    src, row, mu = nba_slate
    line = -(round(mu) + 0.5) + 4.0
    snap = _snapshot(tmp_path, row, line)
    code, out = _main(monkeypatch, capsys, src,
                      ["--league", "nba", "--date", "2030-01-15", "--bankroll", "1000",
                       "--snapshot", str(snap), "--market-weight", "1.0"])
    assert code == 0, out
    assert "BET" in out and "1 recommended" in out


def test_a_market_the_league_does_not_offer_is_refused(tmp_path, monkeypatch, capsys,
                                                       nba_slate):
    src, row, mu = nba_slate
    snap = _snapshot(tmp_path, row, -3.5)
    code, out = _main(monkeypatch, capsys, src,
                      ["--league", "nba", "--date", "2030-01-15", "--bankroll", "1000",
                       "--snapshot", str(snap), "--market", "total"])
    assert code == 2 and "total" in out


# --------------------------------------------------------------- ingests ----

def _nhl_game(gid, state, hs=None, as_=None):
    g = {"id": gid, "season": 20262027, "gameType": 2, "gameState": state,
         "startTimeUTC": "2026-10-10T23:00:00Z", "neutralSite": False,
         "homeTeam": {"abbrev": "TOR"}, "awayTeam": {"abbrev": "MTL"}}
    if hs is not None:
        g["homeTeam"]["score"], g["awayTeam"]["score"] = hs, as_
        g["gameOutcome"] = {"lastPeriodType": "REG"}
    return g


def test_nhl_ingest_keeps_only_final_scores_but_schedules_everything():
    from model.ingest.nhl_api import parse_week
    payload = {"gameWeek": [{"date": "2026-10-10", "games": [
        _nhl_game(1, "OFF", 3, 2),
        _nhl_game(2, "LIVE", 1, 0),           # a score, but not a result
        _nhl_game(3, "FUT"),
        {**_nhl_game(4, "OFF", 2, 1), "gameType": 1},   # preseason
    ]}]}
    finals, sched = parse_week(payload, 2027)
    assert [f["game_id"] for f in finals] == [1]
    assert [s["game_id"] for s in sched] == [1, 2, 3]
    assert {s["game_state"] for s in sched} == {"OFF", "LIVE", "FUT"}


def _espn_event(eid, home, away, hs, as_, completed, stype=2):
    comp = {"type": {"abbreviation": "STD"}, "neutralSite": False,
            "status": {"type": {"name": "STATUS_FINAL" if completed else "STATUS_SCHEDULED",
                                "completed": completed}},
            "competitors": [
                {"homeAway": "home", "score": str(hs),
                 "team": {"abbreviation": home, "displayName": home + " FC"}},
                {"homeAway": "away", "score": str(as_),
                 "team": {"abbreviation": away, "displayName": away + " FC"}}]}
    return {"id": str(eid), "date": "2026-10-21T23:30Z",
            "season": {"year": 2027, "type": stype}, "competitions": [comp]}


def test_nba_ingest_writes_what_the_loader_reads(tmp_path):
    from model.fit_nba import load
    from model.ingest.nba_espn import parse_scoreboard
    rows = parse_scoreboard({"events": [
        _espn_event(1, "BOS", "NY", 110, 104, True),
        _espn_event(2, "LAL", "GS", 0, 0, False),
        _espn_event(3, "PHX", "DEN", 99, 98, True, stype=1),   # preseason
    ]})
    assert [r["id"] for r in rows] == [1, 2]
    assert all(r["source"] == "espn_scoreboard" for r in rows)
    df = pd.DataFrame(rows)
    committed = pd.read_parquet(SDV / "nba_2025.parquet")
    used = ["season", "season_type", "type_abbreviation", "date", "home_abbreviation",
            "away_abbreviation", "home_score", "away_score", "status_type_completed",
            "id", "neutral_site", "status_type_name", "home_display_name"]
    assert set(used) <= set(df.columns) and set(used) <= set(committed.columns)
    df.to_parquet(tmp_path / "nba_2027.parquet")
    played = load((2027,), data=str(tmp_path / "nba_{year}.parquet"))
    assert played[["home", "away", "margin"]].values.tolist() == [["BOS", "NY", 6]]
    src = nba._schedule(str(tmp_path / "nba_2027.parquet"))
    assert src.loc["2"].completed == False and src.loc["2"].home == "LAL"  # noqa: E712


def test_nba_ingest_verify_finds_a_disagreement():
    from model.ingest.nba_espn import compare
    committed = pd.read_parquet(SDV / "nba_2025.parquet")
    assert compare(committed, committed) == []
    fresh = committed.copy()
    i = fresh[(fresh.season_type == 2) & (fresh.type_abbreviation == "STD")].index[5]
    fresh.loc[i, "home_score"] += 1
    diffs = compare(fresh, committed)
    assert len(diffs) == 1 and "home_score" in diffs[0]


def test_nba_ingest_will_not_overwrite_a_committed_season():
    from model.ingest.nba_espn import is_ours
    for y in (2021, 2022, 2023, 2024, 2025):
        assert not is_ours(SDV / f"nba_{y}.parquet")


def _mlb_game(pk, home, away, dh="N", num=1, hsp=None, asp=None, state="Preview"):
    side = lambda ab, sp: {"team": {"abbreviation": ab, "name": ab + " Club"},
                           **({"probablePitcher": {"id": sp, "fullName": "P"}} if sp else {})}
    return {"gamePk": pk, "gameDate": "2026-09-22T22:05:00Z", "doubleHeader": dh,
            "gameNumber": num, "status": {"abstractGameState": state,
                                          "detailedState": "Scheduled"},
            "teams": {"home": side(home, hsp), "away": side(away, asp)}}


def test_mlb_slate_keys_match_the_finals_the_cron_writes_next_morning():
    from deploy.mlb_daily_update import parse_schedule_finals
    from model.ingest.mlb_slate import parse_slate
    games = [_mlb_game(1, "ATL", "SF", "Y", 1, 111, 222),
             _mlb_game(2, "ATL", "SF", "Y", 2, 333, None),
             _mlb_game(3, "NYY", "BOS", hsp=444, asp=555)]
    slate = parse_slate({"dates": [{"date": "2026-09-22", "games": games}]},
                        "2026-09-22", bridge={111: "fried001"}, parks={"ATL": "ATL03"})
    assert [g["game_key"] for g in slate] == ["ATL202609220", "ATL202609221", "NYA202609220"]
    assert slate[0]["home_sp"] == "fried001" and slate[0]["away_sp"] == "mlbam222"
    assert slate[1]["away_sp"] is None
    assert slate[2]["park"] == "UNK"
    finals = [dict(g, status={"abstractGameState": "Final"}) for g in games]
    done = parse_schedule_finals({"dates": [{"date": "2026-09-22", "games": finals}]})
    keys = [f"{f['home_team']}{f['date'].replace('-', '')}{f['game_number']}" for f in done]
    assert keys == [g["game_key"] for g in slate]


def test_mlb_slate_knows_the_last_day_games_went_final():
    from model.ingest.mlb_slate import last_final_date
    payload = {"dates": [
        {"date": "2026-07-12", "games": [{"status": {"abstractGameState": "Final"}}]},
        {"date": "2026-07-14", "games": [{"status": {"abstractGameState": "Preview"}}]},
    ]}
    assert last_final_date(payload, "2026-07-17") == "2026-07-12"   # All-Star break


def test_mlb_slate_records_exactly_the_games_played_to_a_result():
    """Postponements come back abstractGameState Final with no score; they are
    not results. A game in progress is not either, and is recorded so the live
    source can refuse the slate."""
    from model.ingest.mlb_slate import finals_on, unfinished
    done = lambda g, detail="Final": dict(g, status={"abstractGameState": "Final",
                                                    "detailedState": detail})
    def scored(g):
        g = dict(g); g["teams"] = {k: dict(v, score=3) for k, v in g["teams"].items()}
        return g
    ppd = done(_mlb_game(4, "SEA", "HOU"), "Postponed")
    live = dict(_mlb_game(5, "COL", "ARI"), status={"abstractGameState": "Live"})
    payload = {"dates": [{"date": "2026-09-22", "games": [
        scored(done(_mlb_game(1, "ATL", "SF", "Y", 1))),
        scored(done(_mlb_game(2, "ATL", "SF", "Y", 2))),
        ppd, live]}]}
    assert finals_on(payload, "2026-09-22") == ["ATL202609220", "ATL202609221"]
    assert unfinished(payload, "2026-09-23") == ["COL202609220"]
    assert unfinished(payload, "2026-09-22") == []

