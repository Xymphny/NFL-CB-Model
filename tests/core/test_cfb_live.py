"""The live CFB source: weekly ratings against ESPN's schedule.

Before it, CFB priced only games in a 2021-2023 cache, so it could never be
paper-traded and would never reach the ledger floor that sizes a stake.
"""

import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_trade as PT  # noqa: E402
from coverline.execution import settle as S  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402
from coverline.execution.matching import UnknownTeamName  # noqa: E402
from coverline.leagues.cfb import live  # noqa: E402
from coverline.leagues.cfb.model import MARGIN_COEFFICIENTS_DVOA_ONLY as C  # noqa: E402
from coverline.leagues.cfb.teams import TABLE  # noqa: E402

#: id, week, kickoff (UTC), home, away, neutral, completed, score
GAMES = [
    (1, 3, "2030-09-14T19:30Z", "Ohio State", "Michigan", False, True, (31, 17)),
    (2, 4, "2030-09-21T16:00Z", "Ohio State", "Michigan", False, False, (0, 0)),
    (3, 4, "2030-09-21T19:30Z", "Georgia", "Alabama", True, False, (0, 0)),
    (4, 4, "2030-09-21T20:00Z", "Georgia", "Howard", False, False, (0, 0)),
    # 10:30 pm Eastern on the 21st: the 22nd in UTC.
    (5, 4, "2030-09-22T02:30Z", "Alabama", "Michigan", False, False, (0, 0)),
    (6, 5, "2030-09-28T16:00Z", "Michigan", "Georgia", False, False, (0, 0)),
]
RATINGS = {"Ohio State": 0.30, "Michigan": 0.10, "Georgia": 0.25, "Alabama": 0.20}


def _write(tmp, weeks=((4, "2030-09-15T10:00:00+00:00"),)):
    raw, rd = tmp / "raw", tmp / "ratings"
    raw.mkdir(), rd.mkdir()
    pd.DataFrame([{
        "game_id": i, "season": 2030, "week": w, "game_date": k[:10], "start": k,
        "neutral_site": n, "home_team": h, "away_team": a, "home_score": sc[0],
        "away_score": sc[1], "completed": done, "status": "x"}
        for i, w, k, h, a, n, done, sc in GAMES]).to_parquet(raw / "espn_2030.parquet")
    for wk, at in weeks:
        (rd / f"2030-week-{wk:02d}.json").write_text(json.dumps({
            "league": "CFB", "season": 2030, "week": wk, "computed_at": at,
            "ratings": [{"team": t, "total_rating": v, "elo_rating": 1500.0}
                        for t, v in RATINGS.items()]}))
    return raw, rd


@pytest.fixture()
def src(tmp_path):
    raw, rd = _write(tmp_path)
    return live.CFBLiveSource.load(2030, raw=raw, ratings_dir=rd)


BEFORE = "2030-09-20T12:00:00Z"


def test_a_current_game_is_priced_on_the_dvoa_only_vector(src):
    f = src.features("2", BEFORE)
    assert f.rating_diff == pytest.approx(0.20) and f.elo_present is False
    d = live.build_model(src).predict("2", BEFORE)
    # mu_margin: the predictor. margin_mean() is the key-number-weighted pmf mean.
    assert d.mu_margin == pytest.approx(C["intercept"] + C["rating_diff"] * 0.20)


def test_a_neutral_site_game_is_refused(src):
    with pytest.raises(live.GameNotPriceable, match="neutral site"):
        src.features("3", BEFORE)


def test_an_unrated_team_is_refused_not_zeroed(src):
    with pytest.raises(live.GameNotPriceable, match="not rated"):
        src.features("4", BEFORE)


def test_a_started_game_is_refused(src):
    with pytest.raises(live.LookaheadRefused):
        src.features("2", "2030-09-21T16:00:01Z")


def test_ratings_computed_after_kickoff_are_never_used(tmp_path):
    raw, rd = _write(tmp_path, weeks=((3, "2030-09-08T10:00:00+00:00"),
                                      (4, "2030-09-21T17:00:00+00:00")))   # after game 2
    s = live.CFBLiveSource.load(2030, raw=raw, ratings_dir=rd)
    assert s.snapshot_for(s.schedule.loc["2"].start, pd.Timestamp("2030-09-22", tz="UTC")).week == 3
    assert s.snapshot_for(s.schedule.loc["5"].start, pd.Timestamp("2030-09-22T01:00Z")).week == 4


def test_ratings_two_weeks_behind_are_refused_one_week_is_tolerated(tmp_path):
    raw, rd = _write(tmp_path, weeks=((3, "2030-09-08T10:00:00+00:00"),))
    s = live.CFBLiveSource.load(2030, raw=raw, ratings_dir=rd)
    s.features("2", BEFORE)                                  # week 4 on week 3: ok
    with pytest.raises(live.StaleRatings, match="2 weeks behind"):
        s.features("6", "2030-09-27T12:00:00Z")              # week 5 on week 3


def test_missing_inputs_name_the_command_that_fixes_them(tmp_path):
    with pytest.raises(live.MissingSeasonData, match="cfb_espn.py --seasons 2031-2031"):
        live.CFBLiveSource.load(2031, raw=tmp_path, ratings_dir=tmp_path)


def test_slates_use_the_eastern_date(src):
    assert [g.game_id for g in src.slate("2030-09-21")] == ["2", "3", "4", "5"]
    assert src.slate("2030-09-22") == []
    assert src.slate("2030-09-14") == []                     # played
    week, games = src.week_slate("2030-09-16")
    assert week == 4 and {g.date for g in games} == {"2030-09-21"}


def test_the_season_is_named_for_its_autumn():
    assert live.season_of("2030-09-21") == 2030 and live.season_of("2031-01-05") == 2030


def test_the_team_table_maps_feed_names_to_espn_and_refuses_the_rest():
    assert TABLE.verified
    assert TABLE.to_code("Ohio State Buckeyes") == "Ohio State"
    assert TABLE.to_code("Miami (OH) RedHawks") == "Miami (OH)"
    assert TABLE.to_code("Miami Hurricanes") == "Miami"
    assert TABLE.to_code("Hawaii Rainbow Warriors") == "Hawai'i"
    with pytest.raises(UnknownTeamName):
        TABLE.to_code("Utah Tech Trailblazers")


def test_every_mapped_school_is_spelled_as_espn_spells_it():
    """The codes are the live source's team names, so one ESPN does not use
    would match nothing, silently."""
    espn = set()
    for p in sorted((ROOT / "data" / "raw" / "cfb").glob("espn_*.parquet")):
        d = pd.read_parquet(p, columns=["home_team", "away_team"])
        espn |= set(d.home_team) | set(d.away_team)
    assert TABLE.codes() <= espn, sorted(TABLE.codes() - espn)


def test_finals_are_completed_games_only(tmp_path):
    _write(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "raw").rename(tmp_path / "data" / "raw" / "cfb")
    f = S.cfb_finals(tmp_path)
    assert f == {"1": (31, 17)}
    assert S.FINALS["cfb"] is S.cfb_finals


def test_cfb_captures_run_by_eastern_date():
    ev = [{"id": "a", "sport_key": "americanfootball_ncaaf",
           "commence_time": "2030-09-22T02:30:00Z", "home_team": "H", "away_team": "A",
           "bookmakers": []}]
    from coverline.execution.normalize import normalize
    q = normalize([{**ev[0], "bookmakers": [{"key": "b", "markets": [{"key": "spreads", "outcomes": [
        {"name": "H", "price": 1.91, "point": -3.5}, {"name": "A", "price": 1.91, "point": 3.5}]}]}]}],
        captured_at="x")
    runs, _ = PT.slates("americanfootball_ncaaf", "2030-09-22T00:00:00Z", q)
    assert runs == [["--league", "cfb", "--date", "2030-09-21"]]


def _event(eid, start, home, away, line):
    return {"id": eid, "sport_key": "americanfootball_ncaaf", "commence_time": start,
            "home_team": home, "away_team": away,
            "bookmakers": [{"key": b, "markets": [{"key": "spreads", "outcomes": [
                {"name": home, "price": 1.91, "point": line},
                {"name": away, "price": 1.91, "point": -line}]}]}
                for b in ("pinnacle", "fanduel")]}


def test_a_cfb_capture_becomes_settleable_paper_trades(tmp_path, monkeypatch, src):
    real = PT._runner

    def patched():
        mod = real()
        mod.LEAGUES["cfb"] = dataclasses.replace(
            mod.LEAGUES["cfb"], loader=lambda day: (live.build_model(src), src, src.slate(day)))
        return mod
    monkeypatch.setattr(PT, "_runner", patched)
    monkeypatch.setattr(live, "_ts", lambda a: pd.Timestamp("2030-09-21T15:00Z")
                        if a == "now" else pd.Timestamp(a))
    events = [_event("e2", "2030-09-21T16:00:00Z", "Ohio State Buckeyes", "Michigan Wolverines", -3.5),
              _event("e3", "2030-09-21T16:30:00Z", "Georgia Bulldogs", "Alabama Crimson Tide", -1.5),
              _event("e9", "2030-09-21T17:00:00Z", "Utah Tech Trailblazers", "Utah Utes", -7.5)]
    snap = BronzeStore(tmp_path / "bronze").write_snapshot(
        sport="americanfootball_ncaaf", captured_at="2030-09-21T15:00:00Z", payload=events,
        cost=1, source_url="u").path
    res = PT.paper_trade(snap, tmp_path / "ledger")
    assert res == [(["--league", "cfb", "--date", "2030-09-21"], 0)]
    sigs = BetLedger(tmp_path / "ledger").signals()
    # Only Ohio State-Michigan: Georgia-Alabama is neutral, Utah Tech is unknown.
    assert {s.game_id for s in sigs} == {"2"} and len(sigs) == 4
    assert not any(s.placed for s in sigs) and all(s.league == "cfb" for s in sigs)
