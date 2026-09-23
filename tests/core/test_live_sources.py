"""The live sources for MLB, NHL and NBA: parity with the graded code, then refusals.

Each source replays a fit script's own walk-forward up to the moment asked.
The first test for each is that the replay IS the graded procedure -- same
numbers, to the last digit -- because a live price computed by a lookalike
of the graded code is an ungraded price. Everything after that is a refusal:
a missing season, a stale pull, a game already started, a starter unknown.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.leagues.mlb import live as mlb  # noqa: E402
from coverline.leagues.nba import live as nba  # noqa: E402
from coverline.leagues.nhl import live as nhl  # noqa: E402
from model import fit_nba, fit_nba_walkforward, fit_nhl_rules  # noqa: E402
from model.mlb_model import run_walk_forward  # noqa: E402
from tests.core.test_conformance import assert_distribution_conforms  # noqa: E402

SDV = ROOT / "data" / "raw" / "sportsdataverse"
FITTED = json.loads((ROOT / "data" / "nba_fitted.json").read_text())


# ================================================================== NBA ====

@pytest.fixture(scope="module")
def nba_2023():
    return nba.NBALiveSource.load(2023)


def test_nba_full_replay_reproduces_every_fitted_rating(nba_2023):
    ratings, last = nba_2023.ratings_before("2099-01-01")
    assert last == 2023
    assert set(ratings) == set(FITTED["ratings"])
    for team, r in FITTED["ratings"].items():
        assert round(ratings[team], 4) == r, team


def test_nba_games_priced_at_tip_reproduce_the_graded_margins(nba_2023):
    """Every 20th 2023 game plus the season's first three, where the
    carryover between seasons is applied by this source, not the loop."""
    hp = FITTED["hyperparameters"]
    df = fit_nba.load((2021, 2022, 2023))
    pred = fit_nba_walkforward.walk_forward(df, hp["k"], hp["home_adv"], hp["carryover"])
    order = df.sort_values(["season", "date"]).reset_index(drop=True)
    order["mu"] = pred.mu.values
    games = order[order.season == 2023].reset_index(drop=True)
    model = nba.build_model(nba_2023)
    sched = nba_2023.schedule.reset_index()
    checked = 0
    for i in list(range(3)) + list(range(3, len(games), 20)):
        g = games.iloc[i]
        row = sched[(sched.home == g.home) & (sched.away == g.away)
                    & (sched.tip == pd.Timestamp(g.date))].iloc[0]
        if row.neutral:
            continue
        dist = model.predict(row.game_id, row.tip.isoformat())
        assert dist.mu_margin == pytest.approx(g.mu, abs=1e-12)
        checked += 1
    assert checked > 50
    assert_distribution_conforms(dist, "nba live")


def test_nba_refuses_a_missing_season_and_names_the_file():
    # A season that cannot be on disk yet, so this does not depend on what
    # has been pulled -- the first version used 2027 and broke the day it was.
    with pytest.raises(nba.SeasonGap, match=r"nba_20\d\d\.parquet"):
        nba.NBALiveSource.load(2099)


def test_nba_season_is_named_by_the_year_it_ends():
    from datetime import date
    assert nba.season_of(date(2026, 10, 21)) == 2027
    assert nba.season_of(date(2027, 4, 12)) == 2027


@pytest.fixture()
def nba_dir(tmp_path):
    """2021-2022 as committed; 2023 with three late games moved to 2030 and
    made upcoming, so they can be priced 'now' from the other 1,227."""
    for y in (2021, 2022):
        shutil.copy(SDV / f"nba_{y}.parquet", tmp_path / f"nba_{y}.parquet")
    d = pd.read_parquet(SDV / "nba_2023.parquet")
    std = d[(d.season_type == 2) & (d.type_abbreviation == "STD") & ~d.neutral_site]
    ids = std.sort_values("date").id.iloc[-3:].tolist()
    tips = ["2030-01-15T23:30Z", "2030-01-16T00:00Z", "2030-01-17T00:00Z"]
    for i, tip in zip(ids, tips):
        m = d.id == i
        d.loc[m, "date"] = tip
        d.loc[m, ["home_score", "away_score"]] = 0
        d.loc[m, "status_type_completed"] = False
        d.loc[m, "status_type_name"] = "STATUS_SCHEDULED"
    d.to_parquet(tmp_path / "nba_2023.parquet")
    return tmp_path, [str(i) for i in ids]


def test_nba_prices_upcoming_games_from_every_completed_one(nba_dir):
    root, ids = nba_dir
    src = nba.NBALiveSource.load(2023, data=str(root / "nba_{year}.parquet"))
    state: dict = {}
    hp = FITTED["hyperparameters"]
    fit_nba_walkforward.walk_forward(
        fit_nba.load((2021, 2022, 2023), data=str(root / "nba_{year}.parquet")),
        hp["k"], hp["home_adv"], hp["carryover"], state=state)
    g = src.schedule.loc[ids[0]]
    f = src.features(ids[0], "now")
    assert f.rating_diff == pytest.approx(
        state["ratings"][g.home] - state["ratings"][g.away], abs=1e-12)
    assert math.isnan(f.pace)            # no pace model; not a plausible number
    # The slate is the US Eastern date: 00:00Z on the 16th is 7pm on the 15th.
    assert [x.game_id for x in src.slate("2030-01-15")] == ids[:2]
    assert [x.game_id for x in src.slate("2030-01-16")] == [ids[2]]
    assert_distribution_conforms(nba.build_model(src).predict(ids[0], "now"), "nba")


def test_nba_refuses_to_price_a_game_after_it_started(nba_dir):
    root, ids = nba_dir
    src = nba.NBALiveSource.load(2023, data=str(root / "nba_{year}.parquet"))
    with pytest.raises(nba.LookaheadRefused):
        src.features(ids[0], "2030-01-16T01:00:00Z")


def test_nba_refuses_a_stale_file(nba_dir):
    """Last night's game still unfinished in the file means the pull is behind."""
    root, ids = nba_dir
    src = nba.NBALiveSource.load(2023, data=str(root / "nba_{year}.parquet"))
    with pytest.raises(nba.StaleResults, match="not final"):
        src.features(ids[2], "2030-01-16T23:00:00Z")


def test_nba_refuses_while_a_team_is_mid_game(nba_dir):
    root, ids = nba_dir
    src = nba.NBALiveSource.load(2023, data=str(root / "nba_{year}.parquet"))
    first = src.schedule.loc[ids[0]]
    s = src.schedule
    s.loc[ids[2], ["home", "tip"]] = [first.home, pd.Timestamp("2030-01-15T22:00Z")]
    with pytest.raises(nba.GameNotPriceable, match="started and is not final"):
        src.features(ids[0], "2030-01-15T23:00:00Z")


def test_nba_an_in_progress_score_is_never_replayed_as_a_result(nba_dir):
    root, ids = nba_dir
    path = root / "nba_2023.parquet"
    d = pd.read_parquet(path)
    before = fit_nba.load((2023,), data=str(root / "nba_{year}.parquet"))
    d.loc[d.id == int(ids[0]), ["home_score", "away_score"]] = [54, 49]  # half-time
    d.to_parquet(path)
    after = fit_nba.load((2023,), data=str(root / "nba_{year}.parquet"))
    assert len(after) == len(before)


def test_nba_refuses_neutral_sites(nba_2023):
    s = nba_2023.schedule
    gid = s[s.neutral].index[0]
    with pytest.raises(nba.GameNotPriceable, match="neutral site"):
        nba_2023.features(gid, s.loc[gid].tip.isoformat())


# ================================================================== NHL ====

def _nhl_dir(tmp_path, season=2025):
    """Committed finals and goals, plus a schedule whose start order matches
    the graded ordering (date, then game id)."""
    for f in (f"nhl_{season}.parquet", f"goals_{season}.parquet"):
        shutil.copy(fit_nhl_rules.RAW / f, tmp_path / f)
    fin = pd.read_parquet(tmp_path / f"nhl_{season}.parquet")
    fin = fin.assign(key=fin.game_date + "#" + fin.game_id.astype(str)).sort_values("key")
    rk = fin.groupby("game_date").cumcount()
    start = pd.to_datetime(fin.game_date) + pd.Timedelta(hours=16) + pd.to_timedelta(rk, unit="m")
    pd.DataFrame({
        "season": season, "game_id": fin.game_id, "game_date": fin.game_date,
        "start_utc": start.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team_abbr": fin.home_team_abbr, "away_team_abbr": fin.away_team_abbr,
        "game_state": "OFF", "neutral_site": fin.neutral_site,
    }).to_parquet(tmp_path / f"schedule_{season}.parquet", index=False)
    return tmp_path


def test_nhl_next_game_rates_are_the_graded_walk_forward_row():
    games = fit_nhl_rules.load((2025,))
    graded = fit_nhl_rules.rate(games, "nopull_h", "nopull_a")
    for i in (0, 1, 2, 300, 900, len(games) - 1):
        lh, la = nhl.nopull_rates_next(games.iloc[:i], games.home_team_abbr[i],
                                       games.away_team_abbr[i])
        assert (lh, la) == (graded.lam_h[i], graded.lam_a[i])


def test_nhl_source_prices_at_puck_drop_exactly_as_graded(tmp_path):
    src = nhl.NHLLiveSource.load(2025, raw=_nhl_dir(tmp_path))
    graded = fit_nhl_rules.rate(fit_nhl_rules.load((2025,)), "nopull_h", "nopull_a")
    games = src.games.reset_index(drop=True)
    model = nhl.build_model(src)
    checked = 0
    for i in range(0, len(games), 60):
        g = games.iloc[i]
        if src.schedule.loc[g.game_id].neutral_site:
            continue
        f = src.features(g.game_id, g.start.isoformat())
        assert (f.lam_home_nopull, f.lam_away_nopull) == (graded.lam_h[i], graded.lam_a[i])
        assert math.isnan(f.lam_home)      # final-score rates are not produced
        checked += 1
    assert checked > 15
    dist = model.predict(g.game_id, g.start.isoformat())
    assert type(dist).__name__ == "NHLFinalScoreDistribution"
    assert_distribution_conforms(dist, "nhl live")


def test_nhl_refuses_a_season_not_pulled(tmp_path):
    with pytest.raises(nhl.MissingSeasonData, match="nhl_api.py --seasons 2027-2027"):
        nhl.NHLLiveSource.load(2027, raw=tmp_path)


def test_nhl_refuses_a_final_with_no_goal_rows(tmp_path):
    raw = _nhl_dir(tmp_path)
    gl = pd.read_parquet(raw / "goals_2025.parquet")
    fin = pd.read_parquet(raw / "nhl_2025.parquet")
    victim = fin[(fin.home_score + fin.away_score) > 3].game_id.iloc[10]
    gl[gl.game_id != victim].to_parquet(raw / "goals_2025.parquet")
    with pytest.raises(nhl.StaleResults, match="0-0"):
        nhl.NHLLiveSource.load(2025, raw=raw)


def test_nhl_refuses_a_pull_that_is_behind(tmp_path):
    raw = _nhl_dir(tmp_path)
    s = pd.read_parquet(raw / "schedule_2025.parquet")
    s.loc[500, "game_state"] = "FUT"
    s.to_parquet(raw / "schedule_2025.parquet")
    src = nhl.NHLLiveSource.load(2025, raw=raw)
    late = src.games.iloc[-1]
    with pytest.raises(nhl.StaleResults, match="not final"):
        src.features(late.game_id, late.start.isoformat())


def test_nhl_refuses_after_puck_drop(tmp_path):
    src = nhl.NHLLiveSource.load(2025, raw=_nhl_dir(tmp_path))
    g = src.games.iloc[100]
    with pytest.raises(nhl.LookaheadRefused):
        src.features(g.game_id, (g.start + pd.Timedelta(minutes=1)).isoformat())


# ================================================================== MLB ====

@pytest.fixture(scope="module")
def caches():
    return mlb.load_caches()


def test_mlb_state_before_a_game_reproduces_its_graded_expected_runs(caches):
    sched, pit = caches
    walk = run_walk_forward(sched, pit).set_index("game_key")
    dup = set(sched.game_key[sched.game_key.duplicated()])
    for key in ("NYA202207150", "SFN202603250", walk.index[-1]):
        assert key not in dup
        r = walk.loc[key]
        st = mlb.state_before(sched, pit, r.date, key)
        assert st.expected_runs(r.home_team, r.away_sp, r.away_team, r.park) == r.exp_home
        assert st.expected_runs(r.away_team, r.home_sp, r.home_team, r.park) == r.exp_away


def test_mlb_live_replays_across_seasons_like_every_fit_did(caches):
    """The cold-start WalkForwardSource replays the current season alone;
    every grade here ran the historical cache continuously."""
    sched, _ = caches
    assert sched.season.min() == 2021 and sched.season.max() == 2026


def _keys_on(day):
    sched, _ = mlb.load_caches()
    return sorted(sched[(sched.date == day) & sched.home_score.notna()].game_key)


def _slate(tmp_path, day="2026-09-22", prev="2026-09-21", stamp="20260922T150000Z",
           keys=None, unfinished=(), **over):
    game = {"game_pk": 1, "game_key": f"SFN{day.replace('-', '')}0", "game_number": 0,
            "start_utc": f"{day}T22:15:00Z", "state": "Preview",
            "detailed_state": "Scheduled", "home_team": "SFN", "away_team": "MIN",
            "home_name": "San Francisco Giants", "away_name": "Minnesota Twins",
            "home_sp": "tidwb001", "away_sp": "mattz001", "park": "SFO03"}
    game.update(over)
    (tmp_path / f"{day}-{stamp}.json").write_text(json.dumps(
        {"date": day, "fetched_at": stamp, "previous_final_date": prev,
         "previous_final_keys": _keys_on(prev) if keys is None else list(keys),
         "unfinished_keys": list(unfinished), "games": [game]}))
    return tmp_path


def test_mlb_prices_tonight_from_the_state_before_today(tmp_path, caches):
    src = mlb.MLBLiveSource.load("2026-09-22", slate_dir=_slate(tmp_path))
    f = src.features("SFN202609220", "2026-09-22T18:00:00Z")
    st = mlb.state_before(*caches, "2026-09-22")
    assert f.exp_home == st.expected_runs("SFN", "mattz001", "MIN", "SFO03")
    assert f.exp_away == st.expected_runs("MIN", "tidwb001", "SFN", "SFO03")
    dist = mlb.build_model(src).predict("SFN202609220", "2026-09-22T18:00:00Z")
    assert_distribution_conforms(dist, "mlb live")
    assert [g.game_id for g in src.slate("2026-09-22")] == ["SFN202609220"]


def test_mlb_refuses_a_game_with_no_probable(tmp_path):
    src = mlb.MLBLiveSource.load("2026-09-22", slate_dir=_slate(tmp_path, away_sp=None))
    with pytest.raises(mlb.GameNotPriceable, match="no probable starter for away"):
        src.features("SFN202609220", "2026-09-22T18:00:00Z")


def test_mlb_refuses_when_results_have_not_caught_up(tmp_path):
    with pytest.raises(mlb.StaleResults, match="2026-09-25"):
        mlb.MLBLiveSource.load("2026-09-26", slate_dir=_slate(
            tmp_path, day="2026-09-26", prev="2026-09-25", keys=["NYA202609250"]))


def test_mlb_refuses_a_cache_that_reaches_the_date_but_not_every_game(tmp_path):
    """The first check compared dates only. Five of fifteen results reach the
    same date as all fifteen; this is the case it let through."""
    keys = _keys_on("2026-09-21")
    assert keys, "no results on that date in the committed cache"
    with pytest.raises(mlb.StaleResults, match="1 of"):
        mlb.MLBLiveSource.load("2026-09-22", slate_dir=_slate(
            tmp_path, keys=keys + ["TOR202609210"]))


def test_mlb_refuses_a_slate_pulled_while_games_were_in_progress(tmp_path):
    with pytest.raises(mlb.StaleResults, match="Re-pull"):
        mlb.MLBLiveSource.load("2026-09-22", slate_dir=_slate(
            tmp_path, unfinished=["SEA202609210"]))


def test_mlb_refuses_a_slate_without_game_level_proof(tmp_path):
    _slate(tmp_path)
    f = next(tmp_path.glob("*.json"))
    blob = json.loads(f.read_text())
    del blob["previous_final_keys"]
    f.write_text(json.dumps(blob))
    with pytest.raises(mlb.StaleResults, match="predates"):
        mlb.MLBLiveSource.load("2026-09-22", slate_dir=tmp_path)


def test_mlb_refuses_a_date_with_no_slate(tmp_path):
    with pytest.raises(mlb.NoSlate, match="mlb_slate.py --date 2026-09-22"):
        mlb.MLBLiveSource.load("2026-09-22", slate_dir=tmp_path)


def test_mlb_reads_the_newest_pull_for_the_date(tmp_path):
    _slate(tmp_path, stamp="20260922T120000Z", away_sp=None)
    _slate(tmp_path, stamp="20260922T170000Z")
    src = mlb.MLBLiveSource.load("2026-09-22", slate_dir=tmp_path)
    assert src.payload["fetched_at"] == "20260922T170000Z"


def test_mlb_refuses_after_first_pitch(tmp_path):
    src = mlb.MLBLiveSource.load("2026-09-22", slate_dir=_slate(tmp_path))
    with pytest.raises(mlb.LookaheadRefused):
        src.features("SFN202609220", "2026-09-22T23:00:00Z")
