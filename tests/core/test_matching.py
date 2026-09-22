"""The date-based matcher and the three team tables.

The matcher's value is in what it refuses, so most of these tests build a
payload that SHOULD fail and check that it fails in the named way -- including
one taken from real data: the duplicated doubleheader key in the MLB schedule
cache.
"""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.execution import matching as M  # noqa: E402
from coverline.execution.normalize import Quote  # noqa: E402
from coverline.leagues.mlb.teams import MLB_NAMES, TABLE as MLB  # noqa: E402
from coverline.leagues.nba.teams import TABLE as NBA  # noqa: E402
from coverline.leagues.nhl.teams import TABLE as NHL  # noqa: E402


def q(eid, home, away, ct, book="pinnacle", outcome=None, price=1.91):
    return Quote(event_id=eid, sport="x", commence_time=ct, home_team=home,
                 away_team=away, bookmaker=book, market="h2h",
                 outcome=outcome or home, price_decimal=price, point=None,
                 last_update=None, captured_at="t")


# ---------------------------------------------------------- primitives ----

def test_canonical_strips_accents_stops_case_and_spacing():
    assert M.canonical("Montréal Canadiens") == M.canonical("montreal  CANADIENS")
    assert M.canonical("St. Louis Blues") == M.canonical("St Louis Blues")


def test_late_pacific_starts_land_on_the_day_they_are_played():
    """A 7:10 pm PDT first pitch is 02:10 UTC the next day."""
    assert M.local_date("2026-09-22T02:10:00Z") == "2026-09-21"
    assert M.local_date("2026-09-21T17:05:00Z") == "2026-09-21"


def test_a_table_refuses_two_names_that_collide_to_different_codes():
    with pytest.raises(ValueError, match="refusing to guess"):
        M.TeamTable("x", {"St. Louis": "A", "St Louis": "B"}, True)


def test_an_unknown_name_raises_and_says_whether_the_table_was_verified():
    with pytest.raises(M.UnknownTeamName, match="never been checked"):
        NHL.to_code("Hartford Whalers")
    with pytest.raises(M.UnknownTeamName) as exc:
        MLB.to_code("Montreal Expos")
    assert "never been checked" not in str(exc.value)


# ------------------------------------------------------------ matching ----

GAMES = [M.ModelGame("G1", "2026-09-21", "BAL", "TOR")]


def test_a_simple_game_matches():
    m, r = M.match_by_date(
        [q("e1", "Baltimore Orioles", "Toronto Blue Jays", "2026-09-21T22:36:00Z")],
        MLB, GAMES)
    assert [(x.event_id, x.game_id) for x in m] == [("e1", "G1")]
    assert r == []


def test_a_doubleheader_pairs_by_start_time_in_order():
    games = [M.ModelGame("DH1", "2026-07-04", "BAL", "TOR", order=1),
             M.ModelGame("DH0", "2026-07-04", "BAL", "TOR", order=0)]
    quotes = [q("late", "Baltimore Orioles", "Toronto Blue Jays", "2026-07-04T23:05:00Z"),
              q("early", "Baltimore Orioles", "Toronto Blue Jays", "2026-07-04T17:05:00Z")]
    m, r = M.match_by_date(quotes, MLB, games)
    assert {x.event_id: x.game_id for x in m} == {"early": "DH0", "late": "DH1"}
    assert r == []


def test_a_doubleheader_with_mismatched_counts_is_refused_whole():
    games = [M.ModelGame("DH0", "2026-07-04", "BAL", "TOR", order=0)]
    quotes = [q("a", "Baltimore Orioles", "Toronto Blue Jays", "2026-07-04T17:05:00Z"),
              q("b", "Baltimore Orioles", "Toronto Blue Jays", "2026-07-04T23:05:00Z")]
    m, r = M.match_by_date(quotes, MLB, games)
    assert m == []
    assert {x.event_id for x in r} == {"a", "b"}
    assert all("refusing to guess" in x.reason for x in r)


def test_the_real_duplicated_doubleheader_key_is_refused():
    """ATL202606170 appears twice in model/mlb_schedule_current.csv with
    different scores and starters. Built from the file, not transcribed."""
    s = pd.read_csv(ROOT / "model" / "mlb_schedule_current.csv")
    dup = s[s.game_key == "ATL202606170"]
    assert len(dup) == 2, "the known duplicate has changed; re-read this test"
    games = [M.ModelGame(r.game_key, r.date, r.home_team, r.away_team, int(r.game_number))
             for r in dup.itertuples()]
    quotes = [q("g1", "Atlanta Braves", "San Francisco Giants", "2026-06-17T17:20:00Z"),
              q("g2", "Atlanta Braves", "San Francisco Giants", "2026-06-17T23:20:00Z")]
    m, r = M.match_by_date(quotes, MLB, games)
    assert m == []
    assert all("cannot be known" in x.reason for x in r)


def test_a_game_the_model_does_not_have_is_refused_by_name():
    m, r = M.match_by_date(
        [q("e9", "Boston Red Sox", "New York Yankees", "2026-09-21T23:10:00Z")],
        MLB, GAMES)
    assert m == [] and "no model game for NYA@BOS" in r[0].reason


def test_every_event_lands_in_exactly_one_list():
    quotes = [q("ok", "Baltimore Orioles", "Toronto Blue Jays", "2026-09-21T22:36:00Z"),
              q("miss", "Boston Red Sox", "New York Yankees", "2026-09-21T23:10:00Z"),
              q("ok", "Baltimore Orioles", "Toronto Blue Jays", "2026-09-21T22:36:00Z",
                book="fanduel")]
    m, r = M.match_by_date(quotes, MLB, GAMES)
    ids = [x.event_id for x in m] + [x.event_id for x in r]
    assert sorted(ids) == ["miss", "ok"]


def test_an_unknown_name_is_refused_by_name_without_taking_the_slate_down():
    """It used to raise, which aborted every other game on the slate with a
    traceback -- on the NHL and NBA tables, which have not been checked
    against the feed, that is a question of when, not if. It is now a
    refusal carrying the table's own hint, and the rest still match."""
    m, r = M.match_by_date(
        [q("e", "Montreal Expos", "Toronto Blue Jays", "2026-09-21T22:36:00Z"),
         q("ok", "Baltimore Orioles", "Toronto Blue Jays", "2026-09-21T22:36:00Z")],
        MLB, GAMES)
    assert [x.event_id for x in r] == ["e"]
    assert "Montreal Expos" in r[0].reason
    assert [x.event_id for x in m] == ["ok"]
    _, r = M.match_by_date([q("h", "Quebec Nordiques", "Boston Bruins",
                              "2026-10-10T23:00:00Z")], NHL, [])
    assert "never been checked" in r[0].reason


# ---------------------------------------------------------- the tables ----

def test_the_mlb_table_is_exactly_what_the_observed_payloads_say():
    """Rebuilt from data/mlb_divergence/*.json, so the table cannot drift from
    the Odds API spellings it claims to have been verified against."""
    from deploy.mlb_daily_update import STATS_TO_RETRO
    pairs = defaultdict(set)
    for f in glob.glob(str(ROOT / "data" / "mlb_divergence" / "*.json")):
        for row in json.load(open(f))["divergences"]:
            for n, c in ((row["home_name"], row["home_team"]),
                         (row["away_name"], row["away_team"])):
                if len(c) <= 4:          # rows where the legacy join succeeded
                    pairs[n].add(c)
    assert all(len(v) == 1 for v in pairs.values())
    rebuilt = {n: STATS_TO_RETRO[next(iter(c))] for n, c in pairs.items()}
    assert rebuilt == MLB_NAMES


def test_mlb_codes_are_exactly_the_schedules_codes():
    s = pd.read_csv(ROOT / "model" / "mlb_schedule_current.csv")
    s = s[s.season == s.season.max()]
    assert MLB.codes() == set(s.home_team) | set(s.away_team)


def test_nhl_codes_cover_the_latest_committed_season():
    d = pd.read_parquet(ROOT / "data" / "raw" / "nhl" / "nhl_2025.parquet")
    assert set(d.home_team_abbr) | set(d.away_team_abbr) == NHL.codes()


def test_nba_codes_are_the_fitted_codes():
    fit = json.load(open(ROOT / "data" / "nba_fitted.json"))["ratings"]
    assert NBA.codes() == set(fit)


def test_the_aliases_resolve_to_the_only_team_they_could_mean():
    assert NHL.to_code("Utah Mammoth") == NHL.to_code("Utah Hockey Club") == "UTA"
    assert NBA.to_code("Los Angeles Clippers") == NBA.to_code("LA Clippers") == "LAC"


def test_only_mlb_claims_to_be_verified():
    assert MLB.verified and not NHL.verified and not NBA.verified
