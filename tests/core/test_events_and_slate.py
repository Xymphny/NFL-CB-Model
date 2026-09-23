"""Event matching, and the runner that uses it.

The matcher is the join between the odds feed's vocabulary and the model's.
ADR 0004 is what happens when a join like this drifts unnoticed, so the tests
here are mostly about the join refusing to guess and about the duplicate table
being held to the legacy one.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.execution.normalize import Quote  # noqa: E402
from coverline.leagues.nfl import events as EV  # noqa: E402

RUNNER = ROOT / "scripts" / "recommend_slate.py"


def _q(event_id, home, away, outcome="H", price=1.91, point=-3.0):
    return Quote(event_id=event_id, sport="nfl", commence_time="",
                 home_team=home, away_team=away, bookmaker="pinnacle",
                 market="spreads", outcome=outcome, price_decimal=price,
                 point=point, last_update=None, captured_at="x")


# ----------------------------------------------------------- the mapping ----

def test_the_duplicate_table_agrees_with_the_legacy_one():
    """The guard that makes duplicating it acceptable. Two tables joined on
    the same key is exactly the ADR 0004 failure; the only thing that makes a
    second copy safe is this test."""
    from deploy.odds_watch_job import ODDS_TEAM_TO_ABBR as legacy
    assert EV.NAME_TO_CODE == legacy, (
        "the new-core name map has drifted from deploy/odds_watch_job.py. "
        "One of them is now wrong and nothing else would say which."
    )


def test_the_map_covers_the_whole_league():
    assert len(EV.NAME_TO_CODE) == 32
    assert len(set(EV.NAME_TO_CODE.values())) == 32


def test_the_codes_are_the_canonical_vocabulary():
    from coverline.leagues.nfl.ngs import CANONICAL_CODES
    assert set(EV.NAME_TO_CODE.values()) == set(CANONICAL_CODES)


def test_the_rams_map_to_the_ratings_code_not_the_ngs_one():
    """LA, not LAR. The whole point of ADR 0004."""
    assert EV.to_code("Los Angeles Rams") == "LA"


def test_an_unknown_name_raises_rather_than_fuzzy_matching():
    """A relocation or rebrand should stop the run, not quietly match a team
    to its predecessor."""
    with pytest.raises(EV.UnknownTeamName, match="deliberately"):
        EV.to_code("St. Louis Rams")
    with pytest.raises(EV.UnknownTeamName):
        EV.to_code("Los Angeles")


def test_whitespace_is_tolerated_but_nothing_else_is():
    assert EV.to_code("  Kansas City Chiefs  ") == "KC"


# ----------------------------------------------------------- matching ----

def test_events_resolve_to_model_game_ids():
    q = [_q("bk1", "Los Angeles Rams", "New York Giants"),
         _q("bk2", "Kansas City Chiefs", "Denver Broncos")]
    m = EV.match_events(q, season=2026, week=2)
    assert [x.game_id for x in m] == ["2026-W02-KC-DEN", "2026-W02-LA-NYG"]


def test_only_games_the_model_can_price_are_matched():
    q = [_q("bk1", "Los Angeles Rams", "New York Giants"),
         _q("bk2", "Kansas City Chiefs", "Denver Broncos")]
    m = EV.match_events(q, season=2026, week=2,
                        known_game_ids={"2026-W02-LA-NYG"})
    assert [x.game_id for x in m] == ["2026-W02-LA-NYG"]


def test_unmatched_events_are_returned_for_the_caller_to_notice():
    """A slate that silently prices 6 of 16 games is the shape of a problem
    that goes unnoticed for weeks."""
    q = [_q("bk1", "Los Angeles Rams", "New York Giants"),
         _q("bk2", "Kansas City Chiefs", "Denver Broncos")]
    m = EV.match_events(q, season=2026, week=2,
                        known_game_ids={"2026-W02-LA-NYG"})
    assert EV.unmatched(q, m) == ["bk2"]


def test_duplicate_quotes_for_one_event_collapse():
    q = [_q("bk1", "Los Angeles Rams", "New York Giants", outcome="H"),
         _q("bk1", "Los Angeles Rams", "New York Giants", outcome="A")]
    assert len(EV.match_events(q, season=2026, week=2)) == 1


# ------------------------------------------------------------- runner ----

def _run(*args):
    return subprocess.run([sys.executable, str(RUNNER), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_the_runner_is_dry_by_default():
    src = RUNNER.read_text()
    assert '"--commit"' in src
    assert "DRY BY DEFAULT" in src


def test_the_runner_sizes_from_the_market_grade_not_the_attempt_log():
    """Superseded 2026-09-22 (ADR 0024). This test used to require the
    attempt-log robust weight. That weight is graded model against model and
    was staking football at ~0.91 while the market-relative grade measures no
    edge over the closing spread. The runner now reads the market grade."""
    src = RUNNER.read_text()
    assert "staking_weight(args.league)" in src
    assert "robust_weight()" not in src and "log.weight()" not in src


def test_the_pooled_flag_is_refused_with_its_reason():
    r = _run("--league", "nfl", "--week", "2", "--bankroll", "100000", "--pooled")
    assert r.returncode == 2
    assert "no longer sizes bets" in r.stderr and "0024" in r.stderr


def test_the_runner_reports_its_evidence_before_sizing_anything():
    r = _run("--league", "nfl", "--week", "2", "--bankroll", "100000")
    assert "market grade:" in r.stdout
    assert "attempt log (model vs model; does not size bets)" in r.stdout
    assert r.stdout.index("market grade:") < r.stdout.index("NFL 2026 week 2")


def test_the_runner_exits_cleanly_when_there_is_no_odds_snapshot():
    """Thursday's state. It must say what to do, not traceback."""
    r = _run("--league", "nfl", "--week", "2", "--bankroll", "100000")
    assert r.returncode == 2
    assert "No odds snapshot" in r.stdout
    assert "Traceback" not in r.stderr


def test_the_runner_requires_a_bankroll():
    r = _run("--league", "nfl", "--week", "2")
    assert r.returncode != 0
    assert "bankroll" in (r.stderr + r.stdout).lower()
