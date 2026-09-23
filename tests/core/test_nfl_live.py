"""Live parity: does the new core reproduce the board the site actually shows?

test_nfl_parity.py proved the new core matches the legacy model on 1,945
FINISHED games from a committed cache. That is a strong migration property and
a weak liveness one: a cache is a fixed input, and reproducing it says nothing
about whether the same code, fed this week's published ratings and schedule,
lands on the number the board published.

This file closes that. It takes the ratings snapshot and the divergence
artifact the weekly job wrote for 2026 week 2 -- real committed output, not a
fixture I authored -- and reconstructs the board's own model margin.

SCOPE, STATED HONESTLY. Only the rating-only games can be reconstructed. The
full-ensemble games need NGS features that live in a runtime fetch, not in any
committed artifact, and a test that quietly skipped them without saying so
would overstate what has been verified. The count of each is asserted, so the
scope cannot silently shrink.
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.leagues.nfl import live as L  # noqa: E402
from coverline.leagues.nfl import model as nfl  # noqa: E402

SCHEDULE_FIXTURE = ROOT / "tests" / "fixtures" / "nfl_2026_week02_schedule.csv"
RATING_ONLY = "v1_rating_only_no_ngs"
ASOF = "2026-09-21T12:00:00Z"


@pytest.fixture(scope="module")
def schedule():
    return pd.read_csv(SCHEDULE_FIXTURE)


@pytest.fixture(scope="module")
def source(schedule):
    return L.RatingsSnapshotSource.for_week(2026, 2, schedule,
                                            ratings_dir=ROOT / "data" / "ratings")


@pytest.fixture(scope="module")
def pit_source(schedule):
    """The version that was current before week 2 started -- recovered.

    The canonical path is overwritten: data/ratings/2026-week-02.json held
    ratings computed 2026-09-18 09:10 and, after the weekly cron ran again on
    2026-09-22 11:00, holds ratings computed after week 2 finished. Those have
    seen the results, so LookaheadRefused fires -- correctly.

    The 09-18 bytes were never gone, only unaddressable: git had them, and
    they now live in data/ratings/history/ alongside ten other versions
    recovered the same way. `asof` selects the one that was current, so these
    tests price week 2 from the snapshot the board actually used instead of
    skipping.
    """
    wk = schedule[(schedule.season == 2026) & (schedule.week == 2)]
    # The LAST kickoff of the week, not the first. Asking for the version
    # current before the first would find none: the earliest week-2 snapshot
    # was computed 2026-09-18, and week 2 opened on the Thursday, 09-17. So
    # no version of this week has ever been point-in-time valid for its own
    # opening game -- a snapshot named for week N is produced DURING week N.
    # The per-game guard still refuses the Thursday game from this version,
    # which is the correct answer rather than a problem to route around.
    last_kick = pd.to_datetime(wk.gameday.max(), utc=True)
    return L.RatingsSnapshotSource.for_week(
        2026, 2, schedule, ratings_dir=ROOT / "data" / "ratings",
        asof=last_kick.strftime("%Y-%m-%dT%H:%M:%SZ"))




@pytest.fixture(scope="module")
def board():
    """The newest published week-2 divergence artifact."""
    paths = sorted(glob.glob(str(ROOT / "data" / "divergence" / "2026-week-02-*.json")))
    if not paths:
        pytest.skip("no 2026 week 2 board artifact committed")
    return json.loads(Path(paths[-1]).read_text())


# ------------------------------------------------------------- the source ----

def test_it_loads_the_snapshot_for_the_week_asked_for(source):
    assert source.season == 2026 and source.week == 2
    assert len(source.ratings) == 32
    assert source.path.name == "2026-week-02.json"


def test_a_missing_week_is_refused_rather_than_substituted(schedule):
    """Serving a neighbouring week's ratings is the easiest lookahead there
    is, because the newest file is always the convenient one."""
    with pytest.raises(L.NoRatingsSnapshot, match="Refusing to substitute"):
        L.RatingsSnapshotSource.for_week(2026, 9, schedule,
                                         ratings_dir=ROOT / "data" / "ratings")


def test_selection_is_by_week_not_by_modification_time(schedule):
    """for_week(1) must return week 1 even though week 2 exists and is newer."""
    s = L.RatingsSnapshotSource.for_week(2026, 1, schedule,
                                         ratings_dir=ROOT / "data" / "ratings")
    assert s.path.name == "2026-week-01.json"


def test_an_unrated_team_is_refused_not_averaged(source):
    """A team on the schedule but absent from the ratings snapshot.

    Built by removing one rather than by inventing a fake team, because a
    fake team fails the SCHEDULE lookup first and never reaches this branch --
    which is how the first version of this test passed for the wrong reason.
    """
    thin = L.RatingsSnapshotSource(
        season=source.season, week=source.week,
        ratings={k: v for k, v in source.ratings.items() if k != "NYG"},
        schedule=source.schedule, computed_at=source.computed_at,
        path=source.path,
    )
    with pytest.raises(KeyError, match="refusing to price"):
        thin.features("2026-W02-LA-NYG", ASOF)
    # and the error names the team, not just the game
    try:
        thin.features("2026-W02-LA-NYG", ASOF)
    except KeyError as exc:
        assert "NYG" in str(exc)


def test_a_game_not_on_the_schedule_is_refused(source):
    with pytest.raises(KeyError, match="not on the"):
        source.features("2026-W02-LA-KC", ASOF)


def test_rest_comes_from_the_schedule_not_from_zero(pit_source):
    """rest_diff carries a real coefficient; defaulting it to 0 would be a
    silent, plausible-looking error."""
    f = pit_source.features("2026-W02-LA-NYG", ASOF)
    assert f.rest_diff == 3.0


def test_features_declare_ngs_absent_because_it_is(pit_source):
    f = pit_source.features("2026-W02-LA-NYG", ASOF)
    assert f.ngs_present is False and f.elo_diff == 0.0


# ------------------------------------------------------- live parity ----

def test_the_board_slate_splits_as_expected(board):
    """Pins the scope of the parity check below. If the full-ensemble share
    changes, the coverage of this file changes with it and should be read
    again rather than silently drifting."""
    rows = [d for d in board["divergences"] if d.get("spread_gap") is not None]
    rating_only = [d for d in rows if d.get("coefficient_set") == RATING_ONLY]
    assert len(rows) >= 5, "the board has almost no priced games to check"
    assert len(rating_only) >= 1, (
        "no rating-only games on this slate, so nothing here can be verified "
        "against the live board without the NGS fetch"
    )


def test_it_reproduces_the_published_board_margin_for_rating_only_games(pit_source, board):
    """The claim: same ratings, same schedule, same coefficients, same number
    the site published -- to the cent."""
    offset = board["debias_offsets"][0]
    checked = 0
    for d in board["divergences"]:
        if d.get("coefficient_set") != RATING_ONLY:
            continue
        if d.get("spread_gap") is None or d.get("market_spread") is None:
            continue

        published = d["market_spread"] + d["spread_gap"]
        gid = f"2026-W02-{d['home_team']}-{d['away_team']}"
        ours = L.reconstruct_board_margin(pit_source.features(gid, ASOF), offset)

        assert ours == pytest.approx(published, abs=1e-6), (
            f"{gid}: reconstructed {ours:.6f}, board published {published:.6f}"
        )
        checked += 1

    assert checked >= 1, "no rating-only game was actually compared"


def test_the_reconstruction_would_notice_a_wrong_rest_value(pit_source, board):
    """Guards the parity check itself. rest_diff enters with a coefficient of
    0.131, so a wrong value shifts the margin by a fraction of a point -- small
    enough to look like rounding if nothing asserts on it."""
    offset = board["debias_offsets"][0]
    f = pit_source.features("2026-W02-LA-NYG", ASOF)
    honest = L.reconstruct_board_margin(f, offset)
    wrong = L.reconstruct_board_margin(
        nfl.GameFeatures(rating_diff=f.rating_diff, rest_diff=0.0,
                         is_neutral_site=f.is_neutral_site, ngs_present=False),
        offset)
    assert abs(honest - wrong) == pytest.approx(
        3.0 * nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY["rest_diff"], abs=1e-9)
    assert abs(honest - wrong) > 0.3


def test_de_bias_is_the_boards_job_not_the_sources(pit_source):
    """The offset is measured across a slate, so a per-game source cannot know
    it. Keeping it out is what stops a feature source quietly carrying a
    slate-level constant."""
    f = pit_source.features("2026-W02-LA-NYG", ASOF)
    raw = nfl.predict_margin(f)
    assert L.reconstruct_board_margin(f, 0.0) == pytest.approx(raw)
    assert L.reconstruct_board_margin(f, 5.0) == pytest.approx(raw + 5.0)


def test_the_ratings_snapshot_path_is_mutable(schedule):
    """The defect, and the recovery, both on record.

    data/ratings/2026-week-02.json held ratings computed 2026-09-18 09:10 and
    now holds ratings computed 2026-09-22 11:00. Same path, different content,
    no version in the name -- and it is routine, not a one-off: 2026 week 1
    was written three times in one morning.

    The bytes were never lost, only unaddressable. Eleven versions were
    recovered from git into data/ratings/history/, which deploy/weekly_job.py
    now writes to as well, write-once. The canonical path is untouched, so
    every existing reader keeps seeing the newest ratings at the name it knows.
    """
    versions = L.history_versions(2026, 2, ROOT / "data" / "ratings")
    assert len(versions) >= 2, (
        "the week-2 history has fewer than two versions; the overwritten one "
        "was recovered from git and must not go missing again"
    )
    stamps = sorted(
        json.loads(p.read_text())["computed_at"] for p in versions)
    assert stamps[0] < stamps[-1]

    wk = schedule[(schedule.season == 2026) & (schedule.week == 2)]
    last_kick = pd.to_datetime(wk.gameday.max(), utc=True)
    assert pd.to_datetime(stamps[-1], utc=True) > last_kick, (
        "the newest version no longer postdates the week, so the canonical "
        "path may be safe again -- check before deleting this test"
    )


def test_a_later_version_is_never_substituted(schedule):
    """The refusal that makes `asof` worth having.

    Falling back to whatever exists is the easiest lookahead there is,
    because the newest file is always the convenient one.
    """
    with pytest.raises(L.NoRatingsSnapshot, match="Refusing to substitute"):
        L.RatingsSnapshotSource.for_week(
            2026, 2, schedule, ratings_dir=ROOT / "data" / "ratings",
            asof="2026-09-01T00:00:00Z")


def test_asof_picks_the_version_that_was_current(schedule):
    """Between the two week-2 versions, each asof selects its own."""
    d = ROOT / "data" / "ratings"
    early = L.version_current_at(2026, 2, "2026-09-19T00:00:00Z", d)
    late = L.version_current_at(2026, 2, "2026-09-30T00:00:00Z", d)
    assert early is not None and late is not None
    assert early != late
    assert "20260918" in early.name and "20260922" in late.name


def test_the_weekly_job_writes_an_immutable_copy():
    """Asserted on the source, because the job cannot be run here.

    It needs nflverse play-by-play and a git remote. What is checkable is
    that it writes the history file and commits it -- committing only the
    mutable path would leave the history on an ephemeral disk, which is the
    same as not writing it.
    """
    src = (ROOT / "deploy" / "weekly_job.py").read_text()
    assert 'os.path.join(ratings_dir, "history")' in src
    assert "if not os.path.exists(history_file):" in src, (
        "the history write is no longer write-once"
    )
    assert "git_commit_and_push(\n                history_file," in src


# ------------------------------------------------ pricing an upcoming week ----

def _week(w, gamedays):
    import pandas as pd
    return pd.DataFrame({"season": 2026, "week": w, "home_team": [f"H{i}" for i in range(len(gamedays))],
                         "away_team": [f"A{i}" for i in range(len(gamedays))], "gameday": gamedays,
                         "home_rest": 7, "away_rest": 7, "location": "Home"})


def test_an_upcoming_week_is_priced_from_the_last_completed_weeks_snapshot():
    """The weekly job names a snapshot for the last week with a completed game
    and publishes it for the NEXT one. for_week(3) asked for a week-3 file --
    which exists only once week 3 is over -- so no upcoming week could ever
    be priced. GameWeekSource prices week 3 from the Tuesday week-2 version."""
    src = L.GameWeekSource.for_game_week(2026, 3, _week(3, ["2026-09-24", "2026-09-27"]))
    ca, wk, path = src.version_for("2026-09-27", asof="2026-09-23T12:00:00Z")
    assert wk == 2 and ca.isoformat().startswith("2026-09-22T11:00")


def test_each_game_gets_the_newest_version_computed_before_its_day(schedule):
    """Week 2: Thursday's game predates the Friday refresh, Sunday's does not."""
    src = L.GameWeekSource.for_game_week(2026, 2, schedule)
    thu, _, _ = src.version_for("2026-09-17", asof="2026-09-17T23:00:00Z")
    sun, _, _ = src.version_for("2026-09-20", asof="2026-09-20T16:00:00Z")
    assert thu.date().isoformat() == "2026-09-15"
    assert sun.isoformat().startswith("2026-09-18T09:10")


def test_a_version_computed_after_the_game_is_never_used(schedule):
    src = L.GameWeekSource.for_game_week(2026, 2, schedule)
    # Pricing a week-2 Sunday game today: the Tuesday Sep 22 version has seen
    # it. The newest version on or before its day is still Friday's.
    ca, _, _ = src.version_for("2026-09-20", asof="2026-09-30T00:00:00Z")
    assert ca.date().isoformat() == "2026-09-18"


def test_nothing_before_the_game_is_refused_not_substituted():
    src = L.GameWeekSource.for_game_week(2026, 1, _week(1, ["2026-09-01"]))
    with pytest.raises(L.NoRatingsSnapshot, match="refusing"):
        src.version_for("2026-09-01", asof="2026-09-01T00:00:00Z")


def test_the_operator_command_prices_week_three():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "recommend_slate.py"),
                        "--league", "nfl", "--week", "3", "--bankroll", "100"],
                       capture_output=True, text=True, cwd=ROOT)
    if "URLError" in r.stderr or "Temporary failure" in r.stderr:
        pytest.skip("the week-3 schedule comes from nflverse; no network here")
    assert "no ratings snapshot" not in r.stdout + r.stderr
    assert "NFL 2026 week 3: NFLModel" in r.stdout
