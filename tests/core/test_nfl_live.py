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
def pit_source(source):
    """The same source, for tests that need it to be POINT-IN-TIME VALID.

    These priced week 2 from data/ratings/2026-week-02.json as published
    2026-09-18 09:10. On 2026-09-22 11:00 the weekly cron rewrote that same
    path with ratings computed after week 2 finished, and LookaheadRefused
    began firing -- correctly, because those ratings have seen the results.

    The guard is working. What is broken is upstream: the snapshot path is
    not immutable, so the file a published board was computed from can be
    replaced under the same name and the board becomes unreproducible. See
    test_the_ratings_snapshot_path_is_mutable.

    Only the tests that actually reconstruct a price skip. The ones that check
    loading, naming and refusal do not depend on the timestamp and still run.
    """
    first_kick = pd.to_datetime(source.schedule.gameday.min(), utc=True)
    computed = pd.to_datetime(source.computed_at, utc=True)
    if computed > first_kick:
        pytest.skip(
            f"2026-week-02.json was recomputed {computed.date()}, after week 2 "
            f"began {first_kick.date()}; this test needs the snapshot the "
            "board actually used and it has been overwritten in place"
        )
    return source


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
    """The defect the skip above is a symptom of, asserted so it is on record.

    data/ratings/2026-week-02.json held ratings computed 2026-09-18 09:10 and
    now holds ratings computed 2026-09-22 11:00. Same path, different content,
    no version in the name. Every board published from the first version cites
    a file that no longer contains what it cited.

    This is the shape of the repository's oldest wound -- eight constants
    citing a grid search whose output exists in no committed file -- arriving
    by a different route: not a file that was never written, but a file that
    was written over.

    The fix is upstream and is not made here, because the weekly job is a live
    cron and changing what it writes is a deployment decision. What is made
    here is the record.
    """
    src = L.RatingsSnapshotSource.for_week(2026, 2, schedule,
                                           ratings_dir=ROOT / "data" / "ratings")
    computed = pd.to_datetime(src.computed_at, utc=True)
    first_kick = pd.to_datetime(src.schedule.gameday.min(), utc=True)
    if computed <= first_kick:
        pytest.skip("the snapshot is point-in-time valid again; the pipeline "
                    "may have been fixed, in which case delete this test")
    assert computed > first_kick, (
        "a snapshot named for week 2 was computed after week 2 started"
    )
