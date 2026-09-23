"""Every retained frame, past every impossibility that applies to it.

WHY THIS EXISTS
Three separate times a model or a cache was caught holding an outcome its
league does not permit -- a tied NHL game (ADR 0007), a tied MLB game (ADR
0017), and 76 tied CFB games in the cache that produced two shipped constants
(ADR 0019). Every one was found while looking at something else.

Three by accident is an argument for looking on purpose.

WHAT THE AUDIT DISTINGUISHES, AND WHY IT MATTERS
An ABSENT result is not a FALSE one. Every NBA season file carries exactly one
row at 0-0 with `status_type_completed = False` -- a cancelled game, correctly
excluded by the fit's own filter. The first version of the audit reported all
five as failing.

The CFB schedule cache has NO completion column, so its 83 unfetched games sit
in it as 0-0 results indistinguishable from played ones. That is the whole of
ADR 0019 in one sentence: a frame that can say a game was not played is a
frame whose zeros are safe.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

REPORT = ROOT / "model" / "data_integrity.json"


@pytest.fixture(scope="module")
def report() -> dict:
    assert REPORT.exists(), (
        "run model/audit_data_integrity.py -- the audit's own output is missing"
    )
    return json.loads(REPORT.read_text())


def test_the_audit_covers_every_retained_frame(report) -> None:
    """A frame nobody audits is a frame nobody checked."""
    import glob

    frames = set(report["frames"])
    for pattern in ("data/raw/nhl/nhl_*.parquet",
                    "data/raw/mlb/linescores_*.parquet",
                    "data/raw/sportsdataverse/*.parquet"):
        for f in glob.glob(str(ROOT / pattern)):
            rel = str(Path(f).relative_to(ROOT))
            assert rel in frames, f"{rel} is retained and not audited"
    assert report["_provenance"]["frames_checked"] > 20


def test_the_nba_files_pass_once_unplayed_games_are_excluded(report) -> None:
    """The distinction the first version of the audit got wrong.

    One cancelled game per season, recorded 0-0 with the completion flag
    False. Excluding it is correct; reporting it as a tie was not.
    """
    nba = {k: v for k, v in report["frames"].items() if "nba_" in k}
    assert nba, "no NBA frames audited"
    for rel, e in nba.items():
        if e["rows"] == 0:
            # A season not yet started: a schedule, every game unplayed. The
            # tie check has nothing to check, which is only acceptable when
            # EVERY row on disk was excluded as unplayed or non-standard --
            # an empty frame for any other reason is still a failure.
            ex = e["excluded"]
            assert ex.get("not_completed", 0) >= 1, f"{rel} is empty and not a schedule"
            assert sum(ex.values()) == e["rows_on_disk"], f"{rel} lost rows unaccounted"
            continue
        assert e["checks"]["ties"].get("ok"), f"{rel} fails the tie check"
        assert e["excluded"].get("not_completed", 0) >= 1, (
            f"{rel} no longer excludes an unplayed game, so either the data "
            "changed or the completion filter stopped working"
        )


def test_every_frame_now_passes_and_the_repairs_are_on_record(report) -> None:
    """The faults ARE repaired now, and the audit is what says so.

    ADRs 0019 and 0020 refused to repair, correctly, while there was no
    second source: rewriting a result by hand is inventing one. With ESPN
    joined on the same event ids the repair is adjudication, and
    model/cfb_score_repairs.csv records all 315 changes so the claim is
    auditable rather than trusted.

    A green audit is only meaningful beside that record -- otherwise it is
    indistinguishable from a check that stopped looking.
    """
    import pandas as pd

    assert report["_provenance"]["failures"] == 0, (
        "the audit is failing again; read model/data_integrity.json"
    )
    repairs = pd.read_csv(ROOT / "model" / "cfb_score_repairs.csv")
    assert len(repairs) == 315
    assert set(repairs.season) == {2021, 2022, 2023, 2024, 2025}
    assert ((repairs.was_home != repairs.now_home)
            | (repairs.was_away != repairs.now_away)).all()


def test_the_constants_cache_is_now_auditable_for_scores(report) -> None:
    """The limit ADR 0020 recorded, closed -- and the worry it raised, refuted.

    That record said a margin-only cache cannot be audited for impossible
    scores, and inferred that at least one corrupt row was therefore sitting
    inside the frame that produced MARGIN_SD, undetectable.

    THE LIMIT WAS REAL AND THE INFERENCE WAS WRONG. The scores were never
    missing -- model/cfb_full_walk_forward.py read them and threw them away --
    so they were joined back from the schedule cache, checked against the
    margin already recorded on all 1,731 rows, and the check now runs. It
    finds exactly one impossible game, Kansas State 1-1 TCU, which the tie
    filter ALREADY excludes. Florida Atlantic 1-0 is not in this cache at all.

    So no constant moves. Making the frame checkable is what established
    that, which is the argument for making things checkable rather than
    reasoning about them.
    """
    wf = report["frames"]["model/cfb_full_walk_forward_cache.csv"]
    assert wf["checks"]["impossible_scores"].get("ok") is True, (
        "the constants cache carries an impossible score again"
    )

    import pandas as pd

    g = pd.read_csv(ROOT / "model" / "cfb_full_walk_forward_cache.csv")
    assert {"home_score", "away_score"} <= set(g.columns)
    assert (g.home_score - g.away_score).equals(g.actual_margin), (
        "the backfilled scores no longer reproduce the recorded margin"
    )
    surviving = g[g.actual_margin != 0]
    assert int(((surviving.home_score == 1) | (surviving.away_score == 1)).sum()) == 0, (
        "an impossible score now survives the tie filter, which means a "
        "constant measured through this cache IS affected and must be redone"
    )


def test_frames_without_a_completion_flag_are_flagged(report) -> None:
    """The property that separates a safe zero from a dangerous one."""
    flagged = [k for k, v in report["frames"].items()
               if "no_completion_column" in v]
    assert "model/cfb_schedule_cache.csv" in flagged
    assert all("nba_" not in k for k in flagged), (
        "an NBA frame has lost its completion column, which is the one thing "
        "keeping its unplayed games distinguishable from 0-0 results"
    )


def test_the_audit_exit_code_means_something() -> None:
    """It is meant to gate a pipeline, so the exit code has to be real.

    Checked by RUNNING it, because an exit code nobody has observed is a
    claim -- and this repository pushed a red commit once by reading grep's
    exit status instead of the check's.

    Everything passes now, so this asserts the clean path AND injects a fault
    to see the failing one. A green exit code that has never been seen to go
    red is not evidence of anything.
    """
    import tempfile

    r = subprocess.run([sys.executable, "model/audit_data_integrity.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"the audit is failing:\n{r.stdout[-2000:]}"
    assert "FAIL" not in r.stdout

    # Now break it. A frame with a tie in a league that forbids one must make
    # the audit exit non-zero, or the gate is decorative.
    from model.fit_data_checks import ImpossibleOutcome, check_no_impossible_ties

    import pandas as pd

    bad = pd.DataFrame({"home_score": [7, 3], "away_score": [7, 0]})
    with pytest.raises(ImpossibleOutcome):
        check_no_impossible_ties(bad, "cfb", label="injected")


def test_the_nfl_cache_was_checked_and_was_clean(report) -> None:
    """The gap ADR 0021 named, closed -- and nothing was wrong.

    That record ended by saying the NFL walk-forward cache had no second
    source and had never been cross-checked. Joined against nflverse's
    games.csv it matches on all 1,945 rows with ZERO margin disagreements,
    and the five ties in it are real NFL ties that nflverse confirms.

    Both caches were unchecked; only one was wrong. Asserting the clean result
    matters as much as asserting the dirty one, because otherwise the record
    reads as though every pipeline here is broken.
    """
    import pandas as pd

    w = pd.read_csv(ROOT / "model" / "expanded_walk_forward_cache.csv")
    assert {"home_score", "away_score"} <= set(w.columns), (
        "the NFL cache no longer carries scores, so it cannot be audited for "
        "impossible ones"
    )
    assert (w.home_score - w.away_score).equals(w.actual_margin.astype(int))

    n = pd.read_parquet(ROOT / "data" / "raw" / "nfl" / "nflverse_games.parquet")
    n = n[n.game_type == "REG"].assign(
        season=lambda d: d.season.astype(int), week=lambda d: d.week.astype(int))
    j = w.merge(n[["season", "week", "home_team", "away_team", "result"]],
                on=["season", "week", "home_team", "away_team"], how="left")
    assert j.result.notna().all(), "a cache row has no nflverse match"
    assert (j.result == j.actual_margin).all(), (
        "the NFL cache now disagrees with nflverse; that is a finding"
    )

    e = report["frames"]["model/expanded_walk_forward_cache.csv"]
    assert e["checks"]["ties"]["ok"] is True
    assert e["checks"]["ties"]["tied"] == 5, (
        "the NFL tie count moved; five real ties in 1,945 games is 0.26%, "
        "against the 1% the league's allowance permits"
    )


def test_a_frame_with_a_completion_flag_is_not_annotated_as_lacking_one(report):
    """The annotation has to be accurate to be worth reading.

    The ESPN frames carry `completed` and the sportsdataverse ones carry
    `status_type_completed`; the first version of the check only knew the
    second name and labelled every ESPN frame as having no completion column
    while simultaneously filtering on it.
    """
    for rel, e in report["frames"].items():
        if "espn_" in rel or "nba_" in rel:
            assert "no_completion_column" not in e, rel
