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
        assert e["checks"]["ties"].get("ok"), f"{rel} fails the tie check"
        assert e["excluded"].get("not_completed", 0) >= 1, (
            f"{rel} no longer excludes an unplayed game, so either the data "
            "changed or the completion filter stopped working"
        )


def test_the_known_cfb_faults_are_still_reported(report) -> None:
    """ADR 0019's faults are NOT repaired in the data, deliberately.

    Rewriting rows by hand would be inventing results. They are refused at
    the point of use, and this asserts the audit still sees them -- a silent
    pass here would mean the check stopped working, not that the data healed.
    """
    wf = report["frames"]["model/cfb_full_walk_forward_cache.csv"]
    assert wf["checks"]["ties"]["ok"] is False

    sched = report["frames"]["model/cfb_schedule_cache.csv"]
    assert sched["checks"]["ties"]["ok"] is False
    assert sched["checks"]["impossible_scores"]["ok"] is False, (
        "the schedule cache no longer reports a score of one; football cannot "
        "produce one and two rows in 2021 do"
    )


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
    assert wf["checks"]["impossible_scores"].get("ok") is False, (
        "the score check no longer runs on the constants cache"
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


def test_the_audit_exits_non_zero_when_something_fails() -> None:
    """It is meant to gate a pipeline, so the exit code has to mean something.

    Checked by running it, because an exit code nobody has observed is a
    claim -- and this repository pushed a red commit once by reading grep's
    exit status instead of the check's.
    """
    r = subprocess.run([sys.executable, "model/audit_data_integrity.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1, (
        "the audit exited 0 while the known CFB faults are still present"
    )
    assert "FAIL" in r.stdout
