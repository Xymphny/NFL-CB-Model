"""Training inputs live in the repository, not in a temp directory.

WHY THIS EXISTS
Two fits shipped today reading their inputs from /tmp/fitdata. They were
correct, they were graded, and they were reproducible until the next reboot.
That is not reproducible: the artifacts would have kept asserting a t
statistic that no clone -- and, after a restart, not even this machine --
could re-derive.

It is the same wound this repository already has once, in eight constants
citing a grid search whose output exists in no committed file. The difference
between "the evidence is in /tmp" and "there is no evidence" is a reboot.

So: no fit or ingest script may read training data from a temp directory, and
the parquet files those scripts name must be tracked by git. Both halves
matter -- a path inside the repo that git ignores fails the same way.

THE CORRUPT FILES ARE RETAINED ON PURPOSE
sportsdataverse's NHL 2021-2023 files carry a constant score on every row and
are kept anyway. The guard that refuses them is evidence, and evidence a
reader cannot re-run is a claim. A test asserting the guard fires needs the
input that makes it fire.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

#: Directories a data path must never start with. /var/folders is macOS's
#: real $TMPDIR, which is where a naive tempfile default lands.
TEMP_ROOTS = ("/tmp/", "/var/tmp/", "/var/folders/", "/private/tmp/")

#: Temp paths that are NOT training inputs, each with the reason it is
#: allowed. A stale entry fails: if the line moves or the path changes, the
#: test says so rather than silently widening.
#:
#: The distinction is reproducibility WITHOUT the file. A cache re-downloads.
#: A self-test fixture is written by the test that reads it. Neither is the
#: only copy of anything. /tmp/fitdata was.
ALLOWED = {
    ("analyze_historical_clv.py", "/tmp/clv_analysis_test"):
        "self-test scratch, written and removed by the demo block below it",
    ("clv_tracking.py", "/tmp/clv_test"):
        "self-test scratch, written then shutil.rmtree'd in the same block",
    ("player_projection.py", "/tmp/nfl_pbp_cache"):
        "cache of nflverse play-by-play, re-downloadable and env-overridable "
        "via RZ_CACHE_DIR; nothing graded names it as an input",
}

_SCRIPTS = sorted(
    p for p in (ROOT / "model").rglob("*.py") if "__pycache__" not in str(p)
)


def _tracked() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.splitlines())


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_no_unexplained_temp_path(script: Path) -> None:
    """A data path under /tmp is a fit that expires, unless it is explained."""
    for lineno, line in enumerate(script.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        for root in TEMP_ROOTS:
            if root not in code:
                continue
            match = [k for k in ALLOWED if k[0] == script.name and k[1] in code]
            assert match, (
                f"{script.relative_to(ROOT)}:{lineno} names a path under {root} "
                "-- training inputs belong in data/raw/ and tracked by git, or "
                "the grade they produce outlives the evidence for it. If this "
                "one is a cache or test scratch, add it to ALLOWED with the "
                "reason."
            )


def test_allowlist_has_no_stale_entries() -> None:
    """An allowance that no longer describes real code is a lie in a guard."""
    bodies = {p.name: p.read_text() for p in _SCRIPTS}
    for (name, path), reason in ALLOWED.items():
        assert name in bodies, f"ALLOWED names a script that is gone: {name}"
        assert path in bodies[name], (
            f"ALLOWED[{name}] still allows {path!r} ({reason}) but the file no "
            "longer contains it -- remove the entry"
        )


def test_retained_parquet_is_tracked_by_git() -> None:
    """Present on this disk is not the same as present in the repository."""
    raw = ROOT / "data" / "raw"
    assert raw.is_dir(), "data/raw/ is missing -- training inputs are not retained"
    found = sorted(p.relative_to(ROOT).as_posix() for p in raw.rglob("*.parquet"))
    assert found, "data/raw/ holds no parquet -- nothing is retained"

    tracked = _tracked()
    untracked = [p for p in found if p not in tracked]
    assert not untracked, (
        "retained training data that git does not track: "
        + ", ".join(untracked)
        + " -- a clone cannot reproduce any fit that reads these"
    )


def test_every_fitted_artifact_names_retained_inputs() -> None:
    """A graded artifact must say which retained files produced it.

    Without this the retention is decorative: the files are there, and nothing
    connects them to the number that claims to come from them.
    """
    import json

    for name in ("nhl_fitted.json", "nba_fitted.json"):
        art = json.loads((ROOT / "data" / name).read_text())
        inputs = art.get("inputs")
        assert inputs, f"{name} does not record its input files"
        for rel in inputs:
            assert (ROOT / rel).exists(), f"{name} names a missing input: {rel}"
            assert rel in _tracked(), f"{name} names an untracked input: {rel}"
