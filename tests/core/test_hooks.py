"""The pre-commit hook installer.

WHY THIS EXISTS
scripts/check.sh documented a pre-commit hook from the day it was written,
and nobody installed it. On 2026-09-21 a commit went out red because the
command gating it piped check.sh into grep, and grep exits 0 when it FINDS
lines -- including the line reporting the failure. The gate was reporting
success on evidence of failure.

That is a discipline failure with a structural fix available, which is the
trade this project keeps making in the other direction.

WHAT IS AND IS NOT TESTED HERE
.git/hooks/ is not tracked, so this cannot assert a hook is installed in
anyone's clone -- it would fail for everybody who has not run the installer,
which trains people to ignore it. What it CAN hold is the installer: that it
exists, that it keys on the exit code rather than on output, that it
distinguishes "not verified" from "failed", and that it stays bypassable.
"""

import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

INSTALLER = ROOT / "scripts" / "install_hooks.sh"


def test_the_installer_exists_and_is_executable():
    assert INSTALLER.exists()
    assert INSTALLER.stat().st_mode & stat.S_IXUSR


def test_the_hook_keys_on_the_exit_code_not_on_output():
    """The whole point. A check whose success condition is 'some text
    appeared' is not a check -- that is the bug this hook exists to prevent,
    and writing the hook the same way would be absurd."""
    src = INSTALLER.read_text()
    assert "rc=$?" in src
    assert "$rc -ne 0" in src
    assert not re.search(r"check\.sh.*\|\s*grep", src), (
        "the hook pipes check.sh into grep to decide pass/fail, which is "
        "exactly the mistake it was written to stop"
    )


def test_the_hook_treats_a_missing_environment_as_not_verified():
    """check.sh exits 2 when dependencies are absent. That is different from
    a failure and must not block a commit, or the hook becomes the first
    thing a new clone disables."""
    src = INSTALLER.read_text()
    assert "$rc -eq 2" in src
    assert "NOT the same as a failure" in src


def test_the_hook_stays_bypassable():
    """A hook that cannot be bypassed gets uninstalled the first time it is
    wrong, which is worse than one that can."""
    assert "--no-verify" in INSTALLER.read_text()


def test_the_hook_prefers_the_project_venv():
    """System python on this machine had no dependencies at all. Without the
    venv the hook would exit 2 forever and verify nothing."""
    src = INSTALLER.read_text()
    assert ".venv/bin/python" in src and ".venv/bin:$PATH" in src


def test_installing_is_idempotent(tmp_path):
    """Running it twice must not append a second copy of the hook body."""
    work = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    (work / "scripts").mkdir()
    (work / "scripts" / "install_hooks.sh").write_text(INSTALLER.read_text())
    (work / "scripts" / "check.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    for _ in range(2):
        subprocess.run(["bash", "scripts/install_hooks.sh"], cwd=work,
                       capture_output=True, check=True)
    body = (work / ".git" / "hooks" / "pre-commit").read_text()
    assert body.count("pre-commit: all checks passed") == 1


def test_the_readme_tells_you_to_install_it():
    """An installer nobody knows about is the situation that produced the red
    commit in the first place."""
    text = (ROOT / "README.md").read_text()
    assert "install_hooks.sh" in text
