"""CI must run every suite scripts/check.sh runs.

WHY THIS EXISTS
Until 2026-09-21 the CI workflow ran two suites while scripts/check.sh ran
four. The two it skipped were the core suite -- 223 tests covering the
five-league core, pricing, staking, and every rot defence in the project --
and the test-manifest check. So the machinery built to catch things breaking
unnoticed was itself only checked when a human remembered to run it locally.

That is the failure the workflow's own header describes, one level up: "Every
check that day happened because someone remembered to run it, and on one
occasion nobody did."

The gap did not open deliberately. check.sh grew two new suites and the
workflow was not updated, because nothing connected them. This test is that
connection. It is not about the current four suites; it is about the NEXT one
someone adds to check.sh and forgets to add here.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CHECK_SH = ROOT / "scripts" / "check.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "checks.yml"


def _check_sh_commands() -> list[str]:
    """The commands check.sh runs, as written, minus the frontend build.

    The frontend build lives in a separate CI job with a different toolchain,
    so it is compared separately.
    """
    out = []
    for line in CHECK_SH.read_text().splitlines():
        m = re.match(r'\s*run\s+"[^"]+"\s+(.*\S)\s*$', line)
        if not m:
            continue
        cmd = m.group(1).strip()
        if cmd.startswith("bash -c") or "npm run build" in cmd:
            continue
        out.append(cmd)
    return out


def _workflow_runs() -> list[str]:
    """Every `run:` command in the workflow, flattened."""
    out = []
    for line in WORKFLOW.read_text().splitlines():
        m = re.match(r"\s*-?\s*run:\s*(.*\S)\s*$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def test_both_files_exist():
    assert CHECK_SH.exists(), "scripts/check.sh is missing"
    assert WORKFLOW.exists(), (
        "the CI workflow is missing. If it was deliberately removed, this "
        "test should be removed in the same commit and the reason recorded."
    )


def test_check_sh_actually_runs_something():
    """Guards the guard: if the parser silently matched nothing, every
    comparison below would pass vacuously."""
    cmds = _check_sh_commands()
    assert len(cmds) >= 4, (
        f"parsed only {len(cmds)} commands out of check.sh: {cmds}. Either "
        "suites were removed, or the `run \"label\" cmd` shape changed and "
        "this parser needs updating."
    )


def test_ci_runs_every_python_suite_that_check_sh_runs():
    """The whole point. A suite added to check.sh and not to CI fails here."""
    local = _check_sh_commands()
    ci = _workflow_runs()
    missing = [c for c in local if c not in ci]
    assert not missing, (
        f"scripts/check.sh runs these and CI does not: {missing}.\n"
        "Add them to .github/workflows/checks.yml. A check that only runs "
        "when someone remembers is the failure this project already had once."
    )


def test_the_core_suite_and_manifest_check_are_specifically_covered():
    """Named explicitly, because these two were the ones actually missing and
    a generic comparison would not say so if the parser regressed."""
    ci = " ".join(_workflow_runs())
    assert "pytest tests/core" in ci, "CI does not run the core suite"
    assert "test_manifest.py --check" in ci, "CI does not run the manifest check"


def test_the_frontend_build_is_covered_by_the_other_job():
    """check.sh runs it conditionally; CI has a dedicated job for it."""
    wf = WORKFLOW.read_text()
    assert "npm run build" in wf
    assert "working-directory: frontend" in wf


def test_ci_installs_the_declared_environment():
    """Installing a hand-picked subset is how four guard tests failed on the
    first CI run. It also means requirements.txt is never checked installable."""
    assert "pip install -r requirements.txt" in WORKFLOW.read_text()


def test_the_workflow_pins_a_python_version():
    """An unpinned runtime is a silent-divergence hazard: CI, Render and a
    developer's laptop can all differ.

    Requires a NUMERIC minor version. The first version of this test accepted
    anything starting with "3.", which passed on `3.x` -- a floating range,
    not a pin, and exactly the thing being guarded against. Found by trying it.
    """
    text = WORKFLOW.read_text()
    m = re.search(r"python-version:\s*['\"]?(\d+\.\d+(?:\.\d+)?)['\"]?\s*$",
                  text, re.M)
    assert m, (
        "the workflow does not pin an exact python-version. A floating "
        "specifier like '3.x' or '3' is not a pin: CI would silently move "
        "interpreter as new releases land."
    )
    major, minor = m.group(1).split(".")[:2]
    assert major == "3" and minor.isdigit(), m.group(1)


def test_the_repo_pins_the_same_python_version_everywhere():
    """CI pins 3.11; .python-version must agree, or local and deploy drift
    from what is actually tested."""
    pin = ROOT / ".python-version"
    if not pin.exists():
        pytest.skip(".python-version not present yet")
    ci = re.search(r"python-version:\s*'?\"?([\d.]+)", WORKFLOW.read_text()).group(1)
    local = pin.read_text().strip()
    assert local.startswith(ci), (
        f".python-version says {local}, CI pins {ci}. Tests passing locally "
        "would not mean tests passing in CI."
    )
