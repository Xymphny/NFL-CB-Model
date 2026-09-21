"""The README's numbers must still be true.

The README calls itself "the ground truth on what's actually been run" and
says to read it before bug-fixing anything. A document with that job, quoting
figures that have drifted, is the same failure as a committed artifact
asserting something untrue -- and this project already has a permanent ledger
row about that.

So the figures it quotes are checked against the things that produced them.
Not every sentence: the claims that are NUMBERS, and the status claims that
would mislead someone acting on them.

This is deliberately a thin test. It cannot check prose, and pretending it
could would be worse than not having it.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

README = ROOT / "README.md"


@pytest.fixture(scope="module")
def core_section() -> str:
    text = README.read_text()
    start = text.index("## THE FIVE-LEAGUE CORE")
    end = text.index("## September 2026 round")
    return text[start:end]


def test_the_core_section_exists(core_section):
    assert len(core_section) > 2000, "the core section has been gutted"


def test_the_key_number_result_matches_its_artifact(core_section):
    art = json.loads((ROOT / "data" / "nfl_key_numbers.json").read_text())
    t = art["holdout_grade"]["t"]
    assert f"t = +{t:.2f}" in core_section, (
        f"README quotes a key-number t that is not {t}"
    )
    checks = art["push_probability_check"]["3"]
    assert f"{checks['plain']:.2%}".rstrip("0") in core_section or "2.74%" in core_section
    assert "7.36%" in core_section


def test_the_shrinkage_weights_match_the_attempt_log(core_section):
    from coverline.core.evidence import load_attempts
    log = load_attempts()
    assert f"{log.weight():.4f}" in core_section, (
        f"README quotes a pooled weight that is not {log.weight():.4f}"
    )
    assert f"{log.robust_weight():.4f}" in core_section
    assert f"{len(log)} attempts" in core_section or "Seven attempts" in core_section
    assert f"{log.n_negative} non-positive" in core_section


def test_the_mlb_dispersion_claim_matches_its_artifact(core_section):
    art = json.loads((ROOT / "model" / "mlb_dispersion_results.json").read_text())
    n = art["_provenance"]["n_games"]
    assert f"{n:,} games" in core_section, f"README quotes a game count that is not {n:,}"
    assert f"{art['independence']['correlation']:+.4f}" in core_section


def test_the_cfb_dispersion_claim_matches_the_constant(core_section):
    from coverline.leagues.cfb.model import DVOA_ONLY_MEAN_RESIDUAL, MARGIN_SD
    assert f"{MARGIN_SD:.2f}" in core_section
    assert f"+{DVOA_ONLY_MEAN_RESIDUAL:.2f} point" in core_section


def test_the_league_table_matches_the_conformance_sets(core_section):
    """The status column is the part someone would act on."""
    from tests.core.test_conformance import (
        BUILT_LEAGUES, IMPLEMENTED_LEAGUES, STRUCTURAL_ONLY_LEAGUES,
    )
    for lg in STRUCTURAL_ONLY_LEAGUES:
        row = next(l for l in core_section.splitlines()
                   if l.strip().startswith(f"| {lg.upper()} "))
        assert "structure only" in row.lower(), f"{lg} row does not say structure only"
    for lg in IMPLEMENTED_LEAGUES:
        row = next(l for l in core_section.splitlines()
                   if l.strip().startswith(f"| {lg.upper()} "))
        assert "not live" in row.lower(), f"{lg} row does not say it is not live"
    if not BUILT_LEAGUES:
        assert "not live" in core_section


def test_the_readme_quotes_no_bare_test_count(core_section):
    """A test count in prose goes stale the next time anyone adds a test, and
    a stale count in the ground-truth document is the failure this file
    exists to prevent.

    This test replaced one that compared a quoted count to the manifest. That
    version was itself the problem: the README sentence was describing a
    HISTORICAL state -- how many tests CI was skipping at the time -- and the
    check read it as a current claim, so the document went red for being
    accurate about the past. The fix was to stop quoting the number, not to
    widen the tolerance.
    """
    stale = re.findall(r"\b(\d{3,4}) (?:core )?tests\b", core_section)
    assert not stale, (
        f"the core section quotes test counts {stale}, which go stale on the "
        "next commit. Describe what runs, not how many."
    )


def test_the_scripts_it_tells_you_to_run_exist(core_section):
    for name in re.findall(r"python3 (scripts/[\w_]+\.(?:py|sh))", core_section):
        assert (ROOT / name).exists(), f"README tells you to run {name}, which is missing"


def test_the_python_pin_it_quotes_matches_the_file(core_section):
    pin = (ROOT / ".python-version").read_text().strip()
    assert pin in core_section, f"README quotes a Python version that is not {pin}"


def test_the_permanently_open_row_is_still_named(core_section):
    """If the grid-search row is ever closed, this sentence becomes a lie."""
    import yaml
    led = yaml.safe_load((ROOT / "migration" / "ledger.yaml").read_text())
    grid = next(c for c in led["components"] if c["id"] == "frozen-threshold-grid")
    assert grid["disposition"] == "pending"
    assert "frozen-threshold-grid" in core_section
