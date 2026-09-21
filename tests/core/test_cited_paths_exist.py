"""Every file a docstring names must exist.

WHY THIS EXISTS
`core/interfaces.py` -- the most widely read module in the repository, and the
one that defines the seam everything else imports -- said this about the
point-in-time contract:

    the two-run guard test in tests/core/test_point_in_time.py is designed to
    catch exactly that

That file did not exist. The contract was cited by name, in the file that
defines it, and enforced by nothing. Anyone reading the docstring would have
concluded the guard was there, because there is no reason to check.

It is the same shape as the artifacts that shipped with no producer and the
constants citing a grid search whose output was never written down: a claim
that reads like evidence because it names a location. The remedy is the same
too -- make the citation checkable.

SCOPE
Concrete file paths with a known extension, mentioned anywhere in the source
of a module under src/, model/ or scripts/. Directory references and glob
patterns are not checked: they describe families, not files, and a family can
legitimately be empty.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

#: Extensions worth checking. A path with one of these is a specific file.
EXTENSIONS = (".py", ".json", ".yaml", ".yml", ".md", ".csv", ".sh", ".parquet")

#: Directories whose modules are scanned.
ROOTS = ("src", "model", "scripts", "tests")

#: Tokens that mark a path as a TEMPLATE rather than a citation. A docstring
#: describing "data/ratings/2026-week-NN.json" is naming a family and a
#: reader knows it; requiring that literal file to exist would be a guard
#: misreading prose.
PLACEHOLDERS = ("NN", "XX", "YYYY", "{", "*")

#: Paths that are named but deliberately absent, each with its reason.
#: A stale entry fails its own test below, so this cannot quietly widen.
ALLOWED_ABSENT: dict[str, str] = {
    "model/mlb_lines_cache.csv":
        "a PRECONDITION, not an output: mlb_backtest.py documents that it "
        "runs only when this exists, and no MLB closing lines have been "
        "captured yet. The citation is correct and the file is correctly "
        "missing.",
}

_PATH = re.compile(
    r"(?<![\w./-])((?:src|model|scripts|tests|data|docs|deploy|evidence|"
    r"migration|frontend)/[\w./-]+(?:" + "|".join(
        e.replace(".", r"\.") for e in EXTENSIONS) + r"))"
)


def _sources() -> list[Path]:
    out: list[Path] = []
    for r in ROOTS:
        out += [p for p in (ROOT / r).rglob("*.py")
                if "__pycache__" not in str(p)]
    return sorted(out)


def _cited(path: Path) -> set[str]:
    text = path.read_text(errors="replace")
    return {m.group(1) for m in _PATH.finditer(text)
            if not any(t in m.group(1) for t in PLACEHOLDERS)}


ALL = {p: _cited(p) for p in _sources()}
WITH_CITATIONS = sorted([p for p, c in ALL.items() if c], key=str)


@pytest.mark.parametrize("path", WITH_CITATIONS,
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_every_cited_file_exists(path: Path) -> None:
    missing = sorted(
        c for c in ALL[path]
        if not (ROOT / c).exists() and c not in ALLOWED_ABSENT
    )
    assert not missing, (
        f"{path.relative_to(ROOT)} names files that do not exist: {missing}. "
        "A citation that reads like evidence and points at nothing is worse "
        "than no citation, because nobody checks."
    )


def test_the_absent_allowlist_has_no_stale_entries() -> None:
    """An allowance for a file that now exists is a lie in a guard."""
    for rel, reason in ALLOWED_ABSENT.items():
        assert not (ROOT / rel).exists(), (
            f"{rel} exists now ({reason}) -- remove it from ALLOWED_ABSENT"
        )


def test_the_scan_actually_finds_citations() -> None:
    """A regex that matches nothing passes every test above.

    So: the seam must be seen to cite its own guard, which is the citation
    this whole file was written after.
    """
    seam = ROOT / "src" / "coverline" / "core" / "interfaces.py"
    assert "tests/core/test_point_in_time.py" in ALL[seam]
    assert sum(len(c) for c in ALL.values()) > 30, (
        "the path scanner is finding almost nothing, which means it is not "
        "guarding anything"
    )
