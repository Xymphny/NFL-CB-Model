"""Artifacts must declare a consumer, or say why they have none.

The orphaned-artifact problem has no static-analysis answer: a file nobody
reads looks exactly like a file whose reader is a path string somewhere else.
The only way to know is to write it down and check the writing.

This suite is deliberately strict about its own emptiness. The manifest starts
with zero artifacts, which would make every check below pass vacuously, so
test_empty_manifest_is_declared_not_accidental pins the empty state as a
DECISION. When the first artifact lands, that test is the one that fails and
forces the manifest to be filled rather than bypassed.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_PATH = ROOT / "artifacts.yml"
KINDS = {"bronze", "silver", "gold", "ledger", "archive"}
CORE = ROOT / "src" / "coverline"


def _manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"artifact manifest missing: {MANIFEST_PATH}"
    return yaml.safe_load(MANIFEST_PATH.read_text())


def _artifacts() -> list[dict]:
    return _manifest().get("artifacts") or []


def test_manifest_declares_what_it_does_not_cover():
    """An absence here must never be readable as a clean bill of health for
    the legacy pipeline."""
    scope = _manifest().get("coverage_scope")
    assert scope, "artifacts.yml must state what it does and does not cover"
    assert "legacy" in scope.lower() or "migration/ledger" in scope


#: Modules that write files. Grown deliberately: this list started as
#: [apps] when the core published nothing, and the guard below correctly
#: refused to accept artifact rows while it said so. execution/ was added on
#: 2026-09-21 when the bronze store landed. A new writing package must be
#: added here on purpose, which is the friction the check is for.
WRITING_PACKAGES = ("apps", "execution")


def _core_writes_files() -> bool:
    for pkg in WRITING_PACKAGES:
        d = CORE / pkg
        if d.exists() and any(p.stem != "__init__" for p in d.rglob("*.py")):
            return True
    return False


def test_manifest_matches_whether_the_core_actually_writes_anything():
    """Two failure modes, one test.

    Empty manifest while the core writes files = unregistered artifacts, which
    is the orphan problem. Rows listed while the core writes nothing = the
    manifest is describing files it does not cover, which is worse, because it
    asserts a lineage that is not there.
    """
    if _core_writes_files():
        assert _artifacts(), (
            f"a writing package exists under {WRITING_PACKAGES} but the "
            "manifest is empty; every published file needs a row"
        )
    else:
        assert _artifacts() == [], (
            "artifacts are listed but no core package writes files; either the "
            "manifest describes legacy files outside coverage_scope, or "
            "WRITING_PACKAGES needs updating"
        )


def test_every_writing_package_is_represented_in_the_manifest():
    """A package that writes files but appears in no `produced_by` is the
    orphan case one level up: registered artifacts, unregistered writer."""
    if not _core_writes_files():
        pytest.skip("no writing packages yet")
    producers = " ".join(a.get("produced_by", "") for a in _artifacts())
    for pkg in WRITING_PACKAGES:
        d = CORE / pkg
        if d.exists() and any(p.stem != "__init__" for p in d.rglob("*.py")):
            assert f"coverline/{pkg}/" in producers, (
                f"{pkg}/ writes files but produces no manifest artifact"
            )


def test_every_artifact_has_a_consumer_or_an_excuse():
    for art in _artifacts():
        consumers = art.get("consumed_by") or []
        if consumers:
            continue
        assert art.get("unconsumed_reason"), (
            f"{art['path']} is produced and read by nothing. Either wire up a "
            "consumer, delete it, or record unconsumed_reason saying why it is "
            "written anyway (an audit trail, say)."
        )


def test_artifact_paths_are_unique():
    paths = [a["path"] for a in _artifacts()]
    dupes = {p for p in paths if paths.count(p) > 1}
    assert not dupes, f"two artifacts claim the same path: {sorted(dupes)}"


def test_every_artifact_has_a_known_kind_and_a_producer():
    for art in _artifacts():
        assert art.get("kind") in KINDS, f"{art['path']}: kind must be one of {sorted(KINDS)}"
        assert art.get("produced_by"), f"{art['path']}: no producer named"


def test_named_python_producers_and_consumers_exist():
    """A manifest full of paths to deleted modules is worse than no manifest:
    it asserts a lineage that is not there."""
    for art in _artifacts():
        named = [art["produced_by"], *(art.get("consumed_by") or [])]
        for ref in named:
            if not str(ref).endswith((".py", ".jsx", ".ts", ".tsx")):
                continue
            assert (ROOT / ref).exists(), f"{art['path']} names {ref}, which is not on disk"
