"""The ADR machinery, held to its own promises.

The migration ledger can require an ADR to exist. It cannot require the ADR to
be USEFUL. These tests cover the gap: a drop record whose Recovery section has
been stripped is a file that satisfies the ledger check and tells a future
reader nothing, which is the silent-abandonment failure wearing a filename.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DECISIONS = ROOT / "docs" / "decisions"
TEMPLATE = DECISIONS / "0000-template.md"
STATUSES = {"proposed", "accepted", "rejected", "superseded"}

REQUIRED_SECTIONS = (
    "## Context and Problem Statement",
    "## Considered Options",
    "## Decision Outcome",
    "## Consequences",
    "## Recovery",
    "## Revisit Triggers",
)


def _records() -> list[Path]:
    return sorted(p for p in DECISIONS.glob("*.md")
                  if p.name not in {"README.md", "0000-template.md"})


def test_the_template_exists_and_keeps_its_non_standard_sections():
    """Recovery and Revisit Triggers are the two sections that make this more
    than MADR. If someone 'tidies' the template back to the standard shape,
    every future drop record loses the part that makes it recoverable."""
    assert TEMPLATE.exists(), "the ADR template is missing"
    text = TEMPLATE.read_text()
    assert "## Recovery" in text, "the template lost its Recovery section"
    assert "## Revisit Triggers" in text, "the template lost its Revisit Triggers"
    assert "archive/<ledger-id>" in text, (
        "the template no longer tells authors where to record the git tag"
    )


def test_every_record_is_numbered_and_unique():
    numbers = []
    for path in _records():
        m = re.match(r"^(\d{4})-[a-z0-9-]+\.md$", path.name)
        assert m, f"{path.name} does not match NNNN-kebab-title.md"
        numbers.append(m.group(1))
    assert len(numbers) == len(set(numbers)), f"duplicate ADR numbers: {numbers}"


def test_every_record_has_frontmatter_with_a_known_status():
    for path in _records():
        text = path.read_text()
        assert text.startswith("---\n"), f"{path.name} has no frontmatter"
        block = text.split("---", 2)[1]
        status = re.search(r"^status:\s*(\w+)", block, re.M)
        assert status, f"{path.name} has no status"
        assert status.group(1) in STATUSES, (
            f"{path.name} status {status.group(1)!r} not in {sorted(STATUSES)}"
        )
        assert re.search(r"^date:\s*\d{4}-\d{2}-\d{2}", block, re.M), (
            f"{path.name} has no valid date"
        )


def test_every_record_has_every_section():
    for path in _records():
        text = path.read_text()
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{path.name} is missing {section!r}"


def test_no_record_leaves_a_section_as_a_placeholder():
    """A heading with nothing under it passes a grep and fails a reader."""
    for path in _records():
        text = path.read_text()
        for section in REQUIRED_SECTIONS:
            start = text.index(section) + len(section)
            rest = text[start:]
            end = rest.index("\n## ") if "\n## " in rest else len(rest)
            body = rest[:end].strip()
            assert len(body) > 40, (
                f"{path.name}: section {section!r} is empty or a stub "
                f"({len(body)} chars)"
            )


def test_the_index_lists_every_record():
    """An index that silently falls behind is how a decision becomes
    unfindable while still technically being on disk."""
    index = (DECISIONS / "README.md").read_text()
    for path in _records():
        assert path.name in index, (
            f"{path.name} is not listed in docs/decisions/README.md"
        )


def test_superseded_records_name_their_successor():
    for path in _records():
        block = path.read_text().split("---", 2)[1]
        status = re.search(r"^status:\s*(\w+)", block, re.M).group(1)
        if status != "superseded":
            continue
        successor = re.search(r"^superseded_by:\s*(\S+)", block, re.M)
        assert successor and successor.group(1) not in {"null", "~", ""}, (
            f"{path.name} is superseded but does not say by what"
        )
        assert (DECISIONS / successor.group(1)).exists(), (
            f"{path.name} names a successor that is not on disk"
        )


def test_records_are_not_vacuous():
    assert len(_records()) >= 1, (
        "no decision records at all; if one was deleted rather than superseded, "
        "that is the failure this suite exists to prevent"
    )
