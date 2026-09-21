"""Nothing the repo depends on may be invisible to git.

WHY THIS EXISTS
data/nfl_key_numbers.json and model/nfl_key_numbers.py were both silently
ignored by the `*_key*` rule in .gitignore -- a secrets safety net aimed at
api_key and private_key files that also swallows any legitimate filename
containing "_key". The commit simply refused them, which was lucky; had they
been added before that rule existed, or added with -f by someone in a hurry,
the artifact would have shipped citing a generator that no clone contains.

That is the same shape as the project's oldest open wound: eight constants
citing a grid search whose output exists in no committed file. An ignored file
and a file that was never written look identical to everyone downstream.

So: every path referenced by artifacts.yml or migration/ledger.yaml must be
tracked by git, not merely present on the disk of whoever wrote it.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return set(out.stdout.splitlines())


def _referenced_paths() -> set[str]:
    """Concrete file paths named by the two manifests. Globs are skipped --
    they describe a family, not a file."""
    refs: set[str] = set()

    art = yaml.safe_load((ROOT / "artifacts.yml").read_text())
    for a in art.get("artifacts") or []:
        for ref in [a.get("produced_by"), *(a.get("consumed_by") or [])]:
            if ref:
                refs.add(ref)
        path = a.get("path", "")
        if path and "*" not in path:
            refs.add(path)

    led = yaml.safe_load((ROOT / "migration" / "ledger.yaml").read_text())
    for row in led.get("components") or []:
        new = row.get("new_path") or ""
        # ledger new_paths are prose as often as paths; take only the ones
        # that unambiguously name a single file that exists.
        if new.endswith((".py", ".json", ".yaml", ".csv")) and " " not in new:
            refs.add(new)
        if row.get("adr"):
            refs.add(f"docs/decisions/{row['adr']}")

    return refs


def test_every_referenced_file_that_exists_is_tracked():
    tracked = _tracked()
    untracked = []
    for ref in sorted(_referenced_paths()):
        if not (ROOT / ref).exists():
            continue  # absence is the artifact-manifest guard's job, not this one
        if ref not in tracked:
            untracked.append(ref)
    assert not untracked, (
        f"referenced but NOT TRACKED by git: {untracked}. These exist on this "
        "machine and in no clone. Check .gitignore -- the `*_key*` rule has "
        "caught legitimate files before."
    )


def test_the_key_number_files_specifically_are_tracked():
    """The two that were actually caught. Pinned by name because the
    .gitignore negation protecting them is easy to tidy away."""
    tracked = _tracked()
    for ref in ("model/nfl_key_numbers.py", "data/nfl_key_numbers.json"):
        assert ref in tracked, f"{ref} is not tracked; check the .gitignore negations"


def test_the_gitignore_negations_are_still_present():
    """A direct guard on the fix, since deleting a negation looks like
    tidying and behaves like deletion."""
    text = (ROOT / ".gitignore").read_text()
    for neg in ("!**/nfl_key_numbers.py", "!**/nfl_key_numbers.json",
                "!**/qb_overrides.json"):
        assert neg in text, f"the {neg} negation was removed from .gitignore"


def test_the_broad_secrets_rule_is_still_there():
    """The negations exist BECAUSE the blunt rule is worth keeping. If someone
    solves the collision by deleting `*_key*` instead, that trades a visible
    annoyance for an invisible credential leak.

    Matches an actual RULE LINE, not a substring. The first version of this
    test checked `"*_key*" in text`, which also matched the explanatory
    comment sitting directly above the rule -- so deleting the rule left the
    test green. Caught by trying it.
    """
    lines = [l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()]
    rules = [l for l in lines if l and not l.startswith("#")]
    assert "*_key*" in rules, (
        "the *_key* secrets rule was removed. Narrow it with negations rather "
        "than dropping it -- it is the thing stopping api_key files being "
        "committed."
    )


def test_every_negation_names_a_file_that_exists():
    """A negation for a file nobody has is dead config that reads as
    protection."""
    lines = [l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()]
    for neg in [l for l in lines if l.startswith("!")]:
        stem = neg.lstrip("!").replace("**/", "")
        assert any(p.name == stem for p in ROOT.rglob(stem)), (
            f"{neg} protects a file that does not exist anywhere in the repo"
        )


def test_no_python_module_under_src_is_ignored():
    """The core package itself must be wholly visible."""
    tracked = _tracked()
    missing = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        if str(p.relative_to(ROOT)) not in tracked
    ]
    assert not missing, f"untracked modules under src/: {missing}"
