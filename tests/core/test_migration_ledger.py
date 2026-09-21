"""The ledger's promises, enforced.

A ledger nobody checks is a document. These tests make it a constraint: a
component cannot be dropped without an ADR that exists on disk, a data artifact
cannot be listed without saying how it survives, and the whole file cannot
quietly become empty.

The vacuity test matters most. Every other check here passes trivially against
zero rows, so without it the strongest-looking suite in the repo would go green
on a ledger someone had emptied -- which is precisely the shape of the failure
that blinded 25 of 27 guard tests when a stale artifact went unnoticed.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LEDGER_PATH = ROOT / "migration" / "ledger.yaml"

DISPOSITIONS = {"pending", "ported", "rewritten", "dropped", "deferred"}
DATA_DISPOSITIONS = {"raw", "training", "derived", "cache"}
NEEDS_ADR = {"dropped", "deferred"}

REQUIRED_FIELDS = {
    "id", "old_path", "new_path", "disposition", "adr",
    "data_artifacts", "data_disposition", "parity_status",
    "parity_tolerance", "old_removed", "notes",
}


def _ledger() -> dict:
    assert LEDGER_PATH.exists(), f"the migration ledger is missing: {LEDGER_PATH}"
    return yaml.safe_load(LEDGER_PATH.read_text())


def _rows() -> list[dict]:
    return _ledger()["components"]


def test_ledger_is_not_vacuous():
    """Every other test in this file passes against an empty ledger."""
    rows = _rows()
    assert len(rows) >= 5, (
        f"the migration ledger has only {len(rows)} rows; if components were "
        "removed rather than dispositioned, that is the silent-abandonment "
        "failure this file exists to prevent"
    )


def test_every_row_has_every_field():
    for row in _rows():
        missing = REQUIRED_FIELDS - set(row)
        assert not missing, f"{row.get('id', '<no id>')} is missing {sorted(missing)}"


def test_ids_are_unique():
    ids = [r["id"] for r in _rows()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ledger ids: {sorted(dupes)}"


def test_every_disposition_is_from_the_closed_set():
    for row in _rows():
        assert row["disposition"] in DISPOSITIONS, (
            f"{row['id']} has disposition {row['disposition']!r}; "
            f"the closed set is {sorted(DISPOSITIONS)}. Adding a state means "
            "editing this test deliberately, which is the point."
        )


def test_dropped_and_deferred_rows_name_an_adr_that_exists():
    """Condition 1, enforced. A component may be abandoned; it may not be
    abandoned silently."""
    for row in _rows():
        if row["disposition"] not in NEEDS_ADR:
            continue
        adr = row["adr"]
        assert adr, (
            f"{row['id']} is {row['disposition']} with no ADR. Record what it "
            "was, why it went, and where to find it."
        )
        path = ROOT / "docs" / "decisions" / adr
        assert path.exists(), f"{row['id']} names ADR {adr!r}, which is not on disk"


def test_data_artifacts_declare_how_they_survive():
    """Condition 2, enforced. Naming data without saying what happens to it is
    how training data disappears in a rebuild."""
    for row in _rows():
        if not row["data_artifacts"]:
            continue
        assert row["data_disposition"] in DATA_DISPOSITIONS, (
            f"{row['id']} lists data artifacts {row['data_artifacts']} but its "
            f"data_disposition is {row['data_disposition']!r}; must be one of "
            f"{sorted(DATA_DISPOSITIONS)}"
        )


def test_nothing_is_removed_before_it_is_dispositioned():
    for row in _rows():
        if row["old_removed"]:
            assert row["disposition"] in {"ported", "rewritten", "dropped"}, (
                f"{row['id']} claims the old path was removed while its "
                f"disposition is {row['disposition']!r}"
            )


def test_pending_rows_are_visible_rather_than_forgotten():
    """Not a failure -- pending is legitimate -- but the count is printed so a
    ledger drifting toward all-pending is noticed."""
    pending = [r["id"] for r in _rows() if r["disposition"] == "pending"]
    print(f"\n  {len(pending)} component(s) still pending: {pending}")
    assert len(pending) < 50, "pending backlog has become the whole ledger"


def test_the_known_condition_two_loss_stays_open():
    """The grid-search output that no committed file contains.

    This row must not be quietly closed. Closing it requires either finding the
    output or rerunning and committing the search -- and if someone does either,
    they must edit this test on purpose, which is the intended friction.
    """
    rows = {r["id"]: r for r in _rows()}
    grid = rows.get("frozen-threshold-grid")
    assert grid is not None, "the frozen-threshold-grid row was removed"
    assert grid["disposition"] == "pending", (
        "frozen-threshold-grid was closed; if the grid output was recovered or "
        "regenerated, commit it and update this test deliberately"
    )
    assert grid["data_disposition"] == "training", (
        "the grid output is judgement encoded as data and is NOT reproducible; "
        "reclassifying it as derived or cache would authorise deleting it"
    )
