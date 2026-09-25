"""Immutable snapshot storage, and the gap log that makes absence visible.

BRONZE IS WRITE-ONCE
A captured snapshot is raw data in the ledger's taxonomy: it cannot be
regenerated at any price, because a market close that was not captured is
gone. So this store refuses to overwrite. An attempt to write a snapshot that
already exists raises rather than replacing, because the only ways that
happens are a bug or a duplicate job, and neither should be resolved by
silently discarding one of two versions of the truth.

GAPS ARE RECORDED, NEVER FILLED
The important half of this module. If a capture does not happen -- outage,
quota exhaustion, a crash, a schedule that simply did not fire -- the absence
is written to the gap log with a reason. Nothing interpolates, and no
downstream code is allowed to treat a missing close as an average one.

This matters more than it sounds. CLV is the metric the whole evaluation
approach rests on, and it was adopted for variance reduction. A series with
invented values in it has a variance that is a property of the invention, not
of the betting. One quietly filled gap costs more than ten honestly recorded
ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

GAP_LOG = "_gaps.jsonl"
MANIFEST = "_manifest.jsonl"


class SnapshotExists(FileExistsError):
    """Refusing to overwrite raw data. See module docstring."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(part: str) -> str:
    """Filesystem-safe fragment. Colons break on some filesystems."""
    return part.replace(":", "").replace("/", "-")


@dataclass(frozen=True)
class StoredSnapshot:
    path: Path
    sport: str
    captured_at: str
    n_events: int
    cost: int


class BronzeStore:
    """Append-only snapshot storage rooted at a directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- paths ------------------------------------------------------------

    def snapshot_path(self, sport: str, captured_at: str, kind: str = "current") -> Path:
        return self.root / "odds" / sport / f"{kind}-{_safe(captured_at)}.json"

    @property
    def gap_log(self) -> Path:
        return self.root / "odds" / GAP_LOG

    @property
    def manifest(self) -> Path:
        return self.root / "odds" / MANIFEST

    # -- writing ----------------------------------------------------------

    def write_snapshot(
        self,
        *,
        sport: str,
        captured_at: str,
        payload: Any,
        cost: int,
        source_url: str,
        kind: str = "current",
    ) -> StoredSnapshot:
        path = self.snapshot_path(sport, captured_at, kind)
        if path.exists():
            raise SnapshotExists(
                f"{path} already exists. Bronze is write-once: a second "
                "snapshot for the same sport and timestamp means a duplicate "
                "job or a bug, and overwriting would discard one of two "
                "versions of what the market actually showed."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "_meta": {
                "sport": sport,
                "captured_at": captured_at,
                "written_at": _utc_now(),
                "kind": kind,
                "cost_credits": cost,
                "source_url": source_url,
            },
            "payload": payload,
        }
        path.write_text(json.dumps(record, separators=(",", ":")))

        n = len(payload) if isinstance(payload, list) else 1
        self._append(self.manifest, {
            "sport": sport, "captured_at": captured_at, "kind": kind,
            "path": str(path.relative_to(self.root)), "n_events": n,
            "cost_credits": cost,
        })
        return StoredSnapshot(path=path, sport=sport, captured_at=captured_at,
                              n_events=n, cost=cost)

    def record_gap(
        self, *, sport: str, intended_at: str, reason: str, detail: str = "",
    ) -> None:
        """Write down a capture that did not happen.

        `reason` should be a short machine-readable slug -- quota_exhausted,
        http_error, transport_failure, schedule_missed -- so gaps can be
        counted by cause without parsing prose.
        """
        if not reason:
            raise ValueError("a gap must have a reason; 'unknown' is a reason")
        self._append(self.gap_log, {
            "sport": sport, "intended_at": intended_at, "reason": reason,
            "detail": detail[:500], "recorded_at": _utc_now(),
        })

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    # -- reading ----------------------------------------------------------

    def read_snapshot(self, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    def snapshots(self, sport: str | None = None) -> Iterator[dict[str, Any]]:
        if not self.manifest.exists():
            return iter(())
        rows = (json.loads(l) for l in self.manifest.read_text().splitlines() if l.strip())
        return (r for r in rows if sport is None or r["sport"] == sport)

    def gaps(self, sport: str | None = None) -> list[dict[str, Any]]:
        if not self.gap_log.exists():
            return []
        rows = [json.loads(l) for l in self.gap_log.read_text().splitlines() if l.strip()]
        return [r for r in rows if sport is None or r["sport"] == sport]

    def coverage(self, sport: str) -> dict[str, int]:
        """Captured versus missed, so a degrading job is visible as a number.

        A rising gap count with a flat snapshot count is the shape of a
        capture process that has quietly stopped working.
        """
        # Distinct windows, not rows: logs written before 2026-09-25 recorded
        # the same missed window once per run, and counting rows would make
        # one missed game look like four.
        return {
            "snapshots": sum(1 for _ in self.snapshots(sport)),
            "gaps": len({(g["sport"], g["intended_at"]) for g in self.gaps(sport)}),
        }
