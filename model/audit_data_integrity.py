#!/usr/bin/env python3
"""Run every retained frame past every impossibility that applies to it.

WHY THIS EXISTS
Three times now a model or a cache has been found carrying an outcome its
league does not permit -- a tied NHL game (ADR 0007), a tied MLB game (ADR
0017), and 76 tied CFB games in the cache that produced two shipped constants
(ADR 0019). Each was found while looking at something else.

Finding the same class of fault three times by accident is an argument for
looking on purpose. This walks every retained frame past every check that
applies to it and reports, so the fourth one is found by a script rather than
by luck.

WHAT IS CHECKED
  - TIES, against what each league permits. The NFL genuinely has them at
    about 0.2%; nobody else does.
  - IMPOSSIBLE TEAM SCORES. Football cannot produce a total of one point.
  - SELF-PLAY and DUPLICATE KEYS, which are not rule violations but are
    invisible to every statistical check: they move a mean slightly and
    nothing else.

WHAT IS NOT CHECKED, AND IS WORTH SAYING
Scores that are individually possible and jointly absurd. A 2-0 NFL game is
legal and has happened once since 1940; a 1-0 baseball game is ordinary. This
looks for the impossible, not the improbable, because the improbable needs a
distribution and the impossible needs only the rulebook.

Exit code 1 if anything fails, so this can gate a pipeline. Writes
model/data_integrity.json either way.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model.fit_data_checks import (  # noqa: E402
    ImpossibleOutcome, check_frame_shape, check_impossible_scores,
    check_no_impossible_ties,
)

#: (path, league, margin column or None, key column or None). A frame with a
#: margin column instead of two scores gets the tie check only.
FRAMES: list[tuple[str, str, str | None, str | None]] = [
    ("model/cfb_schedule_cache.csv", "cfb", None, "game_id"),
    ("model/cfb_full_walk_forward_cache.csv", "cfb", "actual_margin", None),
    ("model/expanded_walk_forward_cache.csv", "nfl", "actual_margin", None),
    ("model/mlb_schedule_cache.csv", "mlb", None, None),
]


def _discovered() -> list[tuple[str, str, str | None, str | None]]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "nhl" / "nhl_*.parquet"))):
        out.append((str(Path(f).relative_to(ROOT)), "nhl", None, "game_id"))
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "mlb" /
                                  "linescores_*.parquet"))):
        out.append((str(Path(f).relative_to(ROOT)), "mlb", None, "game_pk"))
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "sportsdataverse" /
                                  "*.parquet"))):
        league = Path(f).stem.split("_")[0]
        out.append((str(Path(f).relative_to(ROOT)), league, None, None))
    return out


def _load(rel: str) -> tuple[pd.DataFrame, dict]:
    """The frame, restricted to games that were actually played.

    AN ABSENT RESULT IS NOT A FALSE ONE, and conflating them made the first
    version of this script report every NBA season as failing. Each of those
    files carries exactly one row at 0-0 with status_type_completed = False --
    a cancelled game, correctly excluded by the fit's own filter, and not a
    tie.

    The distinction is the whole story of ADR 0019 in reverse: the CFB
    schedule cache has NO completion column, so its 83 unfetched games sit in
    it as 0-0 results indistinguishable from played ones. A frame that can
    tell you a game was not played is a frame whose zeros are safe.
    """
    path = ROOT / rel
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    info: dict = {"rows_on_disk": int(len(df)), "excluded": {}}

    if "game_type" in df.columns:
        n = int((df.game_type != "R").sum())
        if n:
            info["excluded"]["not_regular_season"] = n
        df = df[df.game_type == "R"]
    if "type_abbreviation" in df.columns:
        n = int((df.type_abbreviation != "STD").sum())
        if n:
            info["excluded"]["not_standard_game"] = n
        df = df[df.type_abbreviation == "STD"]
    if "status_type_completed" in df.columns:
        n = int((~df.status_type_completed.astype(bool)).sum())
        if n:
            info["excluded"]["not_completed"] = n
        df = df[df.status_type_completed.astype(bool)]
    else:
        info["no_completion_column"] = (
            "this frame cannot distinguish an unplayed game from a 0-0 one"
        )
    for c in ("home_score", "away_score"):
        if c in df.columns:
            n = int(df[c].isna().sum())
            if n:
                info["excluded"]["missing_score"] = (
                    info["excluded"].get("missing_score", 0) + n)
            df = df[df[c].notna()]
    return df.reset_index(drop=True), info


def main() -> int:
    report: dict = {"_provenance": {"script": "model/audit_data_integrity.py"},
                    "frames": {}}
    failures = 0

    for rel, league, margin, key in FRAMES + _discovered():
        if not (ROOT / rel).exists():
            continue
        df, info = _load(rel)
        entry: dict = {"league": league, "rows": int(len(df)), **info,
                       "checks": {}}
        for name, fn in (
            ("ties", lambda: check_no_impossible_ties(
                df, league, label=rel, margin=margin)),
            ("impossible_scores", lambda: check_impossible_scores(
                df, league, label=rel)),
            ("frame_shape", lambda: check_frame_shape(df, label=rel, key=key)),
        ):
            if name == "impossible_scores" and "home_score" not in df.columns:
                entry["checks"][name] = {"skipped": "no score columns"}
                continue
            try:
                entry["checks"][name] = {"ok": True, **fn()}
            except ImpossibleOutcome as exc:
                entry["checks"][name] = {"ok": False, "error": str(exc)[:400]}
                failures += 1
            except (ValueError, KeyError) as exc:
                entry["checks"][name] = {"skipped": str(exc)[:200]}
        report["frames"][rel] = entry

    report["_provenance"]["frames_checked"] = len(report["frames"])
    report["_provenance"]["failures"] = failures
    (HERE / "data_integrity.json").write_text(json.dumps(report, indent=2) + "\n")

    for rel, e in report["frames"].items():
        bad = [n for n, c in e["checks"].items() if c.get("ok") is False]
        status = "FAIL " + ",".join(bad) if bad else "ok"
        note = "" if "no_completion_column" not in e else "  [no completion flag]"
        print(f"  {status:28} {e['rows']:>6} rows  {rel}{note}")
    print(f"\n{len(report['frames'])} frames, {failures} failing checks")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
