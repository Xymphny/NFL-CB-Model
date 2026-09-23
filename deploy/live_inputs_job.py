#!/usr/bin/env python3
"""Refresh the inputs the MLB, NHL and NBA live sources price from.

WHAT IT RUNS, IN ORDER
  nba   model/ingest/nba_espn.py   --seasons Y-Y --force  (the season in progress)
  nhl   model/ingest/nhl_api.py    --seasons Y-Y --force  (finals + schedule)
        model/ingest/nhl_goals.py  --seasons Y-Y --update (goal rows for new finals)
  mlb   model/ingest/mlb_slate.py  --date <today, US Eastern>
  audit model/audit_data_integrity.py
  settle scripts/settle_ledger.py  (closes + outcomes for paper trades; no regrade)
  export scripts/export_board.py + export_record.py  (the dashboard's data)

Then one commit of whatever changed, pushed through deploy/git_utils.

WHY THESE LEAGUES AND NOT MLB RESULTS
MLB results already have a job (mlb-daily-job, 09:00 UTC). This one runs
after it and pulls only the SLATE -- tonight's games and probable starters --
which the live source checks against that job's output game by game.

WHY TWICE A DAY (10:40 and 15:40 UTC)
The morning run lands after every West Coast game from the night before has
gone final, so the NBA and NHL files and the MLB slate's record of previous
results are complete. Probable starters are often still missing that early;
the second run, before the first afternoon first pitch, picks them up. Every
MLB slate pull is kept and the live source reads the newest.

EACH LEAGUE FAILS ALONE
One endpoint being down must not cost the other two leagues their refresh.
Every step runs; what succeeded is committed; then the job exits non-zero and
posts an alert naming what failed, so a broken feed is loud rather than a
file that quietly stops updating. The live sources refuse stale inputs on
their own as well -- this makes the cause visible before that happens.

THE SEASON IS THE ONE IN PROGRESS, named by the year it ends: October 2026
is 2027 for both leagues. Out of season it is the upcoming one, so its
schedule appears as soon as the league publishes it. Committed
sportsdataverse seasons are never touched -- nba_espn.py refuses them.

NOTHING CHANGED MEANS NOTHING COMMITTED. The ingests write byte-identical
files for identical data (checked when this job was built), so an unchanged
day produces no commit and no push.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOB = "live-inputs-job"

#: What this job may commit. Staged by path, never `git add .`.
PATHS = (
    "data/raw/sportsdataverse",
    "data/raw/nhl",
    "data/raw/mlb/slates",
    "model/data_integrity.json",
    "data/ledger",
    "data/site/board_nfl.json", "data/site/board_cfb.json", "data/site/board_mlb.json",
    "data/site/board_nhl.json", "data/site/board_nba.json",
    "data/site/record.json", "data/site/gates.json",
)


def season_ending(d: date) -> int:
    """NBA and NHL seasons are named by the year they end."""
    return d.year + 1 if d.month >= 8 else d.year


def _call(main: Callable[[list[str]], int], argv: list[str]) -> None:
    code = main(argv)
    if code not in (0, None):
        raise RuntimeError(f"exited {code}")


def steps(today_et: date) -> list[tuple[str, Callable[[], None]]]:
    y = season_ending(today_et)
    span = f"{y}-{y}"

    def nba():
        from model.ingest import nba_espn
        _call(nba_espn.main, ["--seasons", span, "--force"])

    def nhl():
        from model.ingest import nhl_api, nhl_goals
        _call(nhl_api.main, ["--seasons", span, "--force"])
        _call(nhl_goals.main, ["--seasons", span, "--update"])

    def mlb():
        from model.ingest import mlb_slate
        _call(mlb_slate.main, ["--date", today_et.isoformat()])

    def audit():
        # Run as a script: it writes model/data_integrity.json and its exit
        # code is not a pass/fail signal, the file is. A frame that fails an
        # integrity check is committed as failing, which turns CI red --
        # the loud outcome, on purpose.
        subprocess.run([sys.executable, str(ROOT / "model" / "audit_data_integrity.py")],
                       cwd=ROOT, check=True)

    def settle():
        # Closes and outcomes for every paper trade, now that the finals above
        # are current. NOT --grade: a regrade can change what the operator
        # command stakes, and that is a decision to make on purpose (ADR 0024).
        # When a league's ledger passes the grading floor, CI's reproduction
        # test fails and says so.
        sys.path.insert(0, str(ROOT / "scripts"))
        import settle_ledger
        _call(settle_ledger.main, [])

    def export():
        # The dashboard's board, record and gates, from inputs just refreshed.
        # Last, so it reflects every step before it.
        sys.path.insert(0, str(ROOT / "scripts"))
        import export_board
        import export_record
        _call(export_board.main, [])
        _call(export_record.main, [])

    return [("nba", nba), ("nhl", nhl), ("mlb", mlb), ("audit", audit),
            ("settle", settle), ("export", export)]


def changed(paths=PATHS) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return [line[3:] for line in out.stdout.splitlines() if line.strip()]


def run(today_et: date | None = None, commit: bool = True) -> int:
    today_et = today_et or datetime.now(ZoneInfo("America/New_York")).date()
    failed: list[str] = []
    for name, fn in steps(today_et):
        print(f"[{JOB}] {name} ...", flush=True)
        try:
            fn()
        except Exception as exc:          # one league must not stop the rest
            traceback.print_exc()
            failed.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")

    files = changed()
    print(f"[{JOB}] {len(files)} changed file(s)")
    for f in files[:20]:
        print(f"    {f}")

    if files and commit:
        from deploy.git_utils import git_commit_and_push
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        git_commit_and_push(list(PATHS), f"Live inputs {stamp}: {len(files)} file(s)"
                            + (f"; FAILED {', '.join(f.split(':')[0] for f in failed)}"
                               if failed else ""))

    if failed:
        msg = "\n".join(failed)
        print(f"[{JOB}] FAILED:\n{msg}")
        try:
            from deploy.notify import send_webhook_alert
            send_webhook_alert(f":rotating_light: **{JOB}** step(s) failed\n{msg}")
        except Exception:
            pass                            # the exit code still says it
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(commit=bool(os.environ.get("GIT_REPO_URL"))))
