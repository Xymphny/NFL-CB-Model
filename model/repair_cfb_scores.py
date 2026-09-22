#!/usr/bin/env python3
"""Replace the corrupt CFB scores with a second source's, and record every change.

WHY THIS IS A CORRECTION AND NOT AN INVENTION
ADRs 0019 and 0020 both refused to repair these rows, on the grounds that
rewriting a result by hand is inventing one. That was right at the time: there
was no second source, so any repair would have been a guess.

There is one now. data/raw/cfb/espn_*.parquet carries the same seasons from a
different pipeline, keyed on the same event ids, with a completion flag the
original cache does not have. Replacing a frozen score with what an
independent source recorded is adjudication, not fabrication -- and every
change is written to model/cfb_score_repairs.csv so the claim can be audited
rather than trusted.

WHAT THE CROSS-CHECK FOUND, WHICH IS WORSE THAN EITHER ADR ESTABLISHED
159 of 1,731 rows in the constants cache -- 9.19%, not the 4.39% the tie check
could see -- carry a wrong score. 58 of those FLIP THE WINNER, so
actual_home_win is wrong on 3.4% of the cache as well.

The pattern is a frozen score rather than a missing one: Vanderbilt 27-28
UConn was really 30-28, Missouri 16-23 Florida was really 24-23. A tie check
finds these only when the freeze happens to land level, which is why 0019
found 76 and there are 159.

Under the corrected scores there are ZERO ties across all 1,731 games, which
is what a sport that abolished them in 1996 should look like.

RATINGS ARE UNAFFECTED. rating_diff is computed from play-by-play VOA in
model/cfb_full_walk_forward.py and never reads a final score, so only the
outcome columns move.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

SCHEDULE = HERE / "cfb_schedule_cache.csv"
WALKFORWARD = HERE / "cfb_full_walk_forward_cache.csv"
REPAIRS = HERE / "cfb_score_repairs.csv"
KEY = ["season", "week", "home_team", "away_team"]


def espn() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "cfb" / "espn_*.parquet")))
    if not files:
        raise SystemExit("run model/ingest/cfb_espn.py first")
    e = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    e = e[e.completed & e.home_score.notna() & e.away_score.notna()]
    return e[["game_id", "home_score", "away_score"]].astype(
        {"home_score": int, "away_score": int})


def main() -> int:
    ref = espn().rename(columns={"home_score": "h_ref", "away_score": "a_ref"})

    sched = pd.read_csv(SCHEDULE)
    j = sched.merge(ref, on="game_id", how="left")
    fixable = j.h_ref.notna()
    changed = fixable & ((j.home_score != j.h_ref) | (j.away_score != j.a_ref))

    repairs = j[changed][["game_id", "season", "week", "home_team", "away_team",
                          "home_score", "away_score", "h_ref", "a_ref"]].rename(
        columns={"home_score": "was_home", "away_score": "was_away",
                 "h_ref": "now_home", "a_ref": "now_away"})
    # ACCUMULATE. This script is idempotent -- a second run finds nothing left
    # to change -- so overwriting the record on the second run erased the
    # first run's 215 repairs and kept only the 100 new ones. The record of
    # what was changed is the entire justification for changing it, and it
    # cannot be allowed to shrink.
    if REPAIRS.exists():
        prior = pd.read_csv(REPAIRS)
        repairs = pd.concat([prior, repairs], ignore_index=True)
        repairs = repairs.drop_duplicates("game_id", keep="first")
    repairs.sort_values(["season", "week", "game_id"]).to_csv(REPAIRS, index=False)

    sched.loc[changed, "home_score"] = j.loc[changed, "h_ref"].astype(int)
    sched.loc[changed, "away_score"] = j.loc[changed, "a_ref"].astype(int)
    sched.to_csv(SCHEDULE, index=False)
    print(f"schedule cache: {int(changed.sum())} of {int(fixable.sum())} "
          f"matched rows repaired ({int((~fixable).sum())} unmatched, left alone)")

    w = pd.read_csv(WALKFORWARD)
    ids = sched[KEY + ["game_id"]]
    wj = w.merge(ids, on=KEY, how="left").merge(ref, on="game_id", how="left")
    fix = wj.h_ref.notna()
    diff = fix & ((wj.home_score != wj.h_ref) | (wj.away_score != wj.a_ref))
    flips = int((diff & ((wj.home_score - wj.away_score > 0)
                         != (wj.h_ref - wj.a_ref > 0))).sum())

    w.loc[diff, "home_score"] = wj.loc[diff, "h_ref"].astype(int)
    w.loc[diff, "away_score"] = wj.loc[diff, "a_ref"].astype(int)
    w["actual_margin"] = w.home_score - w.away_score
    w["actual_home_win"] = w.actual_margin > 0
    w.to_csv(WALKFORWARD, index=False)
    print(f"walk-forward cache: {int(diff.sum())} of {len(w)} rows repaired, "
          f"{flips} of them flipping the winner")
    print(f"ties remaining: {int((w.actual_margin == 0).sum())}")
    print(f"repairs written to {REPAIRS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
