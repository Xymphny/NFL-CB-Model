#!/usr/bin/env python3
"""Put the two scores back into a cache that kept only the margin.

WHY
ADR 0020 recorded a structural limit: a margin-only cache cannot be audited
for impossible scores. Kansas State 1-1 TCU appears as a margin of zero and is
caught by the tie check; Florida Atlantic 1-0 Georgia Southern appears as a
margin of +1 and is invisible. So at least one impossible game sat inside the
frame that produced MARGIN_SD and nothing could find it.

The scores were never missing. model/cfb_full_walk_forward.py reads
game["home_score"] and game["away_score"], computes a margin, and discards
both -- and model/cfb_schedule_cache.csv still has them.

THIS IS A DERIVATION, NOT A REPAIR
Nothing here invents a result. The scores are joined from the schedule cache
on (season, week, home team, away team), and the join is CHECKED: every
row must match exactly one schedule row, and the schedule's margin must equal
the margin already in the cache. All 1,731 rows match and no margin disagrees,
which is also the evidence that the corruption in ADR 0019 propagated straight
through rather than arising in the walk-forward step.

The generator is fixed too, so a regenerated cache carries the scores without
this script. It exists for the cache that already shipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

CACHE = HERE / "cfb_full_walk_forward_cache.csv"
SCHEDULE = HERE / "cfb_schedule_cache.csv"
KEY = ["season", "week", "home_team", "away_team"]


def main() -> int:
    w = pd.read_csv(CACHE)
    if {"home_score", "away_score"} <= set(w.columns):
        print("cache already carries both scores; nothing to do")
        return 0

    s = pd.read_csv(SCHEDULE).dropna(subset=["home_score", "away_score"])
    if int(s.duplicated(KEY).sum()):
        raise SystemExit("the schedule cache has duplicate keys; the join "
                         "would be ambiguous and is refused")

    j = w.merge(s[KEY + ["home_score", "away_score"]], on=KEY, how="left",
                validate="one_to_one")
    missing = int(j.home_score.isna().sum())
    if missing:
        raise SystemExit(f"{missing} rows have no schedule match; a partial "
                         "backfill would leave a column that is sometimes "
                         "evidence and sometimes absent")

    disagree = j[(j.home_score - j.away_score) != j.actual_margin]
    if len(disagree):
        raise SystemExit(
            f"{len(disagree)} rows disagree with the margin already recorded. "
            "That is a finding, not something to overwrite."
        )

    j["home_score"] = j.home_score.astype(int)
    j["away_score"] = j.away_score.astype(int)
    j.to_csv(CACHE, index=False)
    print(f"backfilled {len(j)} rows; the score check now applies to this cache")

    from model.fit_data_checks import ImpossibleOutcome, check_impossible_scores
    try:
        check_impossible_scores(j, "cfb", label=str(CACHE.relative_to(ROOT)))
        print("no impossible scores found in it")
    except ImpossibleOutcome as exc:
        print(f"AND IT IMMEDIATELY FINDS ONE: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
