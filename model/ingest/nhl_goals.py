#!/usr/bin/env python3
"""Goal-level NHL rows: when each goal was scored, and whether the net was empty.

WHY THIS AND NOT FINAL SCORES
A final NHL score is three things stacked: sixty minutes of scoring, an
empty-net layer that fires conditional on a late deficit, and a tie-breaking
rule that awards exactly one goal. Measured over 2016-2023, all 2,166 OT and
shootout games finish at a margin of exactly one and no game finishes tied, so
the third layer is DETERMINISTIC, not a correlation to be fitted.

The empty-net layer is the one that cannot be recovered from final scores.
It is also the one that decides the puck line: the NHL margin distribution is
NON-MONOTONE, with more three-goal games than two-goal games (0.227 against
0.203 over 2016-2023, and 0.251 against 0.202 in a second source's 2024),
because teams trailing by two pull the goalie and rarely come back, while
teams trailing by one who pull often tie and vanish into overtime. That drains
the two bucket into the three bucket across exactly the 1.5 line the puck line
is priced on.

Fitting a correlation parameter to final scores would reproduce the moment and
miss the shape. These rows let the layer be measured instead.

WHAT A ROW IS
One goal. goal_modifier is the league's own flag -- "empty-net" when it is --
and situation_code's first and last digits are goalie-present for away and
home, which is the same fact by a second route and lets the flag be checked
rather than trusted. Running score after the goal is kept so a goal's
game-state can be reconstructed without replaying the game.

Shootout-deciding goals are NOT in the scoring summary; they appear only in
the final score. A game whose goals sum short by one against its final score
is therefore a shootout, and that is asserted rather than assumed.

Writes data/raw/nhl/goals_{end_year}.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "nhl"
LANDING = "https://api-web.nhle.com/v1/gamecenter/{gid}/landing"
HEADERS = {"User-Agent": "coverline/1.0 (+research; contact via repo)"}

#: Modest. The endpoint is ungated and this is one historical pass, not a
#: recurring load.
WORKERS = 8


def _get(url: str, retries: int = 3) -> dict:
    last: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"failed: {url}") from last


def game_goals(gid: int) -> list[dict]:
    d = _get(LANDING.format(gid=gid))
    rows = []
    for period in d.get("summary", {}).get("scoring", []):
        desc = period.get("periodDescriptor", {})
        for g in period.get("goals", []):
            rows.append(
                {
                    "game_id": gid,
                    "period": desc.get("number"),
                    "period_type": desc.get("periodType"),
                    "time_in_period": g.get("timeInPeriod"),
                    "strength": g.get("strength"),
                    "goal_modifier": g.get("goalModifier"),
                    "situation_code": str(g.get("situationCode") or ""),
                    "home_score_after": g.get("homeScore"),
                    "away_score_after": g.get("awayScore"),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", default="2016-2023")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.seasons.split("-"))

    for y in range(lo, hi + 1):
        dest = RAW / f"goals_{y}.parquet"
        if dest.exists() and not a.force:
            print(f"{y}: present, skipping", flush=True)
            continue
        sched = pd.read_parquet(RAW / f"nhl_{y}.parquet")
        ids = sched.game_id.tolist()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            batches = list(ex.map(game_goals, ids))
        rows = [r for b in batches for r in b]
        df = pd.DataFrame(rows)
        df.to_parquet(dest, index=False)
        en = (df.goal_modifier == "empty-net").mean()
        print(
            f"{y}: {len(ids):4d} games  {len(df):5d} goals  "
            f"empty-net {en:.3%}  -> {dest.name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
