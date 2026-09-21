#!/usr/bin/env python3
"""Per-inning MLB runs, so the two rules can be measured rather than inferred.

WHY THIS AND NOT FINAL SCORES
ADR 0017 established that an MLB final score is a scoring process with two
league rules on top: extra innings resolve every game, and the home team
stops batting once it leads -- no bottom of the ninth while ahead, and a
walk-off ends play the instant it takes the lead. Both were diagnosed from
final scores alone, by their fingerprints: the winning home team scores 0.42
fewer runs than the winning away team, its variance is narrower, and it wins
by exactly one 6.1 points more often.

Fingerprints are enough to know a rule is acting. They are not enough to
model it. The quantity a layer needs is the score BEFORE the rule applies --
through eight and a half innings -- and that exists only in a linescore.

The hockey work made the same move for the same reason: ADR 0008's goalie-pull
layer is measurable because goal-level data was pulled, and a per-game
correction fitted to final scores would have been a residual fit wearing a
rule's clothes.

ONE REQUEST PER DATE covers every game that day, which is why five seasons
cost about 900 calls rather than 12,000.

WHAT A ROW IS
One game. `home_batted_ninth` is the fact the whole layer turns on: it is
False whenever the home team was ahead after the top of the ninth, which is
the truncation, and the score at that point is what an untruncated model
should be predicting.

Writes data/raw/mlb/linescores_{season}.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "mlb"
URL = ("https://statsapi.mlb.com/api/v1/schedule"
       "?sportId=1&date={day}&hydrate=linescore")
HEADERS = {"User-Agent": "coverline/1.0 (+research; contact via repo)"}
WORKERS = 8

#: Regular season only, and finished only. A suspended or postponed game has
#: a linescore that describes something other than a played game.
REGULAR = "R"
FINAL = ("Final", "Completed Early", "Game Over")


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


def day_rows(day: str) -> list[dict]:
    payload = _get(URL.format(day=day))
    rows = []
    for d in payload.get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != REGULAR:
                continue
            if g.get("status", {}).get("detailedState") not in FINAL:
                continue
            ls = g.get("linescore") or {}
            innings = ls.get("innings") or []
            if not innings:
                continue
            teams = ls.get("teams") or {}
            home_total = (teams.get("home") or {}).get("runs")
            away_total = (teams.get("away") or {}).get("runs")
            if home_total is None or away_total is None:
                continue

            # Runs through the END OF THE EIGHTH, and through the top of the
            # ninth. The second is the state the rule reads.
            h8 = sum((i.get("home") or {}).get("runs", 0) for i in innings[:8])
            a8 = sum((i.get("away") or {}).get("runs", 0) for i in innings[:8])
            ninth = innings[8] if len(innings) > 8 else None
            a9 = (ninth.get("away") or {}).get("runs", 0) if ninth else 0
            # A home half-inning that was never played has no runs key at all.
            home_batted_ninth = bool(
                ninth and "runs" in (ninth.get("home") or {}))

            rows.append({
                "season": int(g.get("season")),
                "game_pk": int(g["gamePk"]),
                "game_date": g.get("officialDate"),
                "home_team": (g["teams"]["home"]["team"]["name"]),
                "away_team": (g["teams"]["away"]["team"]["name"]),
                "home_score": int(home_total),
                "away_score": int(away_total),
                "innings_played": len(innings),
                "scheduled_innings": int(ls.get("scheduledInnings") or 9),
                "home_through_8": int(h8),
                "away_through_8": int(a8),
                "away_ninth": int(a9),
                "home_batted_ninth": home_batted_ninth,
            })
    return rows


def season_days(year: int) -> list[str]:
    d0, d1 = date(year, 3, 15), date(year, 10, 10)
    out, d = [], d0
    while d <= d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", default="2021-2025")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.seasons.split("-"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for y in range(lo, hi + 1):
        dest = OUT_DIR / f"linescores_{y}.parquet"
        if dest.exists() and not a.force:
            print(f"{y}: present, skipping", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            batches = list(ex.map(day_rows, season_days(y)))
        rows = [r for b in batches for r in b]
        df = pd.DataFrame(rows).drop_duplicates("game_pk")
        df = df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        df.to_parquet(dest, index=False)
        print(f"{y}: {len(df):5d} games  home batted 9th "
              f"{df.home_batted_ninth.mean():.3%}  -> {dest.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
