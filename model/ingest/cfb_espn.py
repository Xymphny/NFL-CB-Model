#!/usr/bin/env python3
"""CFB final scores from a second source, with the completion flag the first lacks.

WHY
model/cfb_schedule_cache.csv carries 144 corrupt rows in 3,863 (ADR 0019 and
0020): 83 stored as 0-0 where no score was ever fetched, 59 frozen at an
intermediate score, and 2 with a team total of one point, which football
cannot produce. It has NO completion column, so its zeros are
indistinguishable from played games -- which is precisely why nobody noticed.

This is the move the NHL work made when sportsdataverse's files turned out to
carry a constant score on every row: pull the same seasons from a different
pipeline, join game by game, and let the two sources adjudicate. Agreement
validates both; disagreement localises the fault.

THE IDS LINE UP. ESPN's event ids are the same id space as the existing
cache's game_id, so the join needs no name matching and no fuzzy dates, which
is the usual way a cross-source check goes wrong.

Writes data/raw/cfb/espn_{season}.parquet.
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
OUT_DIR = ROOT / "data" / "raw" / "cfb"
URL = ("https://site.api.espn.com/apis/site/v2/sports/football/"
       "college-football/scoreboard?dates={day}")
#: NO User-Agent OVERRIDE, and that is not an oversight.
#:
#: This endpoint answers urllib's DEFAULT agent and returns 403 for both a
#: custom string and a browser one. That is the exact opposite of
#: api-web.nhle.com in model/ingest/nhl_api.py, which 403s the default and
#: answers a custom string. Neither is a block being evaded -- both serve
#: this data publicly and ungated -- and recording the asymmetry here saves
#: the next person the twenty minutes it cost to find.
HEADERS: dict[str, str] = {}
WORKERS = 8

#: Regular season. Type 3 is the postseason, which the walk-forward cache
#: does not contain and which would change what a season mean describes.
REGULAR = 2


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
    for e in payload.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        season = e.get("season") or {}
        if season.get("type") != REGULAR:
            continue
        status = (comp.get("status") or {}).get("type") or {}
        teams = {t.get("homeAway"): t for t in comp.get("competitors", [])}
        home, away = teams.get("home"), teams.get("away")
        if not home or not away:
            continue
        def _score(t):
            v = t.get("score")
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        rows.append({
            "game_id": int(e["id"]),
            "season": int(season.get("year")),
            "week": int((e.get("week") or {}).get("number") or 0),
            "game_date": str(e.get("date"))[:10],
            "home_team": (home.get("team") or {}).get("location"),
            "away_team": (away.get("team") or {}).get("location"),
            "home_score": _score(home),
            "away_score": _score(away),
            # THE COLUMN THE FIRST SOURCE DOES NOT HAVE, and the one that
            # makes a zero safe: a game that was not played says so.
            "completed": bool(status.get("completed")),
            "status": status.get("name"),
        })
    return rows


def season_days(year: int) -> list[str]:
    d0, d1 = date(year, 8, 20), date(year + 1, 1, 15)
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", default="2021-2023")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.seasons.split("-"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for y in range(lo, hi + 1):
        dest = OUT_DIR / f"espn_{y}.parquet"
        if dest.exists() and not a.force:
            print(f"{y}: present, skipping", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            batches = list(ex.map(day_rows, season_days(y)))
        rows = [r for b in batches for r in b]
        df = pd.DataFrame(rows).drop_duplicates("game_id")
        df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        df.to_parquet(dest, index=False)
        print(f"{y}: {len(df):5d} games  completed {df.completed.mean():.3%}"
              f"  -> {dest.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
