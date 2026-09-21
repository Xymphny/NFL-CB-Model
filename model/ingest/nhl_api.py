#!/usr/bin/env python3
"""NHL final scores from the league's own API, with the OT/SO flag.

WHY A SECOND SOURCE AT ALL
sportsdataverse's 2021-2023 NHL files carry a constant score on every row --
1,312 games all 6-3 -- which fit_data_checks.py now refuses. That left only
2024 and 2025, both of which are SPENT: 2025 on the static question, 2024 on
the walk-forward one. There was nothing clean left to grade a third question
on, and the third question -- the joint distribution behind the puck line --
is the one still withheld.

So the seasons come from api-web.nhle.com instead. They are unspent because
nothing has ever been graded on them, and they come from a different pipeline
than the corrupt files, which is the only reason to trust them more.

THE FIELD THAT MAKES THIS WORTH DOING
gameOutcome.lastPeriodType is REG, OT or SO. An NHL final score is not a
sample from a scoring process -- it is a regulation score with the league's
tie-breaking rule applied on top, and that rule awards exactly one goal. Any
model of the puck line that cannot see which games were tied after sixty
minutes is modelling the wrong quantity. This column is what makes the
regulation distribution recoverable.

CONFOUNDS, RECORDED NOW
2019-20 stopped in March 2020 at about 70 games a team and resumed as a
24-team tournament; the regular-season rows here end at the stoppage.
2020-21 was 56 games in fixed divisions with no cross-division play, so
strength of schedule is not comparable to any other season in this set.
Neither is dropped here -- dropping them is a modelling choice and belongs
downstream, where it can be recorded -- but neither should be pooled blindly.

Writes data/raw/nhl/nhl_{end_year}.parquet, one row per regular-season game.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "nhl"
SCHEDULE = "https://api-web.nhle.com/v1/schedule/{day}"

#: Regular season only. 1 is preseason, 3 playoffs, 4+ other.
REGULAR_SEASON = 2

#: Politeness, not rate-limit avoidance -- the endpoint is ungated.
SLEEP_S = 0.15

#: The endpoint answers curl and 403s urllib's default agent. This is not a
#: block being evaded -- the data is public and ungated -- it is a default
#: User-Agent that happens to be on a filter list.
HEADERS = {"User-Agent": "coverline/1.0 (+research; contact via repo)"}


def _get(url: str, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def fetch_season(end_year: int) -> pd.DataFrame:
    """Walk the schedule week by week from the season's first day.

    The endpoint hands back nextStartDate, so the walk follows the league's
    own calendar rather than guessing week boundaries.
    """
    season_id = (end_year - 1) * 10000 + end_year
    day = date(end_year - 1, 9, 15)
    stop = date(end_year, 7, 15)
    seen: set[int] = set()
    rows: list[dict] = []

    while day < stop:
        payload = _get(SCHEDULE.format(day=day.isoformat()))
        for week in payload.get("gameWeek", []):
            for g in week.get("games", []):
                if g.get("id") in seen:
                    continue
                if g.get("season") != season_id:
                    continue
                if g.get("gameType") != REGULAR_SEASON:
                    continue
                home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
                if home.get("score") is None or away.get("score") is None:
                    continue  # not played -- postponed, or the season is live
                seen.add(g["id"])
                rows.append(
                    {
                        "season": end_year,
                        "game_id": g["id"],
                        "game_date": week.get("date"),
                        "home_team_abbr": home.get("abbrev"),
                        "away_team_abbr": away.get("abbrev"),
                        "home_score": int(home["score"]),
                        "away_score": int(away["score"]),
                        "last_period_type": g.get("gameOutcome", {}).get(
                            "lastPeriodType"
                        ),
                        "neutral_site": bool(g.get("neutralSite", False)),
                    }
                )
        nxt = payload.get("nextStartDate")
        day = date.fromisoformat(nxt) if nxt else day + timedelta(days=7)
        time.sleep(SLEEP_S)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no games returned for season {end_year}")
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", default="2016-2023",
                   help="inclusive end-year range, e.g. 2016-2023")
    p.add_argument("--force", action="store_true",
                   help="refetch seasons already on disk")
    a = p.parse_args(argv)

    lo, hi = (int(x) for x in a.seasons.split("-"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for y in range(lo, hi + 1):
        dest = OUT_DIR / f"nhl_{y}.parquet"
        if dest.exists() and not a.force:
            print(f"{y}: present, skipping")
            continue
        df = fetch_season(y)
        df.to_parquet(dest, index=False)
        ot = (df.last_period_type != "REG").mean()
        print(f"{y}: {len(df):4d} games  OT/SO {ot:.1%}  -> {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
