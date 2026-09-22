#!/usr/bin/env python3
"""NBA seasons from ESPN's scoreboard, in the columns the committed files use.

WHY
The NBA replay carries ratings across seasons, so it needs every season from
2021 to the one in progress. The committed inputs are sportsdataverse's
files, which stop at 2024-25 (nba_2025). Their raw release paths are not
reachable from the build sandbox, and a live season needs a pull that can be
repeated daily anyway. sportsdataverse builds those files FROM this ESPN
endpoint, so this writes the same fields under the same names, and the
loader (model/fit_nba.load) reads the result without a single change.

TRUST IS CHECKED, NOT ASSUMED
Before relying on this for 2025-26 and 2026-27, run it against a season
that is already committed:

    python model/ingest/nba_espn.py --verify 2025

It pulls 2024-25 fresh and compares it game by game -- event id, both team
abbreviations, both scores -- with data/raw/sportsdataverse/nba_2025.parquet,
and exits 1 on any disagreement in a regular-season game. That is the
cross-source check the NHL and CFB ingests are built on: agreement validates
both, and disagreement localises the fault before it reaches a rating.

WHAT IT WRITES
data/raw/sportsdataverse/nba_{season}.parquet, beside the committed seasons,
because that is the path every loader reads. Each row carries
source = "espn_scoreboard" so no one mistakes it for a sportsdataverse file.
A committed sportsdataverse season is never overwritten, even with --force. For the season in
progress, re-run with --force before each slate: the live source refuses a
file in which a game that tipped hours ago is not yet final.

Scheduled games are written too, with completed = False. They are the slate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "sportsdataverse"
URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
       "scoreboard?dates={day}&limit=100")
#: No User-Agent override -- see model/ingest/cfb_espn.py for why this
#: endpoint is the opposite of the NHL one.
HEADERS: dict[str, str] = {}
WORKERS = 6
REGULAR = 2


def _get(url: str, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed: {url}") from last


def _score(t: dict) -> int | None:
    try:
        return int(t.get("score"))
    except (TypeError, ValueError):
        return None


def parse_scoreboard(payload: dict) -> list[dict]:
    """Regular-season rows in the committed files' column names. Pure."""
    rows = []
    for e in payload.get("events", []):
        season = e.get("season") or {}
        if season.get("type") != REGULAR:
            continue
        comp = (e.get("competitions") or [{}])[0]
        status = (comp.get("status") or {}).get("type") or {}
        teams = {t.get("homeAway"): t for t in comp.get("competitors", [])}
        home, away = teams.get("home"), teams.get("away")
        if not home or not away:
            continue
        ht, at = home.get("team") or {}, away.get("team") or {}
        rows.append({
            "id": int(e["id"]),
            "season": int(season.get("year")),
            "season_type": int(season.get("type")),
            "type_abbreviation": (comp.get("type") or {}).get("abbreviation"),
            "date": e.get("date"),
            "game_date": str(e.get("date"))[:10],
            "neutral_site": bool(comp.get("neutralSite", False)),
            "status_type_name": status.get("name"),
            "status_type_completed": bool(status.get("completed")),
            "home_abbreviation": ht.get("abbreviation"),
            "away_abbreviation": at.get("abbreviation"),
            "home_display_name": ht.get("displayName"),
            "away_display_name": at.get("displayName"),
            "home_score": _score(home),
            "away_score": _score(away),
            "source": "espn_scoreboard",
        })
    return rows


def season_days(end_year: int) -> list[str]:
    d, stop, out = date(end_year - 1, 10, 1), date(end_year, 6, 30), []
    while d <= stop:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def fetch_season(end_year: int) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        batches = list(ex.map(lambda d: parse_scoreboard(_get(URL.format(day=d))),
                              season_days(end_year)))
    df = pd.DataFrame([r for b in batches for r in b])
    if df.empty:
        raise RuntimeError(f"no regular-season games returned for {end_year}")
    df = df[df.season == end_year].drop_duplicates("id")
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def compare(fresh: pd.DataFrame, committed: pd.DataFrame) -> list[str]:
    """Disagreements on regular-season games between two pulls. Pure."""
    keep = ["id", "home_abbreviation", "away_abbreviation", "home_score", "away_score"]

    def std(d):
        d = d[(d.season_type == 2) & (d.type_abbreviation == "STD")]
        if "status_type_completed" in d.columns:
            d = d[d.status_type_completed.fillna(False).astype(bool)]
        d = d[keep].copy()
        d["id"] = d["id"].astype(int)
        return d.set_index("id")

    a, b = std(fresh), std(committed)
    out = [f"{i}: only in the committed file" for i in sorted(set(b.index) - set(a.index))]
    out += [f"{i}: only in the fresh pull" for i in sorted(set(a.index) - set(b.index))]
    for i in sorted(set(a.index) & set(b.index)):
        ra, rb = a.loc[i], b.loc[i]
        for c in keep[1:]:
            if str(ra[c]) != str(rb[c]) and not (
                    c.endswith("score") and int(ra[c]) == int(rb[c])):
                out.append(f"{i}: {c} {rb[c]!r} committed vs {ra[c]!r} fresh")
    return out


def is_ours(path: Path) -> bool:
    """True only for a file this script wrote."""
    cols = pd.read_parquet(path).columns
    if "source" not in cols:
        return False
    return bool((pd.read_parquet(path, columns=["source"]).source
                 == "espn_scoreboard").all())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", help="inclusive end-year range, e.g. 2026-2027")
    p.add_argument("--verify", type=int,
                   help="pull this committed season and compare, writing nothing")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    if a.verify:
        committed = pd.read_parquet(OUT_DIR / f"nba_{a.verify}.parquet")
        diffs = compare(fetch_season(a.verify), committed)
        for d in diffs[:40]:
            print(d)
        print(f"{a.verify}: {len(diffs)} disagreement(s)")
        return 1 if diffs else 0

    if not a.seasons:
        p.error("--seasons or --verify is required")
    lo, hi = (int(x) for x in a.seasons.split("-"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for y in range(lo, hi + 1):
        dest = OUT_DIR / f"nba_{y}.parquet"
        if dest.exists() and not a.force:
            print(f"{y}: present, skipping")
            continue
        if dest.exists() and not is_ours(dest):
            # A committed sportsdataverse season is a FIT INPUT. Overwriting
            # it would change the data the graded artifact was built from,
            # and --force is too easy to type to be the only thing in the way.
            print(f"{y}: {dest.name} is a committed sportsdataverse file; "
                  "refusing to overwrite it. Use --verify to compare instead.")
            continue
        df = fetch_season(y)
        df.to_parquet(dest, index=False)
        done = df.status_type_completed.mean()
        print(f"{y}: {len(df):4d} games, {done:.1%} final -> {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
