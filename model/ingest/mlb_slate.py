#!/usr/bin/env python3
"""Tonight's MLB slate with probable starters, keyed the way results will be.

WHY THIS EXISTS
The walk-forward prices a game from the two STARTING PITCHERS, and the
schedule cache only ever holds finished games. So a game cannot be priced
until someone says who is starting it -- which is exactly how books quote it:
MLB lines are conditional on the listed probables, and a scratch changes the
game being priced.

Everything here is keyed the way deploy/mlb_daily_update.py keys the finals
it writes the next morning -- the same team codes, the same doubleheader
game number, the same pitcher ids through the same Chadwick bridge (it
imports those helpers rather than restating them). A probable starter who
lands on a different key from his own completed starts would be priced as a
debut, and nothing downstream would notice.

PROOF OF FRESHNESS TRAVELS WITH THE SLATE
`previous_final_date` is the last date before this one on which any
regular-season game went final, read from the league's own schedule. The
live source refuses to price if the results cache does not reach it. An
All-Star break is therefore not mistaken for a stale cache, and a stale
cache is not mistaken for an off day.

Writes data/raw/mlb/slates/{date}-{stamp}.json. Every pull is kept: the
probables a line was priced against are evidence for grading it later, and
the newest file for a date is the one the live source reads.

statsapi.mlb.com is unreachable from the build sandbox; the parsers are
pure and tested on captured shapes, and the first live run verifies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.mlb_daily_update import (  # noqa: E402
    STATS_TO_RETRO, game_number_of, load_id_bridge, modal_parks, pitcher_key,
)

OUT_DIR = ROOT / "data" / "raw" / "mlb" / "slates"
URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={a}"
       "&endDate={b}&gameTypes=R&hydrate=probablePitcher,team")


def _team(side: dict) -> str:
    ab = side["team"].get("abbreviation")
    return STATS_TO_RETRO.get(ab, ab)


def parse_slate(payload: dict, day: str, bridge: dict, parks: dict) -> list[dict]:
    """Every regular-season game on `day`, with probables as model keys."""
    out = []
    for d in payload.get("dates", []):
        if d.get("date") != day:
            continue
        for g in d.get("games", []):
            h, a = g["teams"]["home"], g["teams"]["away"]
            home, away = _team(h), _team(a)
            num = game_number_of(g)

            def sp(side):
                p = side.get("probablePitcher") or {}
                return (pitcher_key(p["id"], bridge) if p.get("id") else None,
                        p.get("fullName"))

            (hsp, hname), (asp, aname) = sp(h), sp(a)
            status = g.get("status") or {}
            out.append({
                "game_pk": g.get("gamePk"),
                "game_key": f"{home}{day.replace('-', '')}{num}",
                "game_number": num,
                "start_utc": g.get("gameDate"),
                "state": status.get("abstractGameState"),
                "detailed_state": status.get("detailedState"),
                "home_team": home, "away_team": away,
                "home_name": h["team"].get("name"), "away_name": a["team"].get("name"),
                "home_sp": hsp, "away_sp": asp,
                "home_sp_name": hname, "away_sp_name": aname,
                "park": parks.get(home, "UNK"),
            })
    return out


def last_final_date(payload: dict, before: str) -> str | None:
    """Latest date strictly before `before` with a regular-season final."""
    best = None
    for d in payload.get("dates", []):
        if d.get("date", "") >= before:
            continue
        if any((g.get("status") or {}).get("abstractGameState") == "Final"
               for g in d.get("games", [])):
            best = max(best or "", d["date"])
    return best


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import requests

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=date.today().isoformat())
    a = p.parse_args(argv)
    day = a.date
    lo = (date.fromisoformat(day) - timedelta(days=10)).isoformat()
    prev_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()

    today = requests.get(URL.format(a=day, b=day), timeout=30).json()
    back = requests.get(URL.format(a=lo, b=prev_day), timeout=30).json()
    hist = pd.read_csv(ROOT / "model" / "mlb_schedule_cache.csv")
    games = parse_slate(today, day, load_id_bridge(), modal_parks(hist))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"date": day, "fetched_at": stamp,
           "previous_final_date": last_final_date(back, day), "games": games}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{day}-{stamp}.json"
    dest.write_text(json.dumps(out, indent=1))
    missing = sum(1 for g in games if not (g["home_sp"] and g["away_sp"]))
    print(f"{day}: {len(games)} games, {missing} without both probables -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
