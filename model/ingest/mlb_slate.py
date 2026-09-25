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
regular-season game went final, read from the league's own schedule, and
`previous_final_keys` are the game keys of every game played to a result on
it. The live source refuses unless EVERY one of those games is in the
results cache. An All-Star break is therefore not mistaken for a stale cache,
and a stale cache is not mistaken for an off day.

A DATE IS NOT ENOUGH, and the first version checked only the date. Run at
9:40pm with five of fifteen games final, the cron's update wrote those five,
the cache's last date became today's, and the check passed -- pricing
tomorrow from ratings missing ten results. `unfinished_keys` closes the other
half: games still in progress when the slate was pulled mean the slate's own
record of what went final is incomplete, and the live source refuses it until
it is re-pulled.

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
SEASON_URL = "https://statsapi.mlb.com/api/v1/seasons/{y}?sportId=1"
SEASON_DIR = ROOT / "data" / "raw" / "mlb"


def season_dates(payload: dict) -> dict | None:
    """The regular season's first and last dates from MLB's seasons endpoint.
    Pure. The capture job reads the result so it stops buying postseason
    odds that no model prices (this ingest is regular season only)."""
    for s in payload.get("seasons", []):
        lo, hi = s.get("regularSeasonStartDate"), s.get("regularSeasonEndDate")
        if lo and hi:
            return {"season": int(s.get("seasonId") or lo[:4]),
                    "regular_season_start": lo, "regular_season_end": hi}
    return None


def update_season(year: int, get=None, out_dir: Path = SEASON_DIR) -> bool:
    """Write data/raw/mlb/season_{year}.json when its content changes."""
    import requests
    get = get or (lambda url: requests.get(url, timeout=30).json())
    dates = season_dates(get(SEASON_URL.format(y=year)))
    if dates is None:
        return False
    path = out_dir / f"season_{year}.json"
    text = json.dumps(dates, indent=1) + "\n"
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True
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


def _played(g: dict) -> bool:
    """Final WITH a result. The API marks a postponement abstractGameState
    Final too, with no score -- the cron writes those as rows with empty
    scores, and they are not results."""
    st = g.get("status") or {}
    if st.get("abstractGameState") != "Final":
        return False
    if any(w in str(st.get("detailedState", "")).lower()
           for w in ("postpon", "cancel", "suspend")):
        return False
    return g["teams"]["home"].get("score") is not None


def _key(g: dict, day: str) -> str:
    return f"{_team(g['teams']['home'])}{day.replace('-', '')}{game_number_of(g)}"


def finals_on(payload: dict, day: str) -> list[str]:
    """Keys of games played to a result on `day`, as the cron will key them."""
    return sorted(_key(g, day) for d in payload.get("dates", [])
                  if d.get("date") == day for g in d.get("games", []) if _played(g))


def unfinished(payload: dict, before: str) -> list[str]:
    """Keys of games before `before` that were still in progress."""
    return sorted(_key(g, d["date"]) for d in payload.get("dates", [])
                  if d.get("date", "") < before for g in d.get("games", [])
                  if (g.get("status") or {}).get("abstractGameState") == "Live")


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import requests

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=date.today().isoformat())
    a = p.parse_args(argv)
    day = a.date
    lo = (date.fromisoformat(day) - timedelta(days=10)).isoformat()
    prev_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()

    try:
        if update_season(int(day[:4])):
            print(f"season_{day[:4]}.json updated")
    except Exception as exc:              # never costs the slate pull
        print(f"season dates unavailable: {type(exc).__name__}: {exc}")

    today = requests.get(URL.format(a=day, b=day), timeout=30).json()
    back = requests.get(URL.format(a=lo, b=prev_day), timeout=30).json()
    hist = pd.read_csv(ROOT / "model" / "mlb_schedule_cache.csv")
    games = parse_slate(today, day, load_id_bridge(), modal_parks(hist))

    if not games:
        # Off day or off-season. Nothing to price, and a file per empty day
        # would bury the pulls that matter.
        print(f"{day}: no regular-season games; nothing written")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prev = last_final_date(back, day)
    out = {"date": day, "fetched_at": stamp,
           "previous_final_date": prev,
           "previous_final_keys": finals_on(back, prev) if prev else [],
           "unfinished_keys": unfinished(back, day),
           "games": games}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{day}-{stamp}.json"
    dest.write_text(json.dumps(out, indent=1))
    missing = sum(1 for g in games if not (g["home_sp"] and g["away_sp"]))
    print(f"{day}: {len(games)} games, {missing} without both probables -> {dest}")
    if out["unfinished_keys"]:
        print(f"  {len(out['unfinished_keys'])} earlier game(s) still in progress; "
              "the live source will refuse this slate until it is re-pulled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
