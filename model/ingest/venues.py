#!/usr/bin/env python3
"""Home venues for the board's venue and travel lines. Display only.

Writes data/venues.json:

    {"nba": {CODE: {name, city, state, lat, lon}}, "nhl": {...},
     "nfl": {CODE: {lat, lon}}}

NBA and NHL arenas come from ESPN's team records (names change with naming
rights, so they are pulled, not typed); NFL stadium names come from nflverse
games.csv per game at export time, and only NFL coordinates are kept here.

COORDINATES ARE CITY-LEVEL, from CITY_COORDS below: good to a few miles,
which is what a travel distance needs and all it is used for. Travel is never
a model input (in_price false everywhere). A city missing from the table is
an error, not a guess.

Team keys are the codes each league's model uses (coverline.leagues.*),
so the board can look venues up by the codes on its games.

    python model/ingest/venues.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data" / "venues.json"
TEAMS = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams"
TEAM = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{id}"

#: "City, ST" -> (lat, lon), city centres.
CITY_COORDS = {
    "Atlanta, GA": (33.749, -84.388), "Boston, MA": (42.360, -71.059),
    "Brooklyn, NY": (40.678, -73.944), "Charlotte, NC": (35.227, -80.843),
    "Chicago, IL": (41.878, -87.630), "Cleveland, OH": (41.499, -81.694),
    "Dallas, TX": (32.777, -96.797), "Denver, CO": (39.739, -104.990),
    "Detroit, MI": (42.331, -83.046), "San Francisco, CA": (37.775, -122.419),
    "Houston, TX": (29.760, -95.370), "Indianapolis, IN": (39.768, -86.158),
    "Los Angeles, CA": (34.052, -118.244), "Inglewood, CA": (33.962, -118.353),
    "Memphis, TN": (35.150, -90.049), "Miami, FL": (25.762, -80.192),
    "Milwaukee, WI": (43.039, -87.906), "Minneapolis, MN": (44.978, -93.265),
    "New Orleans, LA": (29.951, -90.072), "New York, NY": (40.713, -74.006),
    "Oklahoma City, OK": (35.468, -97.516), "Orlando, FL": (28.538, -81.379),
    "Philadelphia, PA": (39.953, -75.165), "Phoenix, AZ": (33.448, -112.074),
    "Portland, OR": (45.515, -122.679), "Sacramento, CA": (38.582, -121.494),
    "San Antonio, TX": (29.424, -98.494), "Toronto, ON": (43.653, -79.383),
    "Salt Lake City, UT": (40.761, -111.891), "Washington, DC": (38.907, -77.037),
    "Anaheim, CA": (33.836, -117.914), "Buffalo, NY": (42.886, -78.878),
    "Calgary, AB": (51.045, -114.057), "Raleigh, NC": (35.780, -78.639),
    "Columbus, OH": (39.961, -82.999), "Edmonton, AB": (53.546, -113.494),
    "Sunrise, FL": (26.134, -80.114), "St. Paul, MN": (44.954, -93.090),
    "Saint Paul, MN": (44.954, -93.090), "Montreal, QC": (45.502, -73.567),
    "Montréal, QC": (45.502, -73.567), "Nashville, TN": (36.163, -86.782),
    "Newark, NJ": (40.736, -74.172), "Elmont, NY": (40.700, -73.713),
    "Ottawa, ON": (45.297, -75.927), "Pittsburgh, PA": (40.441, -79.996),
    "San Jose, CA": (37.339, -121.894), "Seattle, WA": (47.606, -122.332),
    "St. Louis, MO": (38.627, -90.199), "Tampa, FL": (27.951, -82.458),
    "Vancouver, BC": (49.283, -123.121), "Las Vegas, NV": (36.170, -115.140),
    "Winnipeg, MB": (49.895, -97.138), "Oakland, CA": (37.804, -122.271),
    "Glendale, AZ": (33.539, -112.186), "Tempe, AZ": (33.425, -111.940),
}

LEAGUES = {"nba": ("basketball", "nba"), "nhl": ("hockey", "nhl")}

#: When ESPN's venue record carries no address, the team's location names
#: the city -- except for regional names, mapped here.
LOCATION_CITY = {
    "LA": "Los Angeles", "Golden State": "San Francisco", "Utah": "Salt Lake City",
    "Minnesota": "Minneapolis", "Indiana": "Indianapolis", "New Jersey": "Newark",
    "Tampa Bay": "Tampa", "Carolina": "Raleigh", "Florida": "Sunrise",
    "Colorado": "Denver", "Vegas": "Las Vegas", "Washington": "Washington",
    "Arizona": "Tempe", "New York": "New York",
}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def code_of(league: str, display: str) -> str:
    if league == "nba":
        from coverline.leagues.nba.teams import NBA_NAMES
        return NBA_NAMES[display]
    from coverline.leagues.nhl.teams import TABLE
    return TABLE.to_code(display)


def venue_row(team: dict) -> dict:
    """ESPN team record -> {name, city, state, lat, lon}. Pure; raises on a
    city the table does not hold."""
    v = team.get("venue") or (team.get("franchise") or {}).get("venue") or {}
    addr = v.get("address") or {}
    city, state = addr.get("city"), addr.get("state") or addr.get("country")
    key = f"{city}, {state}"
    if not city:
        # No address on the record: the team's location names the city.
        city = LOCATION_CITY.get(team.get("location"), team.get("location"))
        hits = [k for k in CITY_COORDS if k.split(",")[0] == city]
        key = hits[0] if len(hits) == 1 else f"{city}, ?"
        state = key.split(", ")[1]
    if key not in CITY_COORDS:
        raise KeyError(f"no coordinates for {key!r} ({v.get('fullName')}); add it to CITY_COORDS")
    lat, lon = CITY_COORDS[key]
    return {"name": v.get("fullName"), "city": city, "state": state, "lat": lat, "lon": lon}


SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
              "?dates={day}&limit=100")
#: The last weeks of the 2025-26 regular seasons. Where teams actually
#: PLAYED: ESPN's franchise records are stale for some (one still names an
#: arena its team left in 2012), so the venue is read off real home games.
WINDOW = {"nba": ("2026-03-15", "2026-04-12"), "nhl": ("2026-03-15", "2026-04-16")}


def home_venues(events: list[dict]) -> dict[str, dict]:
    """Scoreboard events -> {home team display name: venue}, the LAST home
    game winning. Pure."""
    out: dict[str, dict] = {}
    for e in sorted(events, key=lambda e: e.get("date", "")):
        comp = (e.get("competitions") or [{}])[0]
        if comp.get("neutralSite"):
            continue
        home = next((t["team"] for t in comp.get("competitors", [])
                     if t.get("homeAway") == "home"), None)
        if home and comp.get("venue"):
            out[home["displayName"]] = comp["venue"]
    return out


def pull(league: str) -> dict:
    from datetime import date, timedelta
    sport, lg = LEAGUES[league]
    lo, hi = (date.fromisoformat(x) for x in WINDOW[league])
    events, d = [], lo
    while d <= hi:
        events += _get(SCOREBOARD.format(sport=sport, league=lg, day=d.strftime("%Y%m%d"))).get("events", [])
        d += timedelta(days=1)
    out = {}
    for display, venue in home_venues(events).items():
        out[code_of(league, display)] = venue_row({"venue": venue})
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    from deploy.game_context import STADIUM_COORDS
    art = {"_about": ("Home venues for display and travel distance only; never a model "
                      "input. NBA/NHL arenas from ESPN team records, city-level "
                      "coordinates. Written by model/ingest/venues.py."),
           "nba": pull("nba"), "nhl": pull("nhl"),
           "nfl": {k: {"lat": v[0], "lon": v[1]} for k, v in sorted(STADIUM_COORDS.items())}}
    OUT.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: nba {len(art['nba'])}, nhl {len(art['nhl'])}, "
          f"nfl {len(art['nfl'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
