"""
Per-game context for the board: the full injury report for both teams
and a kickoff-hour weather forecast for outdoor stadiums -- the
clear-cut situational facts a bettor checks before firing, shown next
to the number instead of six browser tabs away.

Honest framing baked into the design: the residual-model experiment
showed weather and injuries are PRICED IN at the closing line (wind
coefficients even flipped sign between train and held-out). This is
transparency and context, not edge -- it explains WHY a line sits
where it sits and helps the person sanity-check a flag, and that's
the claim, nothing more.

Sources:
- Injuries: nflverse injuries parquet (free, updated through the week;
  same feed deploy/qb_status.py already uses). All positions, latest
  report status.
- Weather: Open-Meteo forecast API -- free, NO API KEY, generous
  limits. Called only for outdoor/open-roof stadiums at the game's
  kickoff hour. NOT testable from the build sandbox (domain not
  reachable there); response parsing is unit-tested against the
  documented response shape, and the whole fetch is soft-fail.

Stadium coordinates are a static table -- stadiums don't move.
"""

import os
import json
from datetime import datetime

import pandas as pd
import requests

INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2}
MAX_LISTED = 8

# Home-team -> (lat, lon). Shared-stadium pairs repeat coordinates.
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3240, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9737, -93.2577),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}


def get_injury_report(season, week, injuries=None):
    """{team: [{player, position, status}, ...]} sorted worst-first,
    capped at MAX_LISTED per team."""
    try:
        if injuries is None:
            injuries = pd.read_parquet(INJURIES_URL.format(season=season))
        wk = injuries[(injuries["week"] == week) & injuries["report_status"].isin(STATUS_ORDER)]
        report = {}
        for team, grp in wk.groupby("team"):
            rows = sorted(
                ({"player": r["full_name"], "position": r["position"], "status": r["report_status"]}
                 for _, r in grp.iterrows()),
                key=lambda r: (STATUS_ORDER[r["status"]], r["position"] != "QB"),
            )
            report[team] = rows[:MAX_LISTED]
        return report
    except Exception as e:
        print(f"[game_context] injury report soft-fail: {e}")
        return {}


def parse_open_meteo(payload, kickoff_hour_iso):
    """Pure parsing, unit-testable: pick the forecast row at (or nearest
    after) kickoff hour from Open-Meteo's documented hourly shape."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None
    idx = next((i for i, t in enumerate(times) if t >= kickoff_hour_iso), len(times) - 1)
    return {
        "temp_f": hourly.get("temperature_2m", [None] * len(times))[idx],
        "wind_mph": hourly.get("wind_speed_10m", [None] * len(times))[idx],
        "precip_prob": hourly.get("precipitation_probability", [None] * len(times))[idx],
        "forecast_hour": times[idx],
    }


def get_weather(lat, lon, kickoff_hour_iso):
    """Soft-fail Open-Meteo fetch for one stadium/kickoff."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "forecast_days": 8, "timezone": "America/New_York",
            }, timeout=15)
        resp.raise_for_status()
        return parse_open_meteo(resp.json(), kickoff_hour_iso)
    except Exception as e:
        print(f"[game_context] weather soft-fail ({lat},{lon}): {e}")
        return None


def attach_context(divergences, season, week, games=None):
    """Adds 'injuries' and 'weather' to each divergence row in place.
    Weather only for outdoor/open roofs per the schedule; domes get
    {'roof': ...} so the frontend can say 'indoors' instead of nothing."""
    injuries = get_injury_report(season, week)

    kickoff, roofs = {}, {}
    try:
        if games is None:
            games = pd.read_csv("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
        sched = games[(games["season"] == season) & (games["week"] == week)]
        for _, gm in sched.iterrows():
            key = (gm["home_team"], gm["away_team"])
            roofs[key] = gm.get("roof")
            if pd.notna(gm.get("gameday")) and pd.notna(gm.get("gametime")):
                kickoff[key] = f"{gm['gameday']}T{str(gm['gametime'])[:2]}:00"
    except Exception as e:
        print(f"[game_context] schedule soft-fail: {e}")

    weather_cache = {}
    for d in divergences:
        key = (d["home_team"], d["away_team"])
        d["injuries"] = {
            "home": injuries.get(d["home_team"], []),
            "away": injuries.get(d["away_team"], []),
        }
        roof = roofs.get(key)
        if roof is not None and not isinstance(roof, str):
            roof = None  # pandas NaN from a schedule row with no roof value
        if roof in ("dome", "closed"):
            d["weather"] = {"roof": roof}
        elif d["home_team"] in STADIUM_COORDS and key in kickoff:
            ck = (d["home_team"], kickoff[key])
            if ck not in weather_cache:
                lat, lon = STADIUM_COORDS[d["home_team"]]
                weather_cache[ck] = get_weather(lat, lon, kickoff[key])
            wx = weather_cache[ck]
            d["weather"] = {"roof": roof or "outdoors", **wx} if wx else {"roof": roof or "outdoors"}
        else:
            d["weather"] = None
    return divergences
