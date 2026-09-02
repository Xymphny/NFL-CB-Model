"""
Weather fetching for upcoming games. Currently the model defaults wind
to 0 for any future game (confirmed earlier: nflverse's schedule data
has real wind for PAST games but NaN for future ones, which is
expected -- weather isn't knowable that far ahead). This fetches a
real forecast close to kickoff instead of defaulting to zero.

UNTESTED: api.weather.gov is not on this sandbox's allowed network
list (same category of limitation as ESPN scraping and the Odds API --
this environment's network access is deliberately restricted). The
stadium coordinate table below is real, stable data with no dependency
on live network access; only fetch_forecast() itself is unverified.
Test this directly once deployed with real network access.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

STADIUM_LOCATIONS = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "dome": True},
    "ATL": {"lat": 33.7554, "lon": -84.4008, "dome": True},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "dome": False},
    "BUF": {"lat": 42.7738, "lon": -78.7870, "dome": False},
    "CAR": {"lat": 35.2258, "lon": -80.8528, "dome": False},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "dome": False},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "dome": False},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "dome": False},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "dome": True},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "dome": False},
    "DET": {"lat": 42.3400, "lon": -83.0456, "dome": True},
    "GB": {"lat": 44.5013, "lon": -88.0622, "dome": False},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "dome": True},
    "IND": {"lat": 39.7601, "lon": -86.1639, "dome": True},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "dome": False},
    "KC": {"lat": 39.0489, "lon": -94.4839, "dome": False},
    "LA": {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "LAC": {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "LV": {"lat": 36.0909, "lon": -115.1833, "dome": True},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "dome": False},
    "MIN": {"lat": 44.9738, "lon": -93.2575, "dome": True},
    "NE": {"lat": 42.0909, "lon": -71.2643, "dome": False},
    "NO": {"lat": 29.9511, "lon": -90.0812, "dome": True},
    "NYG": {"lat": 40.8135, "lon": -74.0745, "dome": False},
    "NYJ": {"lat": 40.8135, "lon": -74.0745, "dome": False},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "dome": False},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "dome": False},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "dome": False},
    "SF": {"lat": 37.4030, "lon": -121.9694, "dome": False},
    "TB": {"lat": 27.9759, "lon": -82.5033, "dome": False},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "dome": False},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "dome": False},
}


def fetch_forecast(home_team):
    if home_team not in STADIUM_LOCATIONS:
        return {"wind": 0.0, "error": f"no stadium location for {home_team}"}

    location = STADIUM_LOCATIONS[home_team]
    if location["dome"]:
        return {"wind": 0.0, "dome": True}

    try:
        points_resp = requests.get(
            f"https://api.weather.gov/points/{location['lat']},{location['lon']}",
            headers={"User-Agent": "football-model (contact via repo)"},
            timeout=10,
        )
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecastHourly"]

        forecast_resp = requests.get(forecast_url, headers={"User-Agent": "football-model"}, timeout=10)
        forecast_resp.raise_for_status()
        current_period = forecast_resp.json()["properties"]["periods"][0]

        import re
        wind_str = current_period.get("windSpeed", "0 mph")
        match = re.search(r"\d+", wind_str)
        wind_mph = float(match.group()) if match else 0.0

        return {"wind": wind_mph, "dome": False, "raw_forecast": current_period.get("shortForecast")}

    except Exception as e:
        print(f"[weather] fetch failed for {home_team}: {e}")
        return {"wind": 0.0, "error": str(e)}


if __name__ == "__main__":
    print("Testing weather fetch -- expect this to fail in this sandbox (weather.gov not reachable):")
    for team in ["GB", "MIA", "DET"]:
        result = fetch_forecast(team)
        print(f"  {team}: {result}")
