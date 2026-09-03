"""
Travel distance -- a real, documented gap found by comparing this
model against ESPN FPI's own published methodology: FPI explicitly
includes travel distance ("every additional 1,000 miles traveled more
than your opponent costs about a point... Seattle to Miami is worth
about half a point per game"). This model had no travel feature at
all until now, despite already having real stadium coordinates built
for model/weather.py.

Uses the haversine formula on those same real coordinates -- no new
data source needed, just a feature this model was missing entirely.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.weather import STADIUM_LOCATIONS


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3959
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def travel_distance(away_team, home_team):
    if away_team not in STADIUM_LOCATIONS or home_team not in STADIUM_LOCATIONS:
        return 0.0
    away_loc = STADIUM_LOCATIONS[away_team]
    home_loc = STADIUM_LOCATIONS[home_team]
    return haversine_miles(away_loc["lat"], away_loc["lon"], home_loc["lat"], home_loc["lon"])


if __name__ == "__main__":
    print("Sanity check against FPI's own stated example: Seattle to Miami")
    dist = travel_distance("MIA", "SEA")
    print(f"  MIA @ SEA: {dist:.0f} miles (FPI cites this as their extreme-case example)")
    print()
    print("A few other real distances:")
    for away, home in [("NYJ", "SEA"), ("MIA", "NE"), ("LAC", "MIA"), ("SF", "NYG")]:
        print(f"  {away} @ {home}: {travel_distance(away, home):.0f} miles")
