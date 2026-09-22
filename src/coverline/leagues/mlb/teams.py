"""Odds API team name -> Retrosheet code, which is what the MLB model keys on.

VERIFIED AGAINST THE ODDS API. Every name below was observed in a real Odds
API payload -- the committed data/mlb_divergence/*.json files carry the feed's
own spelling next to the code the legacy board resolved it to. The code is then
mapped statsapi -> Retrosheet through STATS_TO_RETRO in deploy/mlb_daily_update.py,
the table the schedule cache itself was built with. tests/core/test_matching.py
rebuilds this dict from those files and fails if they disagree.
"""

from coverline.execution.matching import TeamTable

MLB_NAMES: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHN",
    "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA",
    "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYN",
    "New York Yankees": "NYA",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDN",
    "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}

TABLE = TeamTable("mlb", MLB_NAMES, verified_against_odds_api=True)
