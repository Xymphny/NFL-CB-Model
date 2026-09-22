"""Odds API team name -> NHL code, as used by the NHL API and sportsdataverse.

NOT YET VERIFIED AGAINST THE ODDS API. No committed file carries an Odds API
NHL payload, so these names come from the league's own data. The matcher RAISES
on any name it does not know; on the first live run, an unknown spelling is the
expected outcome, and the fix is to add that spelling here deliberately.

Accents and full stops are canonicalised away, so "Montreal Canadiens" and
"St Louis Blues" already resolve.

ALIASES, each flagged because it was not observed in a feed:
  Utah Mammoth -- the franchise played 2024-25 as "Utah Hockey Club" and was
  renamed for 2025-26. Its NHL code is assumed to remain UTA. If the league
  changed the code as well, Utah games will be refused as having no model
  game, never matched to a different team, because UTA is the only Utah code.
"""

from coverline.execution.matching import TeamTable

NHL_NAMES: dict[str, str] = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    # unverified alias, see docstring
    "Utah Mammoth": "UTA",
}

TABLE = TeamTable("nhl", NHL_NAMES, verified_against_odds_api=False)
