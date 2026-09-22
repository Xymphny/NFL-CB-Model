"""Odds API team name -> ESPN abbreviation, which is what the NBA model keys on.

NOT YET VERIFIED AGAINST THE ODDS API. Names come from sportsdataverse's ESPN
schedules, filtered to the same rows the fit uses (season_type == 2 and
type_abbreviation == "STD"), so the All-Star and exhibition sides (CAN, CHK,
KEN, SHQ) cannot appear. The matcher RAISES on any unknown spelling.

ALIASES, flagged because they were not observed in a feed:
  Los Angeles Clippers -- ESPN writes "LA Clippers". There is one Clippers, so
  this cannot match the wrong team; at worst it is never used.
"""

from coverline.execution.matching import TeamTable

NBA_NAMES: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GS",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO",
    "New York Knicks": "NY",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
    # unverified alias, see docstring
    "Los Angeles Clippers": "LAC",
}

TABLE = TeamTable("nba", NBA_NAMES, verified_against_odds_api=False)
