"""
Real 2026 CFB returning production (total snaps returned from 2025),
gathered from CBS Sports (TruMedia data, published Aug 22, 2026) --
covers all 138 FBS teams in one real, comprehensive source.

This is a CFB-native alternative to NFL's preseason mechanism, since
CFB has no equivalent to NFL's actual preseason GAMES (backups
playing real snaps) -- there's no real-game data to lean on before
the season starts. Returning production instead directly measures
what NFL's preseason signal only measures indirectly: how much of a
team's real, productive roster carried over.

Directly validates the exact problem this was built to solve: Indiana
(44%, No. 49) and Miami (46%, No. 41) -- the two teams that played in
the actual 2025 national championship game -- both show LOW returning
production despite being the best two teams last year. Using last
season's rating for them without this adjustment would repeat the
same mistake found with Ohio State, just in the opposite direction.

Real names below reconciled with cfbfastR's convention (checked
against real 2025 CFB ratings output).
"""

RETURNING_PRODUCTION_2026 = {
    "Notre Dame": 0.66, "Virginia Tech": 0.62, "Georgia": 0.61, "BYU": 0.60,
    "Maryland": 0.60, "Stanford": 0.59, "New Mexico": 0.59, "Texas": 0.57,
    "Delaware": 0.57, "Air Force": 0.57, "USC": 0.56, "Ohio State": 0.56,
    "Nebraska": 0.55, "Oklahoma": 0.55, "Northwestern": 0.54, "Minnesota": 0.54,
    "Washington": 0.54, "Oregon": 0.54, "Army": 0.54, "Tennessee": 0.53,
    "Texas Tech": 0.52, "Navy": 0.52, "Eastern Michigan": 0.52, "Boise State": 0.52,
    "Florida Atlantic": 0.52, "Fresno State": 0.51, "Pittsburgh": 0.51,
    "Florida": 0.51, "North Dakota State": 0.51, "Michigan": 0.51, "Clemson": 0.50,
    "Houston": 0.50, "Temple": 0.50, "South Carolina": 0.49, "Arizona": 0.49,
    "Miami (OH)": 0.49, "Arkansas State": 0.49, "Louisiana": 0.48, "SMU": 0.46,
    "San Diego State": 0.46, "Miami": 0.46, "Texas A&M": 0.46, "Vanderbilt": 0.46,
    "Texas State": 0.45, "UTSA": 0.45, "Liberty": 0.45, "Syracuse": 0.44,
    "Louisiana Tech": 0.44, "Indiana": 0.44, "Ole Miss": 0.44, "NC State": 0.44,
    "Tulsa": 0.44, "Utah State": 0.43, "TCU": 0.43, "Western Michigan": 0.43,
    "Missouri": 0.43, "Virginia": 0.43, "UCF": 0.42, "Kansas": 0.42,
    "Mississippi State": 0.42, "Marshall": 0.42, "California": 0.41, "Akron": 0.41,
    "Alabama": 0.41, "Wake Forest": 0.41, "Duke": 0.41, "Central Michigan": 0.41,
    "Jacksonville State": 0.40, "Hawai'i": 0.40, "Kansas State": 0.40,
    "Kent State": 0.40, "UCLA": 0.40, "UNLV": 0.39, "Iowa": 0.39,
    "Boston College": 0.39, "Utah": 0.39, "South Alabama": 0.38,
    "New Mexico State": 0.38, "Georgia Tech": 0.38, "UL Monroe": 0.37,
    "Florida International": 0.37, "Purdue": 0.37, "Sam Houston": 0.36,
    "Rutgers": 0.36, "Louisville": 0.36, "Oregon State": 0.36, "Rice": 0.36,
    "Georgia Southern": 0.35, "Tulane": 0.35, "Illinois": 0.35, "LSU": 0.34,
    "Kennesaw State": 0.34, "Nevada": 0.34, "Cincinnati": 0.33,
    "Georgia State": 0.33, "Wisconsin": 0.32, "Troy": 0.32, "Florida State": 0.32,
    "Missouri State": 0.32, "Wyoming": 0.32, "Old Dominion": 0.31,
    "Arizona State": 0.31, "Massachusetts": 0.30, "Bowling Green": 0.30,
    "Middle Tennessee": 0.30, "Baylor": 0.30, "Arkansas": 0.29, "Ohio": 0.29,
    "Charlotte": 0.29, "Auburn": 0.29, "North Carolina": 0.28, "Ball State": 0.28,
    "Coastal Carolina": 0.27, "Northern Illinois": 0.27, "Kentucky": 0.27,
    "Washington State": 0.27, "Michigan State": 0.27, "East Carolina": 0.26,
    "Colorado State": 0.25, "Sacramento State": 0.25, "UTEP": 0.24, "Buffalo": 0.24,
    "App State": 0.24, "Western Kentucky": 0.23, "South Florida": 0.22,
    "Penn State": 0.22, "Colorado": 0.21, "UAB": 0.21, "San Jose State": 0.20,
    "West Virginia": 0.19, "James Madison": 0.19, "Toledo": 0.17,
    "Oklahoma State": 0.11, "Memphis": 0.10, "Iowa State": 0.10,
    "Southern Miss": 0.10, "UConn": 0.07, "North Texas": 0.07,
}

GATHERED_AT = "2026-08-22"
SOURCE = "CBS Sports (TruMedia data)"


if __name__ == "__main__":
    print(f"{len(RETURNING_PRODUCTION_2026)} real teams with 2026 returning production data")
    print(f"Highest: {max(RETURNING_PRODUCTION_2026, key=RETURNING_PRODUCTION_2026.get)} "
          f"({max(RETURNING_PRODUCTION_2026.values()):.0%})")
    print(f"Lowest: {min(RETURNING_PRODUCTION_2026, key=RETURNING_PRODUCTION_2026.get)} "
          f"({min(RETURNING_PRODUCTION_2026.values()):.0%})")
    print(f"Indiana (real 2025 champion): {RETURNING_PRODUCTION_2026.get('Indiana'):.0%}")
    print(f"Miami (real 2025 runner-up): {RETURNING_PRODUCTION_2026.get('Miami'):.0%}")
