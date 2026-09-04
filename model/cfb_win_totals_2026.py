"""
Real 2026 CFB win totals, gathered from sportsbettingdime.com (odds as
of July 29, 2026 at DraftKings/FanDuel/BetMGM) -- the actual fix for
the CFB preseason-prior problem documented in
model/cfb_preseason_prior.py as unsolved: neither last-season DVOA nor
multi-year Elo can see real, current roster composition (confirmed via
Ohio State's real 47-of-91 scholarship turnover), but the real market
already prices this in, the same way NFL's Vegas win totals do.

CONFIRMS a real correction needed elsewhere in this project: this
source states directly "Indiana went 16-0 and won the national title
last season" -- Indiana, not Ohio State, won the actual January 2026
CFP championship (a perfect 16-0 season). Earlier documentation in
this project incorrectly attributed a January 2026 title to Ohio
State, conflating it with their real 2024 season championship.
"""

WIN_TOTALS_2026 = {
    "Air Force": 7.5, "Akron": 4.5, "Alabama": 8.5, "App State": 5.5,
    "Arizona": 7.5, "Arizona State": 6.5, "Arkansas": 4.5, "Arkansas State": 6.5,
    "Army": 7.5, "Auburn": 6.5, "BYU": 8.5, "Ball State": 3.5, "Baylor": 6.5,
    "Boise State": 7.5, "Boston College": 3.5, "Bowling Green": 4.5, "Buffalo": 5.5,
    "California": 6.5, "Central Michigan": 6.5, "Charlotte": 2.5, "Cincinnati": 5.5,
    "Clemson": 7.5, "Coastal Carolina": 4.5, "Colorado": 4.5, "Colorado State": 3.5,
    "Delaware": 6.5, "Duke": 5.5, "East Carolina": 7.5, "Eastern Michigan": 5.5,
    "Florida International": 6.5, "Florida": 7.5, "Florida Atlantic": 5.5, "Florida State": 6.5,
    "Fresno State": 6.5, "Georgia": 9.5, "Georgia Southern": 4.5, "Georgia State": 4.5,
    "Georgia Tech": 6.5, "Hawai'i": 7.5, "Houston": 8.5, "Illinois": 7.5,
    "Indiana": 10.5, "Iowa": 7.5, "Iowa State": 4.5, "Jacksonville State": 7.5,
    "James Madison": 8.5, "Kansas": 5.5, "Kansas State": 8.5, "Kennesaw State": 6.5,
    "Kent State": 3.5, "Kentucky": 4.5, "LSU": 8.5, "Liberty": 8.5,
    "Louisiana": 7.5, "Louisiana Tech": 5.5, "Louisville": 8.5, "Marshall": 7.5,
    "Maryland": 5.5, "Memphis": 7.5, "Miami": 10.5,
    "Miami (OH)": 7.5,
    "Michigan": 8.5, "Michigan State": 4.5, "Middle Tennessee": 3.5, "Minnesota": 6.5,
    "Mississippi State": 4.5, "Missouri": 6.5, "Missouri State": 4.5, "NC State": 7.5,
    "Navy": 7.5, "Nebraska": 6.5, "Nevada": 4.5, "New Mexico": 7.5,
    "New Mexico State": 4.5, "North Carolina": 4.5, "North Texas": 5.5,
    "Northern Illinois": 3.5, "Northwestern": 5.5, "Notre Dame": 11.5, "Ohio": 6.5,
    "Ohio State": 9.5, "Oklahoma": 7.5, "Oklahoma State": 6.5, "Old Dominion": 7.5,
    "Ole Miss": 7.5, "Oregon": 10.5, "Oregon State": 4.5, "Penn State": 8.5,
    "Pittsburgh": 7.5, "Purdue": 3.5, "Rice": 3.5, "Rutgers": 4.5, "SMU": 8.5,
    "Sacramento State": 4.5, "Sam Houston": 3.5, "San Diego State": 6.5,
    "San Jose State": 4.5, "South Alabama": 5.5, "South Carolina": 6.5,
    "South Florida": 8.5, "Southern Miss": 3.5, "Stanford": 3.5, "Syracuse": 4.5,
    "TCU": 6.5, "Temple": 5.5, "Tennessee": 7.5, "Texas": 9.5, "Texas A&M": 8.5,
    "Texas State": 6.5, "Texas Tech": 10.5, "Toledo": 7.5, "Troy": 6.5,
    "Tulane": 7.5, "Tulsa": 5.5, "UAB": 3.5, "UCF": 5.5, "UCLA": 6.5,
    "UConn": 5.5, "UL Monroe": 3.5,
    "Massachusetts": 2.5, "UNLV": 7.5, "USC": 8.5, "UTEP": 3.5, "UTSA": 7.5,
    "Utah": 8.5, "Utah State": 4.5, "Vanderbilt": 5.5, "Virginia": 7.5,
    "Virginia Tech": 6.5, "Wake Forest": 5.5, "Washington": 7.5,
    "Washington State": 4.5, "West Virginia": 5.5, "Western Kentucky": 6.5,
    "Western Michigan": 7.5, "Wisconsin": 6.5, "Wyoming": 5.5,
}

GATHERED_AT = "2026-07-29"
SOURCE = "sportsbettingdime.com (DraftKings/FanDuel/BetMGM consensus)"

# HONEST NOTE: team names above use cfbfastR's real naming convention
# (confirmed by checking real 2025 CFB ratings output), not the
# source's own labels -- e.g. "App State" not "Appalachian State",
# "Hawai'i" not "Hawaii", "Massachusetts" not "UMass", "Florida
# International" not "FIU". Two teams from the source (Sacramento
# State, San Jose State) were NOT found under any name in the real
# 2025 CFB ratings data -- left as-is here (their real names, if they
# differ) rather than guess a mapping; they'll simply not match when
# blending until confirmed.


if __name__ == "__main__":
    print(f"{len(WIN_TOTALS_2026)} real teams with 2026 win totals")
    print(f"Highest: {max(WIN_TOTALS_2026, key=WIN_TOTALS_2026.get)} ({max(WIN_TOTALS_2026.values())})")
    print(f"Lowest: {min(WIN_TOTALS_2026, key=WIN_TOTALS_2026.get)} ({min(WIN_TOTALS_2026.values())})")
