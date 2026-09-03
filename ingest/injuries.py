"""
Real, official weekly NFL injury reports -- closes a real, specific gap
identified this session: the market has live game-week injury status
(Questionable/Doubtful/Out) that this model has never had access to.
Confirmed via direct search that Big Balls Sports Data (considered as
an alternative) does NOT have this for NFL yet ("Injury reports are
coming soon", per their own docs) -- this uses nflverse's own real
injury report release instead, the same trusted source everything
else in this project is built on.

Real, structured data: player name, position, specific injury type,
practice participation, and official report status. Confirmed real
via a direct check against a known 2023 case: Deshaun Watson (CLE)
correctly shows up as "Out" with "right Shoulder" for weeks 6 and 8,
matching his real 2023 shoulder injury.

HONEST SCOPE NOTE, found while validating: this format tracks
practice-participation status ahead of an UPCOMING game (is a player
expected to play THIS week), not a running injured-reserve log. A
player whose season-ending injury happened mid-game (like Kirk
Cousins' 2023 Achilles tear) may not show up in this data at all,
since there's no "upcoming game" practice report to file once they're
already on IR and not practicing. This is the right scope for closing
the actual gap (week-to-week uncertainty the market prices in), just
not a complete injury history by itself.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"


def load_injury_reports(season):
    url = INJURIES_URL.format(season=season)
    return pd.read_csv(url, low_memory=False)


def get_players_out(season, week, team=None):
    df = load_injury_reports(season)
    out = df[(df["week"] == week) & (df["report_status"] == "Out")]
    if team:
        out = out[out["team"] == team]
    return out[["team", "full_name", "position", "report_primary_injury"]]


if __name__ == "__main__":
    print("Validating against the real, known 2023 Deshaun Watson case...")
    watson_weeks = []
    for week in [6, 7, 8, 9]:
        out = get_players_out(2023, week, team="CLE")
        watson_row = out[out["full_name"] == "Deshaun Watson"]
        if len(watson_row) > 0:
            watson_weeks.append(week)
            print(f"  Week {week}: Watson correctly shown as Out ({watson_row.iloc[0]['report_primary_injury']})")
        else:
            print(f"  Week {week}: Watson not listed as Out")

    assert 6 in watson_weeks and 8 in watson_weeks, "Expected Watson to show as Out in weeks 6 and 8"
    print("\nPASS: real injury data correctly matches the known real case")
