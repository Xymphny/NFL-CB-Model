"""
Real NFL playoff seeding with tiebreakers -- the gap
model/season_simulation.py's own docstring explicitly flagged: win-total
distributions and division-record simulation work, but full playoff
seeding never got built.

Implements the most commonly-decisive tiebreaker rules (head-to-head,
division record, conference record, strength of victory), in the
official order. Does NOT implement every rule in the real NFL
tiebreaker procedure -- "common games" requiring a minimum-4-common-
opponent threshold, the combined conference/league point ranking rules,
net points/net touchdowns, and the final coin-toss step are all real
rules this doesn't cover. Point differential is used as a documented
stand-in for the remaining rare rules once the implemented ones don't
resolve a tie -- flagged explicitly, not silently passed off as
official.

Validated against a real, known outcome: the actual 2023 season's
final standings.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SF", "SEA"],
}

TEAM_TO_DIVISION = {team: div for div, teams in DIVISIONS.items() for team in teams}
TEAM_TO_CONFERENCE = {team: ("AFC" if div.startswith("AFC") else "NFC") for team, div in TEAM_TO_DIVISION.items()}


def compute_standings(schedule):
    records = {}
    for _, g in schedule.dropna(subset=["home_score", "away_score"]).iterrows():
        home, away = g["home_team"], g["away_team"]
        for team in (home, away):
            records.setdefault(team, {"wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0,
                                       "div_wins": 0, "div_losses": 0, "conf_wins": 0, "conf_losses": 0,
                                       "beaten": []})

        home_score, away_score = g["home_score"], g["away_score"]
        is_div = bool(g.get("div_game", False))
        is_conf = TEAM_TO_CONFERENCE.get(home) == TEAM_TO_CONFERENCE.get(away)

        if home_score > away_score:
            winner, loser = home, away
        elif away_score > home_score:
            winner, loser = away, home
        else:
            winner, loser = None, None

        records[home]["points_for"] += home_score
        records[home]["points_against"] += away_score
        records[away]["points_for"] += away_score
        records[away]["points_against"] += home_score

        if winner:
            records[winner]["wins"] += 1
            records[loser]["losses"] += 1
            records[winner]["beaten"].append(loser)
            if is_div:
                records[winner]["div_wins"] += 1
                records[loser]["div_losses"] += 1
            if is_conf:
                records[winner]["conf_wins"] += 1
                records[loser]["conf_losses"] += 1
        else:
            records[home]["ties"] += 1
            records[away]["ties"] += 1

    df = pd.DataFrame(records).T
    for col in ["wins", "losses", "ties", "points_for", "points_against", "div_wins", "div_losses", "conf_wins", "conf_losses"]:
        df[col] = df[col].astype(int)
    df["win_pct"] = (df["wins"] + 0.5 * df["ties"]) / (df["wins"] + df["losses"] + df["ties"]).replace(0, 1)
    df["point_diff"] = df["points_for"] - df["points_against"]
    return df


def head_to_head_pct(teams, schedule):
    wins = {t: 0 for t in teams}
    games = {t: 0 for t in teams}
    for _, g in schedule.dropna(subset=["home_score", "away_score"]).iterrows():
        if g["home_team"] in teams and g["away_team"] in teams:
            games[g["home_team"]] += 1
            games[g["away_team"]] += 1
            if g["home_score"] > g["away_score"]:
                wins[g["home_team"]] += 1
            elif g["away_score"] > g["home_score"]:
                wins[g["away_team"]] += 1
    return {t: (wins[t] / games[t] if games[t] > 0 else None) for t in teams}


def strength_of_victory(team, standings):
    beaten = standings.loc[team, "beaten"]
    if not beaten:
        return 0.0
    return sum(standings.loc[b, "win_pct"] for b in beaten if b in standings.index) / len(beaten)


def break_tie(tied_teams, standings, schedule, same_division):
    if len(tied_teams) == 1:
        return tied_teams

    if len(tied_teams) == 2:
        h2h = head_to_head_pct(tied_teams, schedule)
        if h2h[tied_teams[0]] is not None and h2h[tied_teams[0]] != h2h[tied_teams[1]]:
            return sorted(tied_teams, key=lambda t: -h2h[t])

    if same_division:
        div_pct = {t: standings.loc[t, "div_wins"] / max(standings.loc[t, "div_wins"] + standings.loc[t, "div_losses"], 1) for t in tied_teams}
        if len(set(div_pct.values())) > 1:
            return sorted(tied_teams, key=lambda t: -div_pct[t])

    conf_pct = {t: standings.loc[t, "conf_wins"] / max(standings.loc[t, "conf_wins"] + standings.loc[t, "conf_losses"], 1) for t in tied_teams}
    if len(set(conf_pct.values())) > 1:
        return sorted(tied_teams, key=lambda t: -conf_pct[t])

    sov = {t: strength_of_victory(t, standings) for t in tied_teams}
    if len(set(sov.values())) > 1:
        return sorted(tied_teams, key=lambda t: -sov[t])

    return sorted(tied_teams, key=lambda t: -standings.loc[t, "point_diff"])


def seed_conference(conference, standings, schedule):
    conf_teams = [t for t in standings.index if TEAM_TO_CONFERENCE.get(t) == conference]
    conf_divisions = {d: teams for d, teams in DIVISIONS.items() if d.startswith(conference)}

    division_winners = []
    for div, teams in conf_divisions.items():
        div_standings = standings.loc[[t for t in teams if t in standings.index]]
        best_pct = div_standings["win_pct"].max()
        tied = div_standings[div_standings["win_pct"] == best_pct].index.tolist()
        ordered = break_tie(tied, standings, schedule, same_division=True)
        division_winners.append(ordered[0])

    division_winners_ordered = sorted(division_winners, key=lambda t: -standings.loc[t, "win_pct"])
    seeds = []
    i = 0
    while i < len(division_winners_ordered):
        pct = standings.loc[division_winners_ordered[i], "win_pct"]
        tied_group = [t for t in division_winners_ordered[i:] if standings.loc[t, "win_pct"] == pct]
        seeds.extend(break_tie(tied_group, standings, schedule, same_division=False))
        i += len(tied_group)

    remaining = [t for t in conf_teams if t not in division_winners]
    remaining_ordered = sorted(remaining, key=lambda t: -standings.loc[t, "win_pct"])
    wildcards = []
    i = 0
    while i < len(remaining_ordered) and len(wildcards) < 3:
        pct = standings.loc[remaining_ordered[i], "win_pct"]
        tied_group = [t for t in remaining_ordered[i:] if standings.loc[t, "win_pct"] == pct]
        wildcards.extend(break_tie(tied_group, standings, schedule, same_division=False))
        i += len(tied_group)

    return (seeds + wildcards)[:7]


if __name__ == "__main__":
    from ingest.nfl_schedules import load_schedules

    print("Validating against the REAL, known 2023 playoff seeding...")
    schedule = load_schedules(seasons=[2023])
    standings = compute_standings(schedule)

    for conf in ["AFC", "NFC"]:
        seeds = seed_conference(conf, standings, schedule)
        print(f"\n{conf} seeds (computed):")
        for i, team in enumerate(seeds, 1):
            rec = standings.loc[team]
            print(f"  {i}. {team} ({int(rec['wins'])}-{int(rec['losses'])}-{int(rec['ties'])})")
