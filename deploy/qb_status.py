"""
QB-status awareness from FREE nflverse data -- addressing the measured
finding that 31% of held-out games had a non-modal QB starting, and in
those games the model's disagreement with the market jumps from 2.1 to
3.4 points (the market knows the QB is out; the model doesn't).

DELIBERATELY ANNOTATION-ONLY, not an automatic tier demotion: the
held-out check showed excluding backup games does NOT improve flagged
ATS (backup-game flags graded 50.7-51.8% vs 48.4-53.1% for usual-QB
flags -- samples too small either way, and the market over-adjusting
to backup news is itself a known counter-spot). So this feeds a
visible confidence driver and lets live grading accumulate the
evidence a demotion rule would need.

Sources (both free, both weekly):
- games.csv actual starters -> each team's modal starter season-to-date
  (prior season's modal used for weeks 1-2, when the season sample is
  too thin to define "usual").
- nflverse injuries -> latest report_status for that modal starter.
"""

import os

import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
ALERT_STATUSES = {"Out", "Doubtful", "Questionable"}


def _modal_starters(games, season, through_week):
    """Modal QB per team using games strictly before `through_week`;
    falls back to the prior season when fewer than 3 starts exist."""
    modal = {}
    for use_season, week_cap in ((season, through_week), (season - 1, 99)):
        g = games[(games["season"] == use_season) & (games["week"] < week_cap)]
        qb = pd.concat([
            g.rename(columns={"home_team": "team", "home_qb_name": "qb"})[["team", "qb"]],
            g.rename(columns={"away_team": "team", "away_qb_name": "qb"})[["team", "qb"]],
        ]).dropna(subset=["qb"])
        counts = qb.groupby("team")["qb"].agg(lambda s: (s.value_counts().index[0], int(s.value_counts().iloc[0])))
        for team, (name, n) in counts.items():
            if team not in modal and n >= (3 if use_season == season else 1):
                modal[team] = name
    return modal


def get_qb_alerts(season, week, games=None, injuries=None):
    """Returns {team: reason_string} for teams with QB uncertainty.
    Teams absent from the dict are clear. Empty dict on any data
    problem -- this feature must never break the odds pipeline."""
    try:
        if games is None:
            games = pd.read_csv(GAMES_URL)
        modal = _modal_starters(games, season, week)

        alerts = {}
        if injuries is None:
            injuries = pd.read_parquet(INJURIES_URL.format(season=season))
        inj_qb = injuries[(injuries["position"] == "QB") & (injuries["week"] == week)]
        for _, row in inj_qb.iterrows():
            status = row.get("report_status")
            if status in ALERT_STATUSES and modal.get(row["team"]) == row["full_name"]:
                alerts[row["team"]] = f"{row['full_name']} listed {status}"

        # Second signal: last completed game started by a non-modal QB
        # (covers benchings and injuries that never hit a report).
        played = games[(games["season"] == season) & (games["week"] < week) & games["home_score"].notna()]
        if len(played):
            last_start = {}
            for _, gm in played.sort_values("week").iterrows():
                last_start[gm["home_team"]] = gm["home_qb_name"]
                last_start[gm["away_team"]] = gm["away_qb_name"]
            for team, qb_name in last_start.items():
                if team in alerts or team not in modal:
                    continue
                if qb_name and qb_name != modal[team]:
                    alerts[team] = f"{qb_name} started last game (usual: {modal[team]})"
        return alerts
    except Exception as e:
        print(f"[qb_status] soft-fail, no alerts: {e}")
        return {}


if __name__ == "__main__":
    season = int(os.environ.get("SEASON", 2023))
    week = int(os.environ.get("WEEK", 10))
    alerts = get_qb_alerts(season, week)
    print(f"QB alerts for {season} week {week}: {len(alerts)} teams")
    for team, reason in sorted(alerts.items()):
        print(f"  {team}: {reason}")
