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



def _norm_name(name):
    """Cross-source name comparison: nflverse says "Michael Penix",
    ESPN says "Michael Penix Jr." -- strip suffixes and punctuation so
    a suffix difference can't silently kill a real QB alert (caught in
    testing against real 2025 data, 2026-09-04)."""
    if not name:
        return ""
    parts = name.replace(".", "").split()
    while parts and parts[-1].lower() in ("jr", "sr", "ii", "iii", "iv", "v"):
        parts.pop()
    return " ".join(parts).lower()


DEPTH_CHARTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet"


def _projected_starters(season):
    """Authoritative current QB1 per team from nflverse depth charts --
    published preseason and updated through the year, which fixes the
    Week 1 gap the prior-season-modal fallback cannot: offseason QB
    changes. Caught live 2026-09-04: the 2026 chart shows ATL QB1 is
    an offseason acquisition, so modal-from-2025 named the wrong
    starter and an injury alert fired for a backup. Soft-fail returns
    {} and callers fall back to modal-from-games."""
    try:
        dc = pd.read_parquet(DEPTH_CHARTS_URL.format(season=season))
        qb1 = dc[(dc["pos_abb"] == "QB") & (dc["pos_rank"] == 1)]
        latest = qb1.sort_values("dt").groupby("team").tail(1)
        return dict(zip(latest["team"], latest["player_name"]))
    except Exception as e:
        print(f"[qb_status] depth charts unavailable ({e}); using modal starters")
        return {}


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


def _last_snapshot_qb1_map(data_dir):
    """Last-known-good QB1 map, embedded in the newest divergence
    snapshot by odds_watch. Depth charts move slowly, so yesterday's
    map is a far better outage fallback than the prior-season modal
    (which names the wrong starter for every offseason QB change)."""
    try:
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(data_dir, "divergence", "*.json")))
        for path in reversed(files):
            with open(path) as f:
                snap = __import__("json").load(f)
            if snap.get("qb1_map"):
                return snap["qb1_map"]
    except Exception:
        pass
    return {}


def get_projected_starters(season, data_dir=None):
    """Merged starter truth: live depth chart, else the last snapshot's
    cached map. Also called by odds_watch to embed the map."""
    data_dir = data_dir or os.environ.get("REPO_DATA_PATH", "./data")
    live = _projected_starters(season)
    if live:
        return live
    cached = _last_snapshot_qb1_map(data_dir)
    if cached:
        print(f"[qb_status] using cached QB1 map from last snapshot ({len(cached)} teams)")
    return cached


def get_qb_alerts(season, week, games=None, injuries=None):
    """Returns {team: reason_string} for teams with QB uncertainty.
    Teams absent from the dict are clear. Empty dict on any data
    problem -- this feature must never break the odds pipeline."""
    try:
        if games is None:
            games = pd.read_csv(GAMES_URL)
        # Current depth-chart QB1 is the truth for "who starts"; modal
        # from played games fills any team the chart is missing.
        modal = _modal_starters(games, season, week)
        modal.update(get_projected_starters(season))

        alerts = {}
        try:
            if injuries is None:
                injuries = pd.read_parquet(INJURIES_URL.format(season=season))
            inj_qb = injuries[(injuries["position"] == "QB") & (injuries["week"] == week)]
            qb_rows = [(row["team"], row["full_name"], row.get("report_status")) for _, row in inj_qb.iterrows()]
        except Exception as inj_err:
            # Pre-Wednesday gap: official reports not filed yet. ESPN's
            # current statuses stand in (same mapped vocabulary; see
            # deploy/game_context.py for the fallback's ground rules).
            print(f"[qb_status] nflverse injuries unavailable ({inj_err}); trying ESPN fallback")
            from deploy.game_context import fetch_espn_injuries
            qb_rows = [(team, r["player"], r["status"])
                       for team, rows in fetch_espn_injuries().items()
                       for r in rows if r["position"] == "QB"]
        for team, name, status in qb_rows:
            if status in ALERT_STATUSES and _norm_name(modal.get(team)) == _norm_name(name):
                alerts[team] = f"{name} listed {status}"

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
                if qb_name and _norm_name(qb_name) != _norm_name(modal[team]):
                    alerts[team] = f"{qb_name} started last game (current QB1: {modal[team]})"
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
