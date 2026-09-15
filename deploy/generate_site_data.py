"""
Site data for the Teams / Schedule / Players surfaces ->
data/site/teams.json + data/site/player_leaders.json.

Runs inside the weekly job (after ratings, so SOS uses the freshest
numbers) and on demand. Everything here is presentation data -- no
model inputs, so staleness is a display concern only. Injuries on
team pages refresh weekly with this file; the per-GAME injury panels
on bet cards stay hours-fresh via the odds watch, and the team page
says which is which.

teams.json, per team: identity (name, nickname, division, conference,
ESPN logo URL -- their CDN serves these publicly by team code),
record + points for/against from the live schedule, the full season
schedule with results as played, remaining-opponent list, and
strength of schedule (mean current model rating of remaining
opponents; played-SOS included for context), plus a compact depth
chart (QB/RB/WR/TE two-deep from nflverse) and the merged injury
report.

player_leaders.json: season-to-date top-10s in the core categories
plus a high-usage board (touches = carries + targets). Falls back to
the prior season's final leaders, clearly labeled, until the current
season's first stats publish.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAMS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teams.csv"
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"

DIVISIONS = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LAC": "AFC West", "LV": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SEA": "NFC West", "SF": "NFC West",
}
ESPN_LOGO_ABBR = {"WAS": "wsh", "LA": "lar"}  # everyone else: lowercase nflverse abbr


def logo_url(abbr):
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{ESPN_LOGO_ABBR.get(abbr, abbr.lower())}.png"


def _latest_ratings(data_dir, season):
    from model.prediction import find_latest_ratings_snapshot, load_current_ratings
    path = find_latest_ratings_snapshot(data_dir, season)
    if path is None:
        return {}
    ratings = load_current_ratings(path)
    return {t: ratings.loc[t, "total_rating"] for t in ratings.index}


def _depth_charts(season):
    try:
        from deploy.qb_status import DEPTH_CHARTS_URL
        dc = pd.read_parquet(DEPTH_CHARTS_URL.format(season=season))
        if "pos_abb" not in dc.columns:
            return {}
        keep = dc[dc["pos_abb"].isin(["QB", "RB", "WR", "TE"]) & (dc["pos_rank"] <= 2)]
        latest = keep.sort_values("dt").groupby(["team", "pos_abb", "pos_rank"]).tail(1)
        out = {}
        for (team, pos), grp in latest.groupby(["team", "pos_abb"]):
            out.setdefault(team, {})[pos] = [
                r["player_name"] for _, r in grp.sort_values("pos_rank").iterrows()]
        return out
    except Exception as e:
        print(f"[site_data] depth charts soft-fail: {e}")
        return {}


def build_teams(season, data_dir):
    games = pd.read_csv(GAMES_URL)
    sched = games[(games["season"] == season) & (games["game_type"] == "REG")]
    names = pd.read_csv(TEAMS_URL)
    names = names[names["season"] == names["season"].max()].set_index("team") if "season" in names.columns else names.set_index("team")

    ratings = _latest_ratings(os.path.join(data_dir), season)
    depth = _depth_charts(season)
    try:
        from deploy.game_context import get_injury_report
        played_weeks = sched[sched["home_score"].notna()]["week"]
        current_week = int(sched[sched["home_score"].isna()]["week"].min()) if sched["home_score"].isna().any() else int(sched["week"].max())
        injuries, injury_source = get_injury_report(season, current_week)
    except Exception as e:
        print(f"[site_data] injuries soft-fail: {e}")
        injuries, injury_source = {}, None

    teams = {}
    for abbr in DIVISIONS:
        rows = sched[(sched["home_team"] == abbr) | (sched["away_team"] == abbr)].sort_values("week")
        w = l = t = pf = pa = 0
        schedule = []
        remaining_opps = []
        for _, gm in rows.iterrows():
            home = gm["home_team"] == abbr
            opp = gm["away_team"] if home else gm["home_team"]
            entry = {"week": int(gm["week"]), "opp": opp, "home": bool(home),
                     "kickoff": (f"{gm['gameday']}T{str(gm['gametime'])[:5]}" if pd.notna(gm.get("gameday")) else None)}
            if pd.notna(gm["home_score"]):
                us = gm["home_score"] if home else gm["away_score"]
                them = gm["away_score"] if home else gm["home_score"]
                pf += us; pa += them
                w += us > them; l += us < them; t += us == them
                entry["result"] = f"{'W' if us > them else 'L' if us < them else 'T'} {int(us)}-{int(them)}"
            else:
                remaining_opps.append(opp)
            schedule.append(entry)
        div = DIVISIONS[abbr]
        rem_sos = round(sum(ratings.get(o, 0.0) for o in remaining_opps) / len(remaining_opps), 2) if remaining_opps else None
        played_opps = [s["opp"] for s in schedule if "result" in s]
        played_sos = round(sum(ratings.get(o, 0.0) for o in played_opps) / len(played_opps), 2) if played_opps else None
        teams[abbr] = {
            "abbr": abbr,
            "name": names.loc[abbr, "full"] if abbr in names.index else abbr,
            "nickname": names.loc[abbr, "nickname"] if abbr in names.index else abbr,
            "division": div, "conference": div.split(" ")[0],
            "logo": logo_url(abbr),
            "record": {"w": int(w), "l": int(l), "t": int(t), "pf": int(pf), "pa": int(pa)},
            "schedule": schedule,
            "remaining_sos": rem_sos, "played_sos": played_sos,
            "depth_chart": depth.get(abbr, {}),
            "injuries": injuries.get(abbr, []),
        }
    return {"season": season, "injury_source": injury_source,
            "injury_note": "Team-page injuries refresh weekly; the injury panels on bet cards refresh with every odds run.",
            "teams": teams}


def build_player_leaders(season):
    used_season, label = season, f"{season} season to date"
    try:
        ps = pd.read_parquet(STATS_URL.format(season=season))
        if ps.empty:
            raise ValueError("empty")
    except Exception:
        used_season, label = season - 1, f"{season - 1} final (current season stats not published yet)"
        ps = pd.read_parquet(STATS_URL.format(season=used_season))
    if "season_type" in ps.columns:
        ps = ps[ps["season_type"] == "REG"]
    agg = ps.groupby(["player_display_name", "position", "team"], as_index=False).agg(
        pass_yds=("passing_yards", "sum"), pass_tds=("passing_tds", "sum"),
        rush_yds=("rushing_yards", "sum"), rush_tds=("rushing_tds", "sum"),
        rec_yds=("receiving_yards", "sum"), rec_tds=("receiving_tds", "sum"),
        receptions=("receptions", "sum") if "receptions" in ps.columns else ("targets", "sum"),
        targets=("targets", "sum"), carries=("carries", "sum"),
    )
    agg["touches"] = agg["targets"].fillna(0) + agg["carries"].fillna(0)

    def top(col, n=10, min_pos=None):
        d = agg if min_pos is None else agg[agg["position"].isin(min_pos)]
        rows = d.nlargest(n, col)
        return [{"player": r["player_display_name"], "team": r["team"], "position": r["position"],
                 "value": round(float(r[col]), 0)} for _, r in rows.iterrows() if r[col] > 0]

    return {
        "label": label, "season": used_season,
        "categories": {
            "Passing yards": top("pass_yds"), "Passing TDs": top("pass_tds"),
            "Rushing yards": top("rush_yds"), "Rushing TDs": top("rush_tds"),
            "Receiving yards": top("rec_yds"), "Receiving TDs": top("rec_tds"),
            "Receptions": top("receptions"), "Targets": top("targets"),
        },
        "high_usage": top("touches", n=15),
    }


def generate(data_dir, season):
    out_dir = os.path.join(data_dir, "site")
    os.makedirs(out_dir, exist_ok=True)
    teams = build_teams(season, data_dir)
    with open(os.path.join(out_dir, "teams.json"), "w") as f:
        json.dump(teams, f, indent=1)
    leaders = build_player_leaders(season)
    with open(os.path.join(out_dir, "player_leaders.json"), "w") as f:
        json.dump(leaders, f, indent=1)
    n_inj = sum(len(t["injuries"]) for t in teams["teams"].values())
    print(f"[site_data] teams.json: 32 teams, {n_inj} injury rows, source={teams['injury_source']}")
    print(f"[site_data] player_leaders.json: {leaders['label']}")
    return [os.path.join(out_dir, "teams.json"), os.path.join(out_dir, "player_leaders.json")]


if __name__ == "__main__":
    generate(os.environ.get("REPO_DATA_PATH", "./data"), int(os.environ.get("SEASON", 2026)))
