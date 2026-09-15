"""
In-season bridge: current-season MLB games from the official Stats
API (statsapi.mlb.com, free, keyless) -> the SAME cache schema the
walk-forward model reads, so ratings stay current without waiting
for Retrosheet's post-season archive.

Outputs (append-style, current season only):
  model/mlb_schedule_current.csv, model/mlb_pitching_current.csv
The model loads historical + current via load_mlb_caches().

Cross-season identity: pitcher keys bridge from MLB Stats API ids to
Retrosheet ids via the Chadwick register (GitHub-hosted; bridge
verified live at build time), so a starter's 2025 rating carries into
his 2026 appearances instead of resetting at the season boundary.
Players with no retro id yet (debuts) get "mlbam<id>" -- correct
behavior: no history exists to bridge to. Parks map through each home
team's modal park from the historical cache (stable; neutral-site
games inherit the home team's park, a documented approximation).

Team keys map statsapi abbreviations -> retrosheet keys, including
ATH for the relocated Athletics (verified against the 2025 cache).

Self-healing incremental: each run refetches from the last cached
date minus 2 days (late official corrections) through yesterday.
First run of a season backfills everything (one boxscore call per
game, gentle 0.3s pacing -- a full-season backfill is ~2400 calls).

statsapi is unreachable from the build sandbox: parsers are
unit-tested on captured response shapes; first live run verifies.
"""

import io
import os
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHED_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={a}&endDate={b}"
             "&gameTypes=R&hydrate=team")
BOX_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
REGISTER_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{shard}.csv"

STATS_TO_RETRO = {
    "LAA": "ANA", "AZ": "ARI", "ARI": "ARI", "ATH": "ATH", "OAK": "ATH", "ATL": "ATL",
    "BAL": "BAL", "BOS": "BOS", "CWS": "CHA", "CHC": "CHN", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCA", "LAD": "LAN", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYY": "NYA", "NYM": "NYN", "PHI": "PHI", "PIT": "PIT",
    "SD": "SDN", "SEA": "SEA", "SF": "SFN", "STL": "SLN", "TB": "TBA", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WAS",
}


def load_id_bridge(timeout=45):
    """{mlbam_id(int): retro_key} from the Chadwick register shards."""
    bridge = {}
    for shard in "0123456789abcdef":
        try:
            req = urllib.request.Request(REGISTER_URL.format(shard=shard), headers={"User-Agent": "coverline-mlb"})
            df = pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=timeout).read()),
                             usecols=["key_mlbam", "key_retro"], low_memory=False)
            ok = df.dropna()
            bridge.update({int(m): r for m, r in zip(ok["key_mlbam"], ok["key_retro"])})
        except Exception as e:
            print(f"[mlb_daily] register shard {shard} soft-fail: {e}")
    print(f"[mlb_daily] id bridge: {len(bridge)} mlbam->retro mappings")
    return bridge


def parse_boxscore_pitching(box, bridge):
    """[(retro_team, pitcher_key, started, finished, outs, pitches,
    runs, er, h, hr, bb, so)] for both sides. Pure, unit-tested."""
    rows = []
    for side in ("home", "away"):
        team = box["teams"][side]
        abbr = STATS_TO_RETRO.get(team["team"].get("abbreviation"), team["team"].get("abbreviation"))
        order = team.get("pitchers", [])
        for i, pid in enumerate(order):
            player = team["players"].get(f"ID{pid}")
            if not player:
                continue
            st = (player.get("stats") or {}).get("pitching") or {}
            if not st:
                continue
            ip = str(st.get("inningsPitched", "0.0"))
            whole, frac = (ip.split(".") + ["0"])[:2]
            outs = int(whole) * 3 + int(frac)
            rows.append({
                "team": abbr,
                "pitcher": bridge.get(int(pid), f"mlbam{pid}"),
                "started": 1 if i == 0 else 0,
                "finished": 1 if i == len(order) - 1 else 0,
                "outs": outs,
                "pitches": st.get("numberOfPitches") or st.get("pitchesThrown"),
                "runs": st.get("runs", 0), "er": st.get("earnedRuns", 0),
                "h": st.get("hits", 0), "hr": st.get("homeRuns", 0),
                "bb": st.get("baseOnBalls", 0), "so": st.get("strikeOuts", 0),
            })
    return rows


def parse_schedule_finals(payload):
    """Completed regular-season games -> minimal schedule rows
    (scores from the schedule feed; SPs filled from boxscores)."""
    out = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            if (g.get("status") or {}).get("abstractGameState") != "Final":
                continue
            h, a = g["teams"]["home"], g["teams"]["away"]
            out.append({
                "game_pk": g["gamePk"],
                "date": day["date"],
                "game_number": (g.get("gameNumber", 1) - 1) if g.get("doubleHeader") in ("Y", "S") else 0,
                "home_team": STATS_TO_RETRO.get(h["team"].get("abbreviation"), h["team"].get("abbreviation")),
                "away_team": STATS_TO_RETRO.get(a["team"].get("abbreviation"), a["team"].get("abbreviation")),
                "home_score": h.get("score"), "away_score": a.get("score"),
            })
    return out


def modal_parks(historical_schedule):
    hs = historical_schedule[historical_schedule["season"] == historical_schedule["season"].max()]
    return hs.groupby("home_team")["park"].agg(lambda x: x.value_counts().index[0]).to_dict()


def update(season=None, data_dir="model", pace=0.3):
    season = season or date.today().year
    hist = pd.read_csv(os.path.join(data_dir, "mlb_schedule_cache.csv"))
    parks = modal_parks(hist)
    sched_path = os.path.join(data_dir, "mlb_schedule_current.csv")
    pit_path = os.path.join(data_dir, "mlb_pitching_current.csv")
    existing = pd.read_csv(sched_path) if os.path.exists(sched_path) else pd.DataFrame()
    start = (pd.to_datetime(existing["date"]).max() - timedelta(days=2)).date() if len(existing) else date(season, 3, 1)
    end = date.today() - timedelta(days=0)
    print(f"[mlb_daily] fetching {start} .. {end}")
    payload = requests.get(SCHED_URL.format(a=start.isoformat(), b=end.isoformat()), timeout=40).json()
    finals = parse_schedule_finals(payload)
    print(f"[mlb_daily] {len(finals)} completed games in window")
    bridge = load_id_bridge()
    sched_rows, pit_rows = [], []
    for g in finals:
        try:
            box = requests.get(BOX_URL.format(pk=g["game_pk"]), timeout=30).json()
            rows = parse_boxscore_pitching(box, bridge)
        except Exception as e:
            print(f"[mlb_daily] boxscore {g['game_pk']} soft-fail: {e}")
            continue
        key = f"{g['home_team']}{g['date'].replace('-', '')}{g['game_number']}"
        sp = {r["team"]: r["pitcher"] for r in rows if r["started"]}
        sched_rows.append({
            "season": season, "game_key": key, "date": g["date"], "game_number": g["game_number"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "home_score": g["home_score"], "away_score": g["away_score"],
            "park": parks.get(g["home_team"], "UNK"),
            "home_sp": sp.get(g["home_team"]), "away_sp": sp.get(g["away_team"]),
        })
        for r in rows:
            pit_rows.append({"season": season, "date": g["date"], "game_key": key,
                             "opponent": g["away_team"] if r["team"] == g["home_team"] else g["home_team"], **r})
        time.sleep(pace)
    new_sched = pd.DataFrame(sched_rows)
    new_pit = pd.DataFrame(pit_rows)
    if len(existing):
        new_sched = pd.concat([existing[~existing["game_key"].isin(new_sched.get("game_key", []))], new_sched])
        old_pit = pd.read_csv(pit_path)
        new_pit = pd.concat([old_pit[~old_pit["game_key"].isin(set(new_pit.get("game_key", [])))], new_pit])
    new_sched.sort_values(["date", "game_key"]).to_csv(sched_path, index=False)
    new_pit.sort_values(["date", "game_key"]).to_csv(pit_path, index=False)
    print(f"[mlb_daily] caches now hold {len(new_sched)} current-season games")
    return sched_path, pit_path


def load_mlb_caches(data_dir="model"):
    """Historical + current season, one frame each -- what the model reads."""
    sched = pd.read_csv(os.path.join(data_dir, "mlb_schedule_cache.csv"))
    pit = pd.read_csv(os.path.join(data_dir, "mlb_pitching_cache.csv"))
    for extra, base in (("mlb_schedule_current.csv", sched), ("mlb_pitching_current.csv", pit)):
        p = os.path.join(data_dir, extra)
        if os.path.exists(p):
            cur = pd.read_csv(p)
            if extra.startswith("mlb_schedule"):
                sched = pd.concat([base, cur], ignore_index=True)
            else:
                pit = pd.concat([base, cur], ignore_index=True)
    return sched, pit


if __name__ == "__main__":
    paths = update()
    if os.environ.get("GIT_REPO_URL"):
        from deploy.git_utils import git_commit_and_push
        for p in paths:
            git_commit_and_push(p, commit_message=f"MLB in-season cache: {os.path.basename(p)}")
