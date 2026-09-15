"""
MLB foundation caches from Chadwick retrosplits (GitHub-hosted, free,
keyless -- the nflverse of baseball history, verified live 2026-09):

  model/mlb_schedule_cache.csv  -- one row per regular-season game:
      date, game number (doubleheaders), home/away, scores, park,
      and BOTH starting pitchers (person.key).
  model/mlb_pitching_cache.csv  -- one row per pitcher appearance:
      date, pitcher, team, started?, outs, pitches, runs. This is the
      raw material for starter quality ratings AND the usage-derived
      bullpen availability proxy (back-to-back days, pitch counts).

Layer 1's shape (per the scoping discussion): stable team OFFENSE +
conditional pitching stack (today's starter -> pen) x park. These two
files are that model's entire historical substrate.
"""

import io
import os
import sys
import urllib.request

import pandas as pd

BASE = "https://raw.githubusercontent.com/chadwickbureau/retrosplits/master/daybyday/"


def _fetch(name):
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "coverline-mlb"})
    return pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()), low_memory=False)


def build(first_season, last_season, out_dir="model"):
    sched_rows, pitch_rows = [], []
    for season in range(first_season, last_season + 1):
        teams = _fetch(f"teams-{season}.csv")
        teams = teams[teams["season.phase"] == "R"]
        players = _fetch(f"playing-{season}.csv")
        players = players[players["season.phase"] == "R"]

        # Starting pitcher per team-game.
        sp = players[players["P_GS"] == 1][["game.key", "team.key", "person.key"]]
        sp_map = {(r["game.key"], r["team.key"]): r["person.key"] for _, r in sp.iterrows()}

        # One schedule row per game from the two team-alignment rows.
        by_game = {}
        for _, r in teams.iterrows():
            g = by_game.setdefault(r["game.key"], {})
            side = "home" if r["team.alignment"] == 1 else "away"
            g[side] = r
        for key, g in by_game.items():
            if "home" not in g or "away" not in g:
                continue
            h, a = g["home"], g["away"]
            sched_rows.append({
                "season": season,
                "game_key": key,
                "date": h["game.date"],
                "game_number": h["game.number"],       # 0 single, 1/2 doubleheader
                "home_team": h["team.key"], "away_team": a["team.key"],
                "home_score": h["B_R"], "away_score": a["B_R"],
                "park": h["site.key"],
                "home_sp": sp_map.get((key, h["team.key"])),
                "away_sp": sp_map.get((key, a["team.key"])),
            })

        # Every pitching appearance (starter quality + pen usage).
        pit = players[players["P_G"] == 1]
        for _, r in pit.iterrows():
            pitch_rows.append({
                "season": season, "date": r["game.date"], "game_key": r["game.key"],
                "pitcher": r["person.key"], "team": r["team.key"], "opponent": r["opponent.key"],
                "started": int(r["P_GS"]), "finished": int(r["P_GF"]),
                "outs": r["P_OUT"], "pitches": r["P_PITCH"],
                "runs": r["P_R"], "er": r["P_ER"], "h": r["P_H"], "hr": r["P_HR"],
                "bb": r["P_BB"], "so": r["P_SO"],
            })
        print(f"[mlb_gamelogs] {season}: {sum(1 for s in sched_rows if s['season']==season)} games, "
              f"{sum(1 for p in pitch_rows if p['season']==season)} pitching appearances")

    os.makedirs(out_dir, exist_ok=True)
    sched = pd.DataFrame(sched_rows).sort_values(["date", "game_key"])
    pit = pd.DataFrame(pitch_rows).sort_values(["date", "game_key"])
    sched.to_csv(os.path.join(out_dir, "mlb_schedule_cache.csv"), index=False)
    pit.to_csv(os.path.join(out_dir, "mlb_pitching_cache.csv"), index=False)
    sp_cov = sched[["home_sp", "away_sp"]].notna().all(axis=1).mean()
    print(f"[mlb_gamelogs] wrote {len(sched)} games ({first_season}-{last_season}), "
          f"SP identified both sides in {sp_cov*100:.1f}%")
    return sched, pit


if __name__ == "__main__":
    build(int(os.environ.get("MLB_FIRST", 2015)), int(os.environ.get("MLB_LAST", 2025)))
