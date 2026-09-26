#!/usr/bin/env python3
"""Export the NBA Players tab: availability and minutes. NOT a model input.

Writes data/site/players_nba.json from what model/ingest/nba_players.py keeps
in data/raw/nba. The site renders it and computes nothing.

WHAT IT CONTAINS, PER TEAM
  injuries  ESPN's report, worst status first, with ESPN's own report time.
  rotation  the players who carried the team's last ten games: minutes per
            game played, games played of the ten, starts, games missed.

WHAT IT DELIBERATELY DOES NOT CONTAIN
  Any price, probability or adjustment. The NBA model is team-level; this is
  the information it cannot see, shown so the owner can weigh it. Deriving a
  "player-adjusted" number here would be an ungraded model smuggled into a
  display file, which is the thing ADR 0024 exists to prevent.

NO TIMESTAMP OF ITS OWN. The file carries the report and box-score dates it
was built from, not when it was built, so an unchanged day writes identical
bytes and the live-inputs job commits nothing.

    python scripts/export_players.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RAW = ROOT / "data" / "raw" / "nba"
SITE = ROOT / "data" / "site"
LAST_N = 10
ROTATION = 9

NOTE = ("Not a model input. The NBA model rates teams, not players. This is what "
        "it cannot see: who is playing, and for how long. It is shown so you can "
        "decide whether a suggestion still holds.")


def season_ending(d: date) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def team_names() -> dict[str, str]:
    from coverline.leagues.nba.teams import NBA_NAMES
    out: dict[str, str] = {}
    for name, code in NBA_NAMES.items():
        out.setdefault(code, name)                     # first spelling wins
    return out


def rotation(box: pd.DataFrame, team: str, last_n: int = LAST_N,
             top: int = ROTATION) -> tuple[list[dict], int]:
    """The players who carried `team`'s last `last_n` games. Pure."""
    t = box[box.team == team]
    games = t[["game_id", "date"]].drop_duplicates().sort_values("date").tail(last_n)
    if games.empty:
        return [], 0
    recent = t[t.game_id.isin(games.game_id)]
    n = len(games)
    rows = []
    for (aid, player), g in recent.groupby(["athlete_id", "player"], sort=False):
        played = g[(g.minutes.fillna(0) > 0) & ~g.did_not_play]
        if played.empty:
            continue
        rows.append({
            "player": player,
            "position": g.position.iloc[-1] or "",
            "min_per_game": round(float(played.minutes.mean()), 1),
            "played": int(len(played)),
            "of": n,
            "starts": int(played.starter.sum()),
            "missed": int(n - len(played)),
            "_total": float(played.minutes.sum()),
        })
    rows.sort(key=lambda r: (-r["_total"], r["player"]))
    for r in rows:
        r.pop("_total")
    return rows[:top], n


def build(today: date | None = None, raw: Path = RAW) -> dict:
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    season = season_ending(today)
    names = team_names()

    inj_path = raw / "injuries_current.json"
    injuries = json.loads(inj_path.read_text()) if inj_path.exists() else {}
    box_path = raw / f"player_box_{season}.parquet"
    box = pd.read_parquet(box_path) if box_path.exists() else pd.DataFrame(
        columns=["team", "game_id", "date", "athlete_id", "player", "position",
                 "starter", "minutes", "did_not_play"])

    try:
        logo = {k: v["dark"] for k, v in
                json.loads((ROOT / "data" / "logos.json").read_text()).get("nba", {}).items()}
    except (OSError, ValueError, KeyError):
        logo = {}
    teams = []
    for code in sorted(names):
        rot, n = rotation(box, code)
        teams.append({"team": code, "name": names[code], "logo": logo.get(code),
                      "injuries": injuries.get(code, []),
                      "rotation": rot, "games_in_window": n})

    reported = [r.get("reported") for k, rows in injuries.items() if not k.startswith("_")
                for r in rows if r.get("reported")]
    minutes_through = str(box.date.max())[:10] if len(box) else None
    return {
        "league": "nba",
        "not_a_model_input": True,
        "note": NOTE,
        "season": season,
        "injuries_as_of": max(reported) if reported else None,
        "minutes_through": minutes_through,
        "minutes_note": (None if minutes_through else
                         f"Minutes appear after the first {season - 1}-{str(season)[2:]} "
                         "regular-season games. Last season's rotations are not shown: "
                         "rosters change over the summer."),
        "window": LAST_N,
        "unmatched_team_names": injuries.get("_unmatched", []),
        "teams": teams,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(SITE))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    art = build()
    (out / "players_nba.json").write_text(json.dumps(art, indent=1) + "\n")
    n_inj = sum(len(t["injuries"]) for t in art["teams"])
    n_rot = sum(1 for t in art["teams"] if t["rotation"])
    print(f"players_nba.json: {n_inj} injury rows, rotations for {n_rot} team(s), "
          f"minutes through {art['minutes_through']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
