#!/usr/bin/env python3
"""Team logos for the site, keyed by the codes each league's model uses.

Writes data/logos.json:

    {league: {code: {"dark": url, "light": url}}}

From ESPN's public team lists. Each team is mapped to the model's code by the
same tables the pipeline trusts for everything else -- never by stripping or
guessing -- and any team the tables cannot place is printed by name and left
out: a missing logo shows the code alone, a wrong one would put a team's
badge on another team's game.

  nfl  ESPN display name -> deploy.odds_watch_job.ODDS_TEAM_TO_ABBR (nflverse)
  nba  ESPN display name -> coverline.leagues.nba.teams.NBA_NAMES
  nhl  ESPN display name -> coverline.leagues.nhl.teams.TABLE
  mlb  ESPN display name -> coverline.leagues.mlb.teams.TABLE (Retrosheet)
  cfb  ESPN location     -> itself: the CFB model keys schools by ESPN's
                            location name (leagues/cfb/live.py)

"dark" is ESPN's variant drawn for dark backgrounds, which the site is.
Display only. Refresh when a team rebrands:

    python model/ingest/logos.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data" / "logos.json"
TEAMS = "https://site.api.espn.com/apis/site/v2/sports/{path}/teams?limit=1000"
PATHS = {"nfl": "football/nfl", "nba": "basketball/nba", "nhl": "hockey/nhl",
         "mlb": "baseball/mlb", "cfb": "football/college-football"}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def coder(league: str):
    """display name / location -> model code, raising on an unknown team."""
    if league == "nfl":
        from deploy.odds_watch_job import ODDS_TEAM_TO_ABBR
        return lambda t: ODDS_TEAM_TO_ABBR[t["displayName"]]
    if league == "nba":
        from coverline.leagues.nba.teams import NBA_NAMES
        return lambda t: NBA_NAMES[t["displayName"]]
    if league in ("nhl", "mlb"):
        import importlib
        table = importlib.import_module(f"coverline.leagues.{league}.teams").TABLE
        return lambda t: table.to_code(t["displayName"])
    return lambda t: t["location"]


def urls(team: dict) -> dict | None:
    """ESPN team record -> {"dark", "light"}. Pure."""
    by = {}
    for lg in team.get("logos", []):
        rel = set(lg.get("rel") or [])
        kind = "dark" if "dark" in rel or "/500-dark/" in lg.get("href", "") else "light"
        by.setdefault(kind, lg.get("href"))
    if not by:
        return None
    return {"dark": by.get("dark") or by.get("light"), "light": by.get("light") or by.get("dark")}


def _cfb_unambiguous(teams: list[dict]) -> list[dict]:
    """Where ESPN lists two schools under one location (Troy Trojans and
    Troy Vikings; Charlotte 49ers and Charlotte Saints), keep the one the
    CFB name table maps to that location, and drop the location if none:
    a shared name must never pick a badge by list order."""
    from collections import Counter
    from coverline.leagues.cfb.teams import CFB_NAMES
    n = Counter(t.get("location") for t in teams)
    out = []
    for t in teams:
        loc = t.get("location")
        if n[loc] == 1 or CFB_NAMES.get(t.get("displayName")) == loc:
            out.append(t)
    return out


def build(league: str, payload: dict) -> tuple[dict, list[str]]:
    """({code: urls}, [unplaced display names]). Pure given the payload."""
    to_code = coder(league)
    out, unplaced = {}, []
    teams = [e["team"] for e in payload["sports"][0]["leagues"][0]["teams"]]
    if league == "cfb":
        teams = _cfb_unambiguous(teams)
    for t in teams:
        u = urls(t)
        if not u:
            continue
        try:
            out[to_code(t)] = u
        except Exception:
            unplaced.append(t.get("displayName", "?"))
    return dict(sorted(out.items())), unplaced


def main(argv: list[str] | None = None) -> int:
    art = {"_about": ("ESPN team logos keyed by each model's team codes "
                      "(model/ingest/logos.py). Display only.")}
    for league, path in PATHS.items():
        got, unplaced = build(league, _get(TEAMS.format(path=path)))
        art[league] = got
        note = f", {len(unplaced)} not in the tables" if unplaced else ""
        print(f"{league}: {len(got)} logos{note}")
        if unplaced and league != "cfb":
            print(f"  unplaced: {unplaced}")
    OUT.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
