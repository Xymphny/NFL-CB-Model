"""
Probable starting pitchers + daily schedule from the official MLB
Stats API (statsapi.mlb.com -- free, keyless; unreachable from the
build sandbox, reachable from Render like every other live feed).

This is the qb_status.py of baseball, load-bearing rather than
annotative: MLB lines are QUOTED conditional on listed pitchers, so
every snapshot must carry the probables it was priced against, and a
scratch invalidates the stored line for CLV purposes. Manual
override: data/qb_overrides.json gains an "mlb" section with the
same semantics (starter corrects, alert flags).
"""

import os
from datetime import date

import requests

SCHEDULE_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}"
                "&hydrate=probablePitcher,team&gameTypes=R")


def fetch_probables(for_date=None, timeout=25):
    """[{game_pk, kickoff, home/away team abbr + name, home/away
    probable (name or None)}] for the date. Soft-fail to []."""
    d = (for_date or date.today()).isoformat()
    try:
        payload = requests.get(SCHEDULE_URL.format(d=d), timeout=timeout).json()
    except Exception as e:
        print(f"[mlb_probables] soft-fail: {e}")
        return []
    games = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            def side(s):
                t = g["teams"][s]
                return {
                    "team": t["team"].get("abbreviation") or t["team"].get("teamName"),
                    "name": t["team"].get("name"),
                    "probable": (t.get("probablePitcher") or {}).get("fullName"),
                }
            h, a = side("home"), side("away")
            games.append({
                "game_pk": g.get("gamePk"),
                "kickoff": g.get("gameDate"),
                "status": (g.get("status") or {}).get("abstractGameState"),
                "home_team": h["team"], "home_name": h["name"], "home_probable": h["probable"],
                "away_team": a["team"], "away_name": a["name"], "away_probable": a["probable"],
            })
    return games


def apply_overrides(games):
    try:
        from deploy.qb_status import load_overrides
        ov = load_overrides("mlb")
    except Exception:
        ov = {}
    for g in games:
        for side in ("home", "away"):
            team_ov = ov.get(g[f"{side}_team"]) or {}
            if team_ov.get("starter"):
                g[f"{side}_probable"] = team_ov["starter"]
            if team_ov.get("alert"):
                g[f"{side}_alert"] = team_ov["alert"]
    return games
