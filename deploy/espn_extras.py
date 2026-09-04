"""
Two ESPN additions beyond the injuries fallback, both verified live
against real responses on 2026-09-04:

1. FPI matchup projections (NFL): ESPN's own pregame win probability
   per game, via scoreboard -> per-game summary. Displayed as an
   independent third opinion next to model and market. NOT edge -- FPI
   is public and priced in -- but a model that's an outlier against
   BOTH the market and FPI deserves more suspicion than one that's an
   outlier against the market alone, and showing the triangle is the
   kind of transparency a trustworthy board owes its users.
   (~17 requests per run: 1 scoreboard + 1 summary per game.)

2. College football injuries: same endpoint family and shape as the
   NFL injuries fallback -- but with a crucial semantic difference
   documented here so nobody mistakes it later: college injury
   reporting is NOT league-mandated, so ESPN carries only what
   conferences/teams disclose (3 teams had reports at verification
   time). A team ABSENT from the feed is "no report available", never
   "healthy" -- absent teams get None, not an empty list, and the UI
   renders those differently.

Everything soft-fails; context never blocks a board.
"""

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
CFB_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries"

# ESPN abbreviations that differ from nflverse.
ESPN_ABBR_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}


def _to_nflverse(abbr):
    return ESPN_ABBR_TO_NFLVERSE.get(abbr, abbr)


def parse_fpi_summary(summary, home_abbr, away_abbr):
    """Pure parser (tested against a captured response). gameProjection
    arrives as a string percentage; pairs may not sum to exactly 100."""
    p = summary.get("predictor") or {}
    home = p.get("homeTeam") or {}
    proj = home.get("gameProjection")
    if proj is None:
        return None
    try:
        return {"home": _to_nflverse(home_abbr), "away": _to_nflverse(away_abbr),
                "fpi_home_prob": round(float(proj) / 100.0, 4)}
    except (TypeError, ValueError):
        return None


def fetch_fpi_map(timeout=15):
    """{(home_abbr, away_abbr): fpi_home_prob} for the current week."""
    try:
        sb = requests.get(SCOREBOARD_URL, timeout=timeout).json()
        out = {}
        for ev in sb.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
            home = ((teams.get("home") or {}).get("team") or {}).get("abbreviation")
            away = ((teams.get("away") or {}).get("team") or {}).get("abbreviation")
            if not home or not away:
                continue
            summary = requests.get(SUMMARY_URL.format(event_id=ev["id"]), timeout=timeout).json()
            row = parse_fpi_summary(summary, home, away)
            if row:
                out[(row["home"], row["away"])] = row["fpi_home_prob"]
        return out
    except Exception as e:
        print(f"[espn_extras] FPI soft-fail: {e}")
        return {}


def fetch_cfb_injuries(rating_teams, timeout=15):
    """{ratings_team_name: [{player, position, status}]} for teams WITH
    a disclosed report only. Reuses the injuries parser and the CFB
    longest-prefix name mapper. Absent team => no key => UI shows
    'No report available'."""
    try:
        from deploy.game_context import parse_espn_injuries
        from deploy.cfb_odds_watch import map_odds_names_to_ratings
        payload = requests.get(CFB_INJURIES_URL, timeout=timeout).json()
        espn_names = [{"home_team": t.get("displayName"), "away_team": None}
                      for t in payload.get("injuries", [])]
        mapping, _ = map_odds_names_to_ratings(espn_names, rating_teams)
        name_map = {espn: ours for espn, ours in mapping.items()}
        return parse_espn_injuries(payload, name_map)
    except Exception as e:
        print(f"[espn_extras] CFB injuries soft-fail: {e}")
        return {}
