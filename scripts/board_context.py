#!/usr/bin/env python3
"""Per-game context for the board: key players, venue, weather, injuries,
rest and travel. DISPLAY ONLY.

Every item carries `in_price`: whether the league's model actually reads it.
It is set from the model's inputs (each league's GameFeatures and the vector
the live path prices with -- see IN_PRICE and its tests), never from what
seems likely. Nothing here is a model input, and nothing here may move a
number: board prices are computed before context is attached, and a test
builds a board with and without context and compares every model field.

SOFT-FAIL, ALWAYS. Every source is a fetch that can fail, and a failure
reads as "no report" or "unknown" -- never an empty list that looks healthy,
never a zero, never a blocked export. With COINFLIP_CONTEXT_OFFLINE=1 (the
test suite sets it) no network call is made at all.

THE CHOICES ARE MADE HERE, NOT IN THE SITE. Which injuries show, in what
order, how many per card: decided below and exported, so the site only lays
out what it is given.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

VENUES = ROOT / "data" / "venues.json"
OVERRIDES = ROOT / "data" / "qb_overrides.json"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/{path}/injuries"
ESPN_PATH = {"nhl": "hockey/nhl", "mlb": "baseball/mlb", "cfb": "football/college-football"}

#: Key player role per league (the brief's "key player").
ROLE = {"nfl": "QB1", "cfb": "QB1", "nhl": "starting goalie", "nba": "starter"}

#: Injury cards: how many rows per team, by the game's tier.
SHOWN = {"play": 4, "coin_flip": 0}
SHOWN_DEFAULT = 3

STATUS_ORDER = {"Out": 0, "Injured Reserve": 0, "IR": 0, "60-Day IL": 0, "15-Day IL": 1,
                "10-Day IL": 1, "7-Day IL": 1, "Suspended": 1, "Doubtful": 1,
                "Questionable": 2, "Day-To-Day": 3, "Probable": 4}
NFL_POSITIONS = ["QB", "WR", "RB", "TE", "OT", "G/C", "DE", "DT", "LB", "CB", "S"]
NFL_POS_ALIAS = {"T": "OT", "OT": "OT", "G": "G/C", "C": "G/C", "OG": "G/C", "OL": "G/C",
                 "DL": "DT", "NT": "DT", "ILB": "LB", "OLB": "LB", "MLB": "LB",
                 "FS": "S", "SS": "S", "DB": "CB"}

INJURY_NOTE = {
    "cfb": "Not in the price. College injury reporting is not mandated: a missing "
           "report means unknown, not healthy.",
}
INJURY_NOTE_DEFAULT = "Not in the price."


def offline() -> bool:
    return os.environ.get("COINFLIP_CONTEXT_OFFLINE") == "1"


# ------------------------------------------------------------- in_price ----

def in_price() -> dict[str, dict[str, bool]]:
    """Which context items each league's model reads, from its actual inputs.

    NFL rest: `rest_diff` is a GameFeatures field and the rating-only vector
    every live price uses weights it (0.1310). NFL weather: wind feeds only
    the totals model, and totals are withheld from primary_markets. MLB park:
    the live source's expected runs carry the home park factor. Everything
    else: no such input exists.
    """
    from coverline.leagues.cfb import model as cfb
    from coverline.leagues.mlb import live as mlb_live
    from coverline.leagues.nba import model as nba
    from coverline.leagues.nfl import model as nfl
    nfl_fields = {f.name for f in fields(nfl.GameFeatures)}
    nfl_rest = ("rest_diff" in nfl_fields
                and nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY.get("rest_diff", 0) != 0)
    # Wind reaches only the totals model; a total is priced only if the NFL
    # model lists it among the markets it prices.
    nfl_wind = "total" in nfl.NFLModel.primary_markets.fget(None)
    cfb_fields = {f.name for f in fields(cfb.GameFeatures)}
    nba_fields = {f.name for f in fields(nba.GameFeatures)}
    import inspect
    mlb_park = 'g["park"]' in inspect.getsource(mlb_live.MLBLiveSource.features)
    no = dict.fromkeys(("key_player", "venue", "weather", "injuries", "rest", "travel"), False)
    return {
        "nfl": {**no, "rest": nfl_rest, "weather": bool(nfl_wind)},
        "cfb": {**no, "rest": "rest_diff" in cfb_fields},
        "mlb": {**no, "venue": mlb_park},
        "nba": {**no, "rest": "rest_diff" in nba_fields},
        "nhl": dict(no),
    }


# ------------------------------------------------------------- injuries ----

def _pos_rank(league: str, pos: str) -> int:
    if league != "nfl":
        return 0
    p = NFL_POS_ALIAS.get(pos, pos)
    return NFL_POSITIONS.index(p) if p in NFL_POSITIONS else len(NFL_POSITIONS)


def order_injuries(league: str, rows: list[dict]) -> list[dict]:
    """Worst status first; within a status, key positions first (NFL: QB,
    WR, RB, TE, OT, G/C, DE, DT, LB, CB, S). Pure."""
    return sorted(rows, key=lambda r: (STATUS_ORDER.get(r.get("status"), 9),
                                       _pos_rank(league, r.get("position") or ""),
                                       r.get("player") or ""))


def injury_display(rows: list[dict] | None, tier: str | None) -> dict:
    """What one team's injury list shows on a card of this tier. `None`
    rows is "no report" (unknown), never "healthy". Pure."""
    if rows is None:
        return {"status": "no report", "shown": [], "more": 0, "counts": {}}
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n = SHOWN.get(tier, SHOWN_DEFAULT)
    return {"status": "reported", "shown": rows[:n], "more": max(0, len(rows) - n),
            "counts": counts}


def fetch_espn_injuries(league: str) -> dict | None:
    """ESPN league injury report -> {team display name: [rows]}, or None."""
    if offline():
        return None
    try:
        import requests
        r = requests.get(ESPN_INJURIES.format(path=ESPN_PATH[league]), timeout=20)
        r.raise_for_status()
        out = {}
        for team in r.json().get("injuries", []):
            rows = []
            for inj in team.get("injuries", []):
                a = inj.get("athlete") or {}
                status = normalise_status(inj.get("status") or "")
                if status and a.get("displayName"):
                    rows.append({"player": a["displayName"],
                                 "position": (a.get("position") or {}).get("abbreviation", ""),
                                 "status": status})
            out[team.get("displayName")] = rows
        return out
    except Exception as e:
        print(f"[board_context] {league} injuries soft-fail: {e}")
        return None


def normalise_status(s: str) -> str | None:
    s = s.strip()
    if not s or s == "Active":
        return None
    if s == "Suspension":
        return "Suspended"
    if s.endswith("-IL") or s.endswith(" IL"):
        return s.replace("-IL", " IL")
    return s


def nfl_injuries(season: int, week: int) -> tuple[dict | None, str | None]:
    if offline():
        return None, None
    try:
        from deploy.game_context import get_injury_report
        report, source = get_injury_report(season, week)
        return (report or None), source
    except Exception as e:
        print(f"[board_context] nfl injuries soft-fail: {e}")
        return None, None


# ------------------------------------------------------------- weather -----

def fetch_weather(points: list[tuple[float, float, str]]) -> list[dict | None]:
    """Open-Meteo for many (lat, lon, kickoff hour ET) in ONE request.
    Returns one parsed forecast (or None) per point."""
    if offline() or not points:
        return [None] * len(points)
    try:
        import requests
        from deploy.game_context import parse_open_meteo
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": ",".join(f"{p[0]:.4f}" for p in points),
            "longitude": ",".join(f"{p[1]:.4f}" for p in points),
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "forecast_days": 8, "timezone": "America/New_York"}, timeout=20)
        r.raise_for_status()
        body = r.json()
        body = body if isinstance(body, list) else [body]
        return [parse_open_meteo(b, p[2]) for b, p in zip(body, points)]
    except Exception as e:
        print(f"[board_context] weather soft-fail: {e}")
        return [None] * len(points)


def weather_entry(kind: str, wx: dict | None = None, source: str | None = None,
                  in_price: bool = False) -> dict:
    if kind == "indoors":
        return {"status": "indoors", "in_price": in_price}
    if not wx:
        return {"status": "no report", "in_price": in_price}
    return {"status": "forecast", **wx, "source": source, "in_price": in_price}


# ----------------------------------------------------- venue and travel ----

def venues() -> dict:
    try:
        return json.loads(VENUES.read_text())
    except (OSError, ValueError):
        return {}


def miles(a: tuple[float, float] | None, b: tuple[float, float] | None) -> int | None:
    """Great-circle distance in miles, or None if either end is unknown."""
    if not a or not b:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return int(round(3958.8 * 2 * math.asin(math.sqrt(h))))


# --------------------------------------------------------- key players -----

def overrides(league: str) -> dict:
    try:
        return json.loads(OVERRIDES.read_text()).get(league, {}) or {}
    except (OSError, ValueError):
        return {}


def nfl_qb_alerts(season: int, week: int) -> dict:
    if offline():
        return {}
    from deploy.qb_status import get_qb_alerts_detailed
    return get_qb_alerts_detailed(season, week)


def key_players(league: str, home: str, away: str, alerts: dict) -> list[dict]:
    """[{team, role, text, source}] for either side. `alerts` maps team to
    {"text", "source"}; a manual override beats every feed. Pure given its
    inputs."""
    ov = overrides(league)
    out = []
    for team in (away, home):
        a = ({"text": ov[team]["alert"], "source": "manual override"}
             if (ov.get(team) or {}).get("alert") else alerts.get(team))
        if a:
            out.append({"team": team, "role": ROLE.get(league, "key player"),
                        "text": a["text"], "source": a["source"]})
    return out


def nba_starter_alerts(players: dict | None) -> dict:
    """Starters (started at least half of the last-10 window) listed Out,
    Doubtful or Questionable, from the Players export. {team: alert}."""
    if not players:
        return {}
    out = {}
    for t in players.get("teams", []):
        n = max(t.get("games_in_window") or 0, 1)
        starters = {r["player"] for r in t.get("rotation", []) if r.get("starts", 0) * 2 >= n
                    and n >= 3}
        hit = [i for i in t.get("injuries", [])
               if i["player"] in starters and i["status"] in ("Out", "Doubtful", "Questionable")]
        if hit:
            out[t["team"]] = {"text": "; ".join(f"{i['player']} {i['status']}" for i in hit),
                              "source": "ESPN injury report"}
    return out


# ----------------------------------------------------------- the bundle ----

def base(league: str) -> dict:
    """Items every card carries, with the league's in_price flags."""
    ip = in_price()[league]
    return {"key_player": [], "venue": None,
            "weather": {"status": "no report", "in_price": ip["weather"]},
            "injuries": {"status": "no report", "in_price": ip["injuries"],
                         "note": INJURY_NOTE.get(league, INJURY_NOTE_DEFAULT)},
            "rest": None, "travel": None}


def team_injuries(rows_by_team: dict | None, team: str, league: str) -> list[dict] | None:
    """A team's ordered rows; None when the report is missing entirely, [] when
    the team is on a report and lists nobody."""
    if rows_by_team is None:
        return None
    rows = rows_by_team.get(team)
    if rows is None:
        return [] if league != "cfb" else None     # a CFB absence is unknown
    return order_injuries(league, rows)


def finish_injuries(ctx: dict, tier: str | None) -> None:
    """Apply the card's display rule once the tier is known. In place."""
    inj = ctx.get("injuries") or {}
    if "home_rows" not in inj:
        return
    inj["home"] = injury_display(inj.pop("home_rows"), tier)
    inj["away"] = injury_display(inj.pop("away_rows"), tier)
    inj["counts_only"] = SHOWN.get(tier, SHOWN_DEFAULT) == 0


# ------------------------------------------------------------ per league ----
# Each builder is called ONCE per export with the league's slate, fetches its
# sources once (injuries once, weather in one batched request), and returns
# {game_id: context}. The export attaches it after pricing.

def _injuries_block(league: str, rows_by_team: dict | None, home: str, away: str,
                    source: str | None, ip: bool) -> dict:
    h, a = team_injuries(rows_by_team, home, league), team_injuries(rows_by_team, away, league)
    if h is None and a is None:
        return {"status": "no report", "in_price": ip,
                "note": INJURY_NOTE.get(league, INJURY_NOTE_DEFAULT)}
    return {"status": "reported", "source": source, "in_price": ip,
            "note": INJURY_NOTE.get(league, INJURY_NOTE_DEFAULT),
            "home_rows": h, "away_rows": a}


def _by_code(raw: dict | None, to_code) -> dict | None:
    """{display name: rows} -> {model team code: rows}; unmapped names drop."""
    if raw is None:
        return None
    out = {}
    for name, rows in raw.items():
        try:
            out[to_code(name)] = rows
        except Exception:
            continue
    return out


def nfl(wk: pd.DataFrame, game_ids: list[str], season: int, week: int,
        starts: dict[str, str]) -> dict[str, dict]:
    ip = in_price()["nfl"]
    alerts = nfl_qb_alerts(season, week)
    report, inj_source = nfl_injuries(season, week)
    ven = venues().get("nfl", {})
    now = pd.Timestamp.now(tz="UTC")
    rows = list(zip(game_ids, wk.itertuples()))
    # Weather for open-air games within the forecast horizon, ONE request.
    ask, idx = [], {}
    for gid, r in rows:
        roof = str(getattr(r, "roof", "") or "").lower()
        neutral = str(getattr(r, "location", "Home")) == "Neutral"
        k = pd.Timestamp(starts[gid]) if starts.get(gid) else None
        if roof in ("dome", "closed") or neutral or r.home_team not in ven or k is None:
            continue
        if not (now <= k <= now + pd.Timedelta(days=7)):
            continue
        idx[gid] = len(ask)
        et = k.tz_convert("America/New_York").strftime("%Y-%m-%dT%H:00")
        ask.append((ven[r.home_team]["lat"], ven[r.home_team]["lon"], et))
    wx = fetch_weather(ask)
    out = {}
    for gid, r in rows:
        ctx = base("nfl")
        roof = str(getattr(r, "roof", "") or "").lower() or None
        neutral = str(getattr(r, "location", "Home")) == "Neutral"
        ctx["key_player"] = key_players("nfl", r.home_team, r.away_team, alerts)
        ctx["venue"] = {"name": getattr(r, "stadium", None), "roof": roof,
                        "surface": getattr(r, "surface", None), "neutral": neutral,
                        "in_price": ip["venue"]}
        if roof in ("dome", "closed"):
            ctx["weather"] = weather_entry("indoors", in_price=ip["weather"])
        elif gid in idx:
            ctx["weather"] = weather_entry("forecast", wx[idx[gid]], "Open-Meteo", ip["weather"])
        ctx["injuries"] = _injuries_block("nfl", report, r.home_team, r.away_team,
                                          inj_source, ip["injuries"])
        hr, ar = getattr(r, "home_rest", None), getattr(r, "away_rest", None)
        if pd.notna(hr) and pd.notna(ar):
            ctx["rest"] = {"home_days": int(hr), "away_days": int(ar), "in_price": ip["rest"],
                           "note": "The rest difference is one of the model's inputs."}
        xy = lambda t: (ven[t]["lat"], ven[t]["lon"]) if t in ven else None  # noqa: E731
        # A neutral site is not at the home team's stadium, and its
        # coordinates are not in the table: unknown, not zero.
        ctx["travel"] = {"away_miles": None if neutral else miles(xy(r.away_team), xy(r.home_team)),
                         "in_price": ip["travel"]}
        out[gid] = ctx
    return out


def cfb(src, game_ids: list[str]) -> dict[str, dict]:
    ip = in_price()["cfb"]
    from coverline.leagues.cfb.teams import TABLE
    raw = fetch_espn_injuries("cfb")
    report = _by_code(raw, TABLE.to_code)
    s = src.schedule
    out = {}
    for gid in game_ids:
        g = s.loc[gid]
        ctx = base("cfb")
        ctx["key_player"] = key_players("cfb", g.home_team, g.away_team, {})
        name = g.get("venue") if "venue" in s.columns else None
        if isinstance(name, str) and name:
            ctx["venue"] = {"name": name, "city": g.get("venue_city"), "state": g.get("venue_state"),
                            "neutral": bool(g.neutral_site), "in_price": ip["venue"]}
            if g.get("indoor") is True:
                ctx["weather"] = weather_entry("indoors", in_price=ip["weather"])
        ctx["injuries"] = _injuries_block("cfb", report, g.home_team, g.away_team,
                                          "ESPN", ip["injuries"])
        out[gid] = ctx
    return out


def mlb(src, game_ids: list[str]) -> dict[str, dict]:
    ip = in_price()["mlb"]
    from coverline.leagues.mlb.teams import TABLE
    report = _by_code(fetch_espn_injuries("mlb"), TABLE.to_code)
    prev_starts = _mlb_previous_starts(src)
    out = {}
    for gid in game_ids:
        g = src._game(gid)
        ctx = base("mlb")
        ctx["venue"] = {"name": g.get("venue_name"), "park": g.get("park"),
                        "in_price": ip["venue"],
                        "note": "The park factor is one of the model's inputs."}
        w = g.get("weather") or {}
        if w:
            ctx["weather"] = {"status": "report", "condition": w.get("condition"),
                              "temp_f": w.get("temp"), "wind": w.get("wind"),
                              "source": "MLB Stats API", "in_price": ip["weather"]}
        ctx["injuries"] = _injuries_block("mlb", report, g["home_team"], g["away_team"],
                                          "ESPN", ip["injuries"])
        start = pd.Timestamp(g["start_utc"]) if g.get("start_utc") else None
        flags = {}
        for side in ("home", "away"):
            p = prev_starts.get(g[f"{side}_team"])
            flags[f"{side}_day_after_night"] = bool(
                start is not None and p is not None and _is_night(p) and _is_day(start))
        ctx["rest"] = {**flags, "in_price": ip["rest"],
                       "note": "Day and night by the Eastern clock, so approximate on the West Coast."}
        out[gid] = ctx
    return out


def _is_night(t: pd.Timestamp) -> bool:
    et = t.tz_convert("America/New_York")
    return et.hour >= 18


def _is_day(t: pd.Timestamp) -> bool:
    et = t.tz_convert("America/New_York")
    return et.hour < 17


def _mlb_previous_starts(src) -> dict:
    """{team: start of its previous game} from the prior day's slate pull."""
    try:
        from coverline.leagues.mlb.live import SLATE_DIR
        pulls = sorted(p for p in SLATE_DIR.glob("*.json") if p.name[:10] < src.day)
        if not pulls:
            return {}
        last_day = pulls[-1].name[:10]
        latest = sorted(p for p in pulls if p.name.startswith(last_day))[-1]
        out = {}
        for x in json.loads(latest.read_text())["games"]:
            if x.get("start_utc"):
                t = pd.Timestamp(x["start_utc"])
                for team in (x["home_team"], x["away_team"]):
                    out[team] = max(out.get(team, t), t)
        return out
    except Exception:
        return {}


def _arena_travel(league: str, sched: pd.DataFrame, gid, home_col: str, away_col: str,
                  time_col: str) -> dict:
    """Rest days, back-to-backs, and the away side's miles from its previous
    game's arena (home arena if it has none this season)."""
    ven = venues().get(league, {})
    g = sched.loc[gid]
    xy = lambda team: (ven[team]["lat"], ven[team]["lon"]) if team in ven else None  # noqa: E731
    out = {}
    for side in ("home", "away"):
        team = g[home_col] if side == "home" else g[away_col]
        prior = sched[((sched[home_col] == team) | (sched[away_col] == team))
                      & (sched[time_col] < g[time_col])]
        if len(prior):
            last = prior.sort_values(time_col).iloc[-1]
            days = (g[time_col].normalize() - last[time_col].normalize()).days
            out[f"{side}_rest_days"] = int(days)
            out[f"{side}_back_to_back"] = bool(days <= 1)
            if side == "away":
                out["away_from"] = last[home_col]
    frm = out.pop("away_from", g[away_col])
    return {"rest": {**{k: v for k, v in out.items()}, "in_price": in_price()[league]["rest"]},
            "travel": {"away_miles": miles(xy(frm), xy(g[home_col])),
                       "in_price": in_price()[league]["travel"]}}


def nba(src, game_ids: list[str]) -> dict[str, dict]:
    ip = in_price()["nba"]
    players = _nba_players()
    report = None
    if players is not None:
        report = {t["team"]: [{"player": i["player"], "position": i.get("position", ""),
                               "status": i["status"]} for i in t.get("injuries", [])]
                  for t in players.get("teams", [])}
    alerts = nba_starter_alerts(players)
    ven = venues().get("nba", {})
    s = src.schedule
    out = {}
    for gid in game_ids:
        g = s.loc[gid]
        ctx = base("nba")
        ctx["key_player"] = key_players("nba", g.home, g.away, alerts)
        v = ven.get(g.home)
        ctx["venue"] = ({"name": None, "neutral": True, "in_price": ip["venue"]} if g.neutral
                        else {"name": v and v["name"], "city": v and v["city"], "neutral": False,
                              "in_price": ip["venue"]})
        ctx["weather"] = weather_entry("indoors", in_price=ip["weather"])
        ctx["injuries"] = _injuries_block("nba", report, g.home, g.away, "ESPN", ip["injuries"])
        ctx.update(_arena_travel("nba", s, gid, "home", "away", "tip"))
        ctx["players_link"] = True
        out[gid] = ctx
    return out


def _nba_players() -> dict | None:
    try:
        import export_players
        return export_players.build()
    except Exception as e:
        print(f"[board_context] nba players soft-fail: {e}")
        return None


def nhl(src, game_ids: list[str]) -> dict[str, dict]:
    ip = in_price()["nhl"]
    from coverline.leagues.nhl.teams import TABLE
    report = _by_code(fetch_espn_injuries("nhl"), TABLE.to_code)
    ven = venues().get("nhl", {})
    s = src.schedule
    out = {}
    for gid in game_ids:
        g = s.loc[gid]
        ctx = base("nhl")
        ctx["key_player"] = key_players("nhl", g.home_team_abbr, g.away_team_abbr, {})
        ctx["key_player_note"] = ("No automated goalie feed yet: a starter is flagged only when "
                                  "entered by hand.")
        v = ven.get(g.home_team_abbr)
        neutral = bool(g.get("neutral_site", False))
        ctx["venue"] = ({"name": None, "neutral": True, "in_price": ip["venue"]} if neutral
                        else {"name": v and v["name"], "city": v and v["city"], "neutral": False,
                              "in_price": ip["venue"]})
        # Outdoor specials are played away from the home arena; with no
        # venue feed their weather is unknown, not "indoors".
        ctx["weather"] = (base("nhl")["weather"] if neutral
                          else weather_entry("indoors", in_price=ip["weather"]))
        ctx["injuries"] = _injuries_block("nhl", report, g.home_team_abbr, g.away_team_abbr,
                                          "ESPN", ip["injuries"])
        ctx.update(_arena_travel("nhl", s, gid, "home_team_abbr", "away_team_abbr", "start"))
        out[gid] = ctx
    return out
