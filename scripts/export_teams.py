#!/usr/bin/env python3
"""Team pages for the site (dashboard v2 items 7 and 14). Display only.

Writes, into data/site/:
  teams_nfl.json            ratings snapshot + teams.json, every stat ranked
  teams_nba.json            the 2025-26 season file the model reads
  pitchers_parks_mlb.json   the MLB walk-forward state replayed to today
  attack_defence_nhl.json   2025-26 no-pull rates; NOT used by the model

EVERY RANK IS COMPUTED HERE. The site shows them and orders nothing. Ranks
are 1 = best, with the direction stated per stat: lower is better for
defence VOA, EPA and success allowed; schedule rank 1 is the HARDEST. A null
stat has no rank and shows "--". Ties share the better rank.

Nothing here is a model input, and the pages say which numbers the model
uses and which it does not.

    python scripts/export_teams.py [--league nfl ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SITE = ROOT / "data" / "site"
LEAGUES = ("nfl", "nba", "mlb", "nhl")

NFL_FOOTNOTE = ("This early in a season the rating and the efficiency numbers can disagree. "
                "Only the rating moves the number the board prices with.")

#: (key, source field, higher is better)
NFL_STATS = {
    "model": [("offense_voa", "offense_voa", True), ("defense_voa", "defense_voa", False),
              ("special_teams_voa", "special_teams_voa", True)],
    "efficiency": [("epa_per_play", "epa_per_play_offense", True),
                   ("epa_per_play_allowed", "epa_per_play_allowed", False),
                   ("success_rate", "success_rate_offense", True),
                   ("success_rate_allowed", "success_rate_allowed", False)],
    "ball_security": [("turnover_margin", "turnover_margin", True),
                      ("red_zone_points_per_trip", "red_zone_points_per_trip", True)],
}

#: NBA conferences. Stable; the season file carries none.
NBA_EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NY", "ORL",
            "PHI", "TOR", "WSH"}


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def ranks(values: dict, higher_is_better: bool = True) -> dict:
    """{team: rank} with 1 = best; a null value gets None; ties share the
    better rank. Pure."""
    have = {t: v for t, v in values.items() if _num(v) is not None}
    order = sorted(have.values(), reverse=higher_is_better)
    out = {t: None for t in values}
    for t, v in have.items():
        out[t] = order.index(v) + 1
    return out


def stat_block(rows: dict, spec: list) -> dict:
    """{team: {key: {value, rank, better}}} for one group of stats."""
    out = {t: {} for t in rows}
    for key, field, hib in spec:
        vals = {t: _num(r.get(field)) for t, r in rows.items()}
        rk = ranks(vals, hib)
        for t in rows:
            out[t][key] = {"value": None if vals[t] is None else round(vals[t], 4),
                           "rank": rk[t], "better": "higher" if hib else "lower"}
    return out


# ------------------------------------------------------------------ NFL ----

def latest_ratings(season: int, root: Path = ROOT) -> tuple[dict, Path] | tuple[None, None]:
    files = sorted(glob.glob(str(root / "data" / "ratings" / f"{season}-week-*.json")))
    if not files:
        return None, None
    p = Path(files[-1])
    return json.loads(p.read_text()), p


def division_places(teams: dict) -> dict:
    """{abbr: place} within each division: win %, then point differential."""
    out = {}
    divs: dict = {}
    for a, t in teams.items():
        divs.setdefault(t.get("division"), []).append(a)
    for members in divs.values():
        def key(a):
            r = teams[a].get("record") or {}
            g = (r.get("w", 0) + r.get("l", 0) + r.get("t", 0)) or 1
            return (-(r.get("w", 0) + 0.5 * r.get("t", 0)) / g, -(r.get("pf", 0) - r.get("pa", 0)))
        for i, a in enumerate(sorted(members, key=key)):
            out[a] = i + 1
    return out


def build_nfl(season: int | None = None, root: Path = ROOT) -> dict:
    season = season or date.today().year
    snap, path = latest_ratings(season, root)
    site_teams = json.loads((root / "data" / "site" / "teams.json").read_text())["teams"]
    if snap is None:
        return {"league": "nfl", "status": "no ratings yet", "teams": []}
    rated = {r["team"]: r for r in snap["ratings"]}
    info = {t["abbr"]: t for t in (site_teams.values() if isinstance(site_teams, dict) else site_teams)}
    abbrs = sorted(set(rated) | set(info))
    rating_rank = ranks({a: rated.get(a, {}).get("total_rating") for a in abbrs}, True)
    blocks = {g: stat_block({a: rated.get(a, {}) for a in abbrs}, spec)
              for g, spec in NFL_STATS.items()}
    sched = {"played_sos": True, "remaining_sos": True}          # 1 = hardest
    sos_rank = {k: ranks({a: (info.get(a) or {}).get(k) for a in abbrs}, True) for k in sched}
    places = division_places(info)
    out = []
    for a in abbrs:
        r, t = rated.get(a, {}), info.get(a, {})
        played = [g for g in t.get("schedule", []) if g.get("result")]
        upcoming = [g for g in t.get("schedule", []) if not g.get("result")]
        counts: dict = {}
        for i in t.get("injuries") or []:
            counts[i["status"]] = counts.get(i["status"], 0) + 1
        out.append({
            "abbr": a, "name": t.get("name"), "division": t.get("division"),
            "conference": t.get("conference"), "logo": t.get("logo"),
            "record": t.get("record"), "division_place": places.get(a),
            "rating": {"value": _num(r.get("total_rating")), "rank": rating_rank[a], "of": len(abbrs),
                       "p05": _num(r.get("rating_p05")), "p95": _num(r.get("rating_p95"))},
            **{g: blocks[g][a] for g in NFL_STATS},
            "schedule": {k.replace("_sos", ""): {"value": _num(t.get(k)), "rank": sos_rank[k][a],
                                                  "better": "rank 1 is the hardest"}
                         for k in sched},
            "qb1": ((t.get("depth_chart") or {}).get("QB") or [None])[0],
            "injury_counts": counts,
            "results": played, "next": upcoming[:3],
        })
    return {"league": "nfl", "season": season, "ratings_version": path.name,
            "ratings_computed_at": snap.get("computed_at"), "ratings_week": snap.get("week"),
            "footnote": NFL_FOOTNOTE, "teams": out}


# ------------------------------------------------------------------ NBA ----

def build_nba(season: int = 2026, root: Path = ROOT) -> dict:
    from coverline.leagues.nba import live
    data = str(root / "data" / "raw" / "sportsdataverse" / "nba_{year}.parquet")
    src = live.NBALiveSource.load(season, data=data)
    h = src.history[src.history.season == season].sort_values("tip")
    if h.empty:
        return {"league": "nba", "season": season, "status": "no games", "teams": []}
    final, _ = src.ratings_before(h.tip.max() + pd.Timedelta(hours=12))
    opening, _ = src.ratings_before(h.tip.min())
    carry = src.hyperparameters["carryover"]
    teams = sorted(set(h.home) | set(h.away))
    rec = {}
    for t in teams:
        games = []
        for r in h.itertuples():
            if t not in (r.home, r.away):
                continue
            home = r.home == t
            pf, pa = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
            games.append({"date": str(r.tip)[:10], "home": home, "pf": int(pf), "pa": int(pa)})
        w = sum(g["pf"] > g["pa"] for g in games)
        n = len(games)
        rec[t] = {
            "w": w, "l": n - w, "win_pct": w / n if n else None,
            "net": sum(g["pf"] - g["pa"] for g in games) / n if n else None,
            "pts_for": sum(g["pf"] for g in games) / n if n else None,
            "pts_against": sum(g["pa"] for g in games) / n if n else None,
            "rating": final.get(t),
            "home": _wl([g for g in games if g["home"]]), "road": _wl([g for g in games if not g["home"]]),
            "last_10": _wl(games[-10:]), "within_5": _wl([g for g in games if abs(g["pf"] - g["pa"]) <= 5]),
            "opening_rating": None if opening.get(t) is None else opening[t] * carry,
            "first_game": games[0]["date"] if games else None,
        }
    rk = {k: ranks({t: rec[t][k] for t in teams}, hib) for k, hib in
          (("win_pct", True), ("net", True), ("pts_for", True), ("pts_against", False), ("rating", True))}
    conf = {t: "East" if t in NBA_EAST else "West" for t in teams}
    place = {}
    for c in ("East", "West"):
        members = sorted((t for t in teams if conf[t] == c),
                         key=lambda t: (-(rec[t]["win_pct"] or 0), -(rec[t]["net"] or 0)))
        place.update({t: i + 1 for i, t in enumerate(members)})
    out = []
    for t in teams:
        r = rec[t]
        out.append({"team": t, "conference": conf[t], "conference_place": place[t],
                    "record": {"w": r["w"], "l": r["l"], "rank": rk["win_pct"][t]},
                    **{k: {"value": None if r[k] is None else round(r[k], 3), "rank": rk[k][t],
                           "better": "lower" if k == "pts_against" else "higher"}
                       for k in ("net", "pts_for", "pts_against", "rating")},
                    "home": r["home"], "road": r["road"], "last_10": r["last_10"],
                    "within_5": r["within_5"],
                    "opening_rating": None if r["opening_rating"] is None else round(r["opening_rating"], 3),
                    "first_game": r["first_game"]})
    return {"league": "nba", "season": season, "of": len(teams), "outlook": "Not simulated yet.",
            "rating_note": "Model rating in points of margin, after the season's last game.",
            "teams": out}


def _wl(games: list[dict]) -> str:
    w = sum(g["pf"] > g["pa"] for g in games)
    return f"{w}-{len(games) - w}"


# ------------------------------------------------------------------ MLB ----

MLB_NOTE = ("How a game is priced: each side's expected runs are its offence against the "
            "other side's pitching -- the starter for 58% of the outs, the bullpen for 42% -- "
            "times the park, every rating shrunk toward league average until it has earned "
            "its own record.")


def build_mlb(day: str | None = None, root: Path = ROOT) -> dict:
    from coverline.leagues.mlb import live
    from model.mlb_model import CRED_OUTS_PEN, CRED_OUTS_SP
    day = day or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    sched, pit = live.load_caches(root / "model")
    st = live.state_before(sched, pit, day)
    teams = sorted(set(sched[sched.season == sched.season.max()].home_team))
    parks = (sched.sort_values("date").groupby("home_team").park.last()).to_dict()
    recent = pit[(pit.date < day) & (pit.date >= (pd.Timestamp(day) - pd.Timedelta(days=3))
                                     .strftime("%Y-%m-%d")) & (pit.started == 0)]
    workload = recent.groupby("team").outs.sum().to_dict()
    rows = {}
    for t in teams:
        _, pen_outs = st.pen_outs[t]
        rows[t] = {"offense": st.offense(t), "pen_ra27": st.pen_ra27(t),
                   "pen_cred": pen_outs / (pen_outs + CRED_OUTS_PEN),
                   "park_factor": st.park_factor(parks.get(t)), "park": parks.get(t),
                   "relief_outs_3d": int(workload.get(t, 0))}
    rk = {"offense": ranks({t: rows[t]["offense"] for t in teams}, True),
          "pen_ra27": ranks({t: rows[t]["pen_ra27"] for t in teams}, False),
          "park_factor": ranks({t: rows[t]["park_factor"] for t in teams}, True)}
    team_rows = [{"team": t, "park": rows[t]["park"],
                  "offense": {"value": round(rows[t]["offense"], 3), "rank": rk["offense"][t],
                              "better": "higher", "unit": "runs per game"},
                  "pen_ra27": {"value": round(rows[t]["pen_ra27"], 3), "rank": rk["pen_ra27"][t],
                               "better": "lower", "credibility": round(rows[t]["pen_cred"], 3)},
                  "park_factor": {"value": round(rows[t]["park_factor"], 3),
                                  "rank": rk["park_factor"][t], "better": "rank 1 is the most runs"},
                  "bullpen_workload": {"relief_outs_last_3_days": rows[t]["relief_outs_3d"],
                                       "in_price": False}}
                 for t in teams]
    probables = []
    try:
        slate = live.MLBLiveSource.load(day)
        for g in slate.payload.get("games", []):
            for side in ("away", "home"):
                pid, name = g.get(f"{side}_sp"), g.get(f"{side}_sp_name")
                row = {"game_key": g["game_key"], "team": g[f"{side}_team"],
                       "team_name": g.get(f"{side}_name"), "pitcher": name,
                       "start_utc": g.get("start_utc")}
                if not pid:
                    row.update({"status": "refused", "reason": "no probable starter: TBD"})
                else:
                    _, outs = st.sp_outs[pid]
                    row.update({"status": "priced", "ra27": round(st.sp_ra27(pid), 3),
                                "own_record_share": round(outs / (outs + CRED_OUTS_SP), 3)})
                probables.append(row)
    except Exception as e:
        probables = []
        print(f"[export_teams] mlb probables soft-fail: {e}")
    return {"league": "mlb", "as_of": day, "league_rpg": round(st.league_rpg(), 3),
            "note": MLB_NOTE, "teams": team_rows, "probables": probables}


# ------------------------------------------------------------------ NHL ----

def build_nhl(root: Path = ROOT) -> dict:
    from model import fit_nhl_rules
    d = fit_nhl_rules.load((2026,), raw=root / "data" / "raw" / "nhl")
    teams = sorted(set(d.home_team_abbr) | set(d.away_team_abbr))
    agg = {t: {"gp": 0, "nopull_for": 0, "nopull_against": 0, "pull_for": 0, "pull_against": 0,
               "points": 0, "past_regulation": 0} for t in teams}
    for r in d.itertuples():
        past = r.last_period_type in ("OT", "SO")
        for t, gf, ga, pf, pa, won in (
                (r.home_team_abbr, r.nopull_h, r.nopull_a, r.pull_h, r.pull_a, r.home_score > r.away_score),
                (r.away_team_abbr, r.nopull_a, r.nopull_h, r.pull_a, r.pull_h, r.away_score > r.home_score)):
            a = agg[t]
            a["gp"] += 1
            a["nopull_for"] += gf
            a["nopull_against"] += ga
            a["pull_for"] += pf
            a["pull_against"] += pa
            a["points"] += 2 if won else (1 if past else 0)
            a["past_regulation"] += int(past)
    per = {t: {"gf": a["nopull_for"] / a["gp"], "ga": a["nopull_against"] / a["gp"]}
           for t, a in agg.items()}
    rk = {"gf": ranks({t: per[t]["gf"] for t in teams}, True),
          "ga": ranks({t: per[t]["ga"] for t in teams}, False),
          "points": ranks({t: agg[t]["points"] for t in teams}, True)}
    out = [{"team": t, "gp": agg[t]["gp"],
            "nopull_gf": {"value": round(per[t]["gf"], 3), "rank": rk["gf"][t], "better": "higher"},
            "nopull_ga": {"value": round(per[t]["ga"], 3), "rank": rk["ga"][t], "better": "lower"},
            "pulled_goalie_for": agg[t]["pull_for"], "pulled_goalie_against": agg[t]["pull_against"],
            "points": {"value": agg[t]["points"], "rank": rk["points"][t], "better": "higher"},
            "past_regulation": agg[t]["past_regulation"]}
           for t in teams]
    return {"league": "nhl", "season": 2026,
            "banner": ("NOT USED BY THE MODEL. NHL ratings reset every season; these are last "
                       "season's rates, shown until every team has 10 games this season."),
            "rates_note": "No-pull goals per game: the rates the model is built on, with "
                          "empty-net goals removed.", "teams": out}


def with_logos(league: str, art: dict) -> dict:
    """Each team's logo URL from data/logos.json, where one is known."""
    try:
        logo = {k: v["dark"] for k, v in
                json.loads((ROOT / "data" / "logos.json").read_text()).get(league, {}).items()}
    except (OSError, ValueError, KeyError):
        logo = {}
    for t in art.get("teams", []):
        code = t.get("abbr") or t.get("team")
        if code in logo:
            t["logo"] = logo[code]
    for p in art.get("probables", []):
        if p.get("team") in logo:
            p["logo"] = logo[p["team"]]
    return art


BUILDERS = {"nfl": ("teams_nfl.json", build_nfl), "nba": ("teams_nba.json", build_nba),
            "mlb": ("pitchers_parks_mlb.json", build_mlb),
            "nhl": ("attack_defence_nhl.json", build_nhl)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=LEAGUES, action="append")
    ap.add_argument("--out", default=str(SITE))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for league in a.league or LEAGUES:
        name, fn = BUILDERS[league]
        try:
            art = with_logos(league, fn())
        except Exception as exc:                  # a page never blocks the others
            print(f"[export_teams] {league} failed: {type(exc).__name__}: {exc}")
            failed.append(league)
            continue
        (out / name).write_text(json.dumps(art, indent=1, default=str) + "\n")
        print(f"{name}: {len(art.get('teams', []))} teams")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
