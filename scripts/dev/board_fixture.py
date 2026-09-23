#!/usr/bin/env python3
"""SYNTHETIC board fixtures for designing and screenshotting the site.

Runs the real export (scripts/export_board.py) against a synthetic odds
snapshot on real NBA and NHL opening nights, so every number on a card is the
core's -- but the PRICES are invented. Written to frontend/public/dev-fixtures/
(gitignored) and loaded only with ?fixture=1. Never copied into data/site.
"""
import json, shutil, sys, tempfile, dataclasses
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import export_board as X
import export_record as XR
from coverline.execution.bronze import BronzeStore
from coverline.leagues.nba.teams import NBA_NAMES
from coverline.leagues.nhl.teams import NHL_NAMES

OUT = ROOT / "frontend" / "public" / "dev-fixtures"
rng = np.random.default_rng(7)
BOOKS = ("pinnacle", "fanduel", "draftkings", "betmgm", "caesars")


def names(table):
    out = {}
    for n, c in table.items():
        out.setdefault(c, n)
    return out


def event(eid, sport, start, home, away, spread, total, ml_home):
    bms = []
    for b in BOOKS:
        j = rng.normal(0, 0.02)
        pt = spread + rng.choice([0, 0, 0, 0.5, -0.5]) if spread is not None else None
        markets = []
        if spread is not None:
            markets.append({"key": "spreads", "outcomes": [
                {"name": home, "price": round(1.91 + j, 2), "point": pt},
                {"name": away, "price": round(1.91 - j, 2), "point": -pt}]})
        away_dec = 1 / max(0.05, (1.045 - 1 / ml_home))
        markets.append({"key": "h2h", "outcomes": [
            {"name": home, "price": round(ml_home + j, 2)},
            {"name": away, "price": round(away_dec - j, 2)}]})
        if total is not None:
            markets.append({"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.91, "point": total},
                {"name": "Under", "price": 1.91, "point": total}]})
        bms.append({"key": b, "markets": markets})
    return {"id": eid, "sport_key": sport, "commence_time": start,
            "home_team": home, "away_team": away, "bookmakers": bms}


def build(league, day, src, model, sport, table, R, spread_scale, total):
    store = BronzeStore(Path(tempfile.mkdtemp()))
    games = src.slate(day)
    nm = names(table)
    for k, at in enumerate(("T12:00:00Z", "T21:30:00Z")):
        payload = []
        for g in games:
            d = model.predict(g.game_id, "now")
            mu = d.margin_mean()
            p_home = 1 / (1 + np.exp(-mu / spread_scale))
            nudge = rng.normal(0, 0.08)
            start = (src.schedule.loc[g.game_id].tip if league == "nba"
                     else src.schedule.loc[g.game_id].start).strftime("%Y-%m-%dT%H:%M:%SZ")
            spread = (-round((mu + rng.normal(0, 3.5) - k * 0.5) * 2) / 2 + 0.5) if league == "nba" else None
            ml = 1 / min(0.85, max(0.15, p_home + nudge))
            payload.append(event(f"ev-{g.game_id}", sport, start, nm[g.home], nm[g.away],
                                 spread, total, round(ml, 2)))
        store.write_snapshot(sport=sport, captured_at=f"{day}{at}", payload=payload, cost=1, source_url="synthetic")
    b = X.build(league, store, R, day=day)
    b["synthetic"] = "SYNTHETIC PRICES for design work -- not real odds"
    b["status"] = X.summarise(b)
    return b


def main():
    from coverline.leagues.nba import live as nba
    from coverline.leagues.nhl import live as nhl
    OUT.mkdir(parents=True, exist_ok=True)
    R = X._runner()
    nsrc = nba.NBALiveSource.load(2027)
    R.LEAGUES["nba"] = dataclasses.replace(R.LEAGUES["nba"], loader=lambda day: (nba.build_model(nsrc), nsrc, nsrc.slate(day)))
    hsrc = nhl.NHLLiveSource.load(2027)
    R.LEAGUES["nhl"] = dataclasses.replace(R.LEAGUES["nhl"], loader=lambda day: (nhl.build_model(hsrc), hsrc, hsrc.slate(day)))
    for league, day, src, model, sport, table, scale, total in (
            ("nba", "2026-10-21", nsrc, nba.build_model(nsrc), "basketball_nba", NBA_NAMES, 7.0, None),
            ("nhl", "2026-10-01", hsrc, nhl.build_model(hsrc), "icehockey_nhl", NHL_NAMES, 1.5, 6.5)):
        b = build(league, day, src, model, sport, table, R, scale, total)
        (OUT / f"board_{league}.json").write_text(json.dumps(X._clean(b), indent=1))
        print(league, b["status"])
    for league in ("nfl", "cfb", "mlb"):
        shutil.copy(ROOT / "data" / "site" / f"board_{league}.json", OUT / f"board_{league}.json")
    for f in ("record.json", "gates.json"):
        shutil.copy(ROOT / "data" / "site" / f, OUT / f)
    shutil.copy(ROOT / "data" / "prop_grades" / "summary.json", OUT / "prop_grades_summary.json")


if __name__ == "__main__":
    main()
