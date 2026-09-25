#!/usr/bin/env python3
"""Historical closing lines from ESPN's public odds records. Free, once.

WHY
MLB, NHL and NBA had no market grade: no free historical line source was
known, and the paid historical backfill costs ~90,000 Odds API credits for
one season. ESPN keeps, per game, the odds it displayed -- with explicit OPEN
and CLOSE blocks for each side's moneyline, spread (points and price) and the
total -- from late 2023 on (ESPN BET, then DraftKings; several books before).
This pulls them into committed files so model/grade_market_weight.py can
grade those leagues against the market now, not after 150 paper trades.

WHAT IT WRITES
data/raw/{nba,nhl,mlb}/espn_closes_{season}.parquet, one row per regular-
season game with a pre-game CLOSE block, from one provider:

    espn_id, date (UTC kickoff), home, away (ESPN display names),
    home_abbr, away_abbr, provider,
    home_ml, away_ml                  closing moneylines, American
    home_spread, home_spread_price,   closing spread from the HOME side, and
    away_spread_price                 both sides' prices
    total, over_price, under_price
    home_ml_open, away_ml_open

NO OPENING SPREAD. ESPN's open pointSpread carries the wrong sign on some
records (a +320 home underdog opened "-5.5"), so it is not stored: a field
that looks usable and is not is worse than an absent one.

WHICH PROVIDER
The first of PREFERRED with a close block; else any non-live provider with
one. In-play feeds ("... Live Odds") are never used: a live price is not a
close. Games with no close block are left out, and the count is printed.

ONE BOOK, NOT THE SHARPEST. A retail book's close is a noisier market than
a sharp book's. Good enough to grade a model against; recorded per row so a
grade can say what it was graded on.

POLITE BY DESIGN. An undocumented API: a few workers, a pause per request,
and the pull is one-shot -- a season already written is skipped.

    python model/ingest/espn_closes.py --league nhl --seasons 2024-2026
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates={day}&limit=200"
ODDS = ("https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}"
        "/events/{gid}/competitions/{gid}/odds")
LEAGUES = {
    # league: (espn sport, espn league, first day, last day, season ends next year)
    "nba": ("basketball", "nba", "10-01", "04-30", True),
    "nhl": ("hockey", "nhl", "09-25", "04-30", True),
    "mlb": ("baseball", "mlb", "03-15", "10-05", False),
}
PREFERRED = ("ESPN BET", "DraftKings", "Caesars Sportsbook", "Bet365", "BetfairSportsbook")
REGULAR = 2
WORKERS = 4
PAUSE_S = 0.1


def _get(url: str, retries: int = 5) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed: {url}") from last


def _american(block: dict | None, key: str) -> float | None:
    v = ((block or {}).get(key) or {}).get("american")
    if v in (None, "", "EVEN", "even"):
        return 100.0 if v in ("EVEN", "even") else None
    try:
        return float(str(v).replace("+", ""))
    except ValueError:
        return None


def parse_close(item: dict) -> dict | None:
    """One provider's odds record -> closing fields, or None without a close.
    Pure."""
    name = (item.get("provider") or {}).get("name") or ""
    if "live" in name.lower():
        return None
    h, a = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
    hc, ac = h.get("close"), a.get("close")
    if not hc or not ac:
        return None
    tc = item.get("close") or {}
    row = {
        "provider": name,
        "home_ml": _american(hc, "moneyLine"), "away_ml": _american(ac, "moneyLine"),
        "home_spread": _american(hc, "pointSpread"),
        "home_spread_price": _american(hc, "spread"), "away_spread_price": _american(ac, "spread"),
        "total": _american(tc, "total") if tc.get("total") else item.get("overUnder"),
        "over_price": _american(tc, "over"), "under_price": _american(tc, "under"),
        "home_ml_open": _american(h.get("open"), "moneyLine"),
        "away_ml_open": _american(a.get("open"), "moneyLine"),
    }
    if row["home_ml"] is None and row["home_spread"] is None:
        return None
    return row


def pick(items: list[dict]) -> dict | None:
    """The preferred provider's close among a game's odds records. Pure."""
    parsed = [p for p in (parse_close(i) for i in items) if p]
    # COMPLETENESS BEFORE BRAND. Before ESPN BET (2023) the first-choice book's
    # close often carried a spread and no moneyline, which dropped 1,121 MLB
    # games from a moneyline grade. A record with both closes beats one with
    # either, and only then does the book order apply.
    def rank(p):
        full = (p["home_ml"] is not None) + (p["home_spread"] is not None)
        book = PREFERRED.index(p["provider"]) if p["provider"] in PREFERRED else len(PREFERRED)
        return (-full, book)
    return min(parsed, key=rank) if parsed else None


def day_events(sport: str, league: str, day: str) -> list[dict]:
    sb = _get(SCOREBOARD.format(sport=sport, league=league, day=day))
    out = []
    for e in sb.get("events", []):
        if (e.get("season") or {}).get("type") != REGULAR:
            continue
        comp = (e.get("competitions") or [{}])[0]
        teams = {t.get("homeAway"): t.get("team") or {} for t in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        out.append({"espn_id": str(e["id"]), "date": e.get("date"),
                    "neutral_site": bool(comp.get("neutralSite")),
                    "home": teams["home"].get("displayName"), "away": teams["away"].get("displayName"),
                    "home_abbr": teams["home"].get("abbreviation"),
                    "away_abbr": teams["away"].get("abbreviation")})
    return out


def game_close(sport: str, league: str, ev: dict) -> dict | None:
    time.sleep(PAUSE_S)
    j = _get(ODDS.format(sport=sport, league=league, gid=ev["espn_id"]))
    items = []
    for it in j.get("items", []):
        items.append(it if "provider" in it else _get(it["$ref"]))
    c = pick(items)
    return {**ev, **c} if c else None


def season_days(league: str, season: int) -> list[str]:
    _, _, lo, hi, nxt = LEAGUES[league]
    start_year = season - 1 if nxt else season
    d0 = date.fromisoformat(f"{start_year}-{lo}")
    d1 = min(date.fromisoformat(f"{season}-{hi}"), date.today() - timedelta(days=1))
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def pull(league: str, season: int) -> pd.DataFrame:
    sport, lg = LEAGUES[league][:2]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        days = list(ex.map(lambda d: day_events(sport, lg, d), season_days(league, season)))
    events = list({e["espn_id"]: e for d in days for e in d}.values())
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(lambda e: game_close(sport, lg, e), events))
    got = [r for r in rows if r]
    print(f"{league} {season}: {len(events)} regular-season games, {len(got)} with a close "
          f"({len(events) - len(got)} without)", flush=True)
    df = pd.DataFrame(got)
    if len(df):
        df = df.sort_values(["date", "espn_id"]).reset_index(drop=True)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=sorted(LEAGUES), required=True)
    ap.add_argument("--seasons", required=True, help="Y-Y, the years the seasons END")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.seasons.split("-"))
    for y in range(lo, hi + 1):
        dest = ROOT / "data" / "raw" / a.league / f"espn_closes_{y}.parquet"
        if dest.exists() and not a.force:
            print(f"{a.league} {y}: present, skipping")
            continue
        df = pull(a.league, y)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        if len(df):
            print(f"  -> {dest.relative_to(ROOT)}  providers {df.provider.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
