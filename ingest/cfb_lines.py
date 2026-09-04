"""
CFB closing lines from CFBD's free /lines endpoint -- the missing half
of the CFB market comparison. The ratings side has existed for a while
(cfb_full_walk_forward_cache.csv, 1,731 games); this supplies the real
market numbers to grade it against.

NOT TESTED AGAINST THE LIVE API from the sandbox this was written in
(api.collegefootballdata.com unreachable there). Before trusting any
backtest built on this, verify in order:
  1. Response shape: each game object carries a `lines` list, one entry
     per provider, each with `spread`, `overUnder`, `provider`.
  2. SIGN CONVENTION -- the single most dangerous thing here, and this
     project has already caught one real sign bug (4 of 16 Week 1 NFL
     lines had flipped home spreads). CFBD's `spread` is the HOME
     team's number: negative = home favored. This module NEGATES it so
     positive = home favored, matching this repo's convention
     everywhere else. Verify with 3 games you know: e.g. a ranked home
     team hosting a cupcake must come out large and positive.
  3. Spot-check `spread` here against closing numbers from a book you
     trust for the same games -- CFBD aggregates provider feeds and
     the last stored line is usually but not always the true close.

Env: CFBD_API_KEY (free tier: collegefootballdata.com/key).
"""

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com/lines"
PREFERRED_PROVIDERS = ["consensus", "DraftKings", "ESPN Bet", "Bovada"]


def _pick_line(lines):
    """One (spread, total) per game: preferred provider order, else the
    median across whatever providers exist."""
    if not lines:
        return None, None
    by_provider = {ln.get("provider"): ln for ln in lines}
    for name in PREFERRED_PROVIDERS:
        ln = by_provider.get(name)
        if ln and ln.get("spread") is not None:
            return float(ln["spread"]), (float(ln["overUnder"]) if ln.get("overUnder") is not None else None)
    spreads = [float(ln["spread"]) for ln in lines if ln.get("spread") is not None]
    totals = [float(ln["overUnder"]) for ln in lines if ln.get("overUnder") is not None]
    spread = float(pd.Series(spreads).median()) if spreads else None
    total = float(pd.Series(totals).median()) if totals else None
    return spread, total


def fetch_season_lines(season, api_key=None, season_type="regular", pause=0.6):
    api_key = api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY not set")
    headers = {"Authorization": f"Bearer {api_key}"}

    rows = []
    # Week-by-week keeps each response small and rate-limit friendly.
    for week in range(1, 16):
        resp = requests.get(
            BASE_URL,
            params={"year": season, "week": week, "seasonType": season_type},
            headers=headers, timeout=30,
        )
        resp.raise_for_status()
        for game in resp.json():
            spread, total = _pick_line(game.get("lines", []))
            if spread is None:
                continue
            rows.append({
                "season": season,
                "week": game.get("week", week),
                "home_team": game["homeTeam"],
                "away_team": game["awayTeam"],
                # NEGATED: CFBD spread is the home number (negative =
                # home favored); repo convention is positive = home
                # favored. See sign-verification checklist above.
                "spread_line": -spread,
                "total_line": total,
                "home_score": game.get("homeScore"),
                "away_score": game.get("awayScore"),
            })
        time.sleep(pause)
    return pd.DataFrame(rows)


def build_lines_cache(seasons, out_path=None):
    out_path = out_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "cfb_lines_cache.csv")
    frames = []
    for season in seasons:
        print(f"Fetching CFB lines for {season}...")
        frames.append(fetch_season_lines(season))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(df)} games with lines")
    return df


if __name__ == "__main__":
    build_lines_cache([2021, 2022, 2023])
