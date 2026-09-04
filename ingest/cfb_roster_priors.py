"""
CFB roster intelligence from CFBD's FREE endpoints -- the early-season
informational features where CFB markets are genuinely weakest (~40%
annual roster churn means preseason and September lines lean on stale
information more than any NFL number ever does).

Pulls three feature families per team-season:
- /player/returning  -> returning production (total/offense/defense PPA
  returning, % of usage returning) -- the single best-documented
  early-season predictor class in CFB.
- /ratings/sp        -> SP+ overall/offense/defense as a market-aware
  baseline to test AGAINST (if our features add nothing over SP+,
  they add nothing).
- /player/portal     -> net transfer portal flow per team (counts and
  average rating of incoming vs outgoing).

NOT TESTED against the live API (api.collegefootballdata.com
unreachable from the build sandbox). Same drill as ingest/cfb_lines.py:
verify response shapes on first run, and the team-name join to the
ratings cache reports its own match rate so failures are loud.

Usage with the harness: build_roster_priors([2021, 2022, 2023]) writes
model/cfb_roster_priors.csv; model/cfb_ats_backtest.py automatically
merges it when present and reports whether the features improve
held-out ATS over the rating-only baseline -- the exact same
add-a-feature-class discipline as the NFL pressure/FTN experiments.

Env: CFBD_API_KEY (free tier).
"""

import os
import time

import pandas as pd
import requests

BASE = "https://api.collegefootballdata.com"


def _get(path, params, api_key):
    resp = requests.get(BASE + path, params=params,
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_season(season, api_key):
    rows = {}

    for item in _get("/player/returning", {"year": season}, api_key):
        rows.setdefault(item["team"], {"season": season, "team": item["team"]}).update({
            "ret_ppa_total": item.get("totalPPA"),
            "ret_ppa_pass": item.get("totalPassingPPA"),
            "ret_usage": item.get("usage"),
            "ret_pass_usage": item.get("passingUsage"),
        })
    time.sleep(0.5)

    for item in _get("/ratings/sp", {"year": season}, api_key):
        if item.get("team") is None:
            continue
        rows.setdefault(item["team"], {"season": season, "team": item["team"]}).update({
            "sp_overall": item.get("rating"),
            "sp_offense": (item.get("offense") or {}).get("rating"),
            "sp_defense": (item.get("defense") or {}).get("rating"),
        })
    time.sleep(0.5)

    portal_in, portal_out = {}, {}
    for p in _get("/player/portal", {"year": season}, api_key):
        rating = p.get("rating") or 0
        if p.get("destination"):
            portal_in.setdefault(p["destination"], []).append(rating)
        if p.get("origin"):
            portal_out.setdefault(p["origin"], []).append(rating)
    for team in set(portal_in) | set(portal_out):
        inc, out = portal_in.get(team, []), portal_out.get(team, [])
        rows.setdefault(team, {"season": season, "team": team}).update({
            "portal_net_count": len(inc) - len(out),
            "portal_net_rating": (sum(inc) - sum(out)),
        })

    return pd.DataFrame(list(rows.values()))


def build_roster_priors(seasons, out_path=None, api_key=None):
    api_key = api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY not set")
    out_path = out_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "cfb_roster_priors.csv")
    frames = []
    for season in seasons:
        print(f"Fetching CFB roster priors for {season}...")
        frames.append(fetch_season(season, api_key))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(df)} team-seasons, "
          f"{df['ret_ppa_total'].notna().sum() if 'ret_ppa_total' in df else 0} with returning production")
    return df


if __name__ == "__main__":
    build_roster_priors([2021, 2022, 2023])
