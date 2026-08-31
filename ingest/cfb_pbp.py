"""
CFB play-by-play ingestion via raw HTTP requests to the CollegeFootballData
(CFBD) API — decision made in Section 7 (raw requests, not the official
`cfbd` Python client, for full control over caching/rate limits on the
free tier's 1,000 calls/month).

IMPORTANT: this module has NOT been run or tested in the build sandbox —
api.collegefootballdata.com is not reachable from that environment's
network allowlist. Test this against a real API key in your own
environment before trusting it.

Get a free API key at: https://collegefootballdata.com/key
"""

import os
import time
import requests
import pandas as pd

CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Free tier is 1,000 calls/month — cache aggressively and batch by week
# rather than pulling per-game, to stay well under that.
_session = requests.Session()


def _get(path: str, params: dict, api_key: str) -> list:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    resp = _session.get(f"{CFBD_BASE_URL}{path}", headers=headers, params=params, timeout=30)
    resp.raise_for_status()

    remaining = resp.headers.get("X-CallLimit-Remaining")
    if remaining is not None:
        print(f"[cfb_pbp] API calls remaining this month: {remaining}")

    return resp.json()


def load_week(season: int, week: int, season_type: str = "regular", api_key: str = None) -> pd.DataFrame:
    """
    Load one week of FBS play-by-play data.

    Parameters
    ----------
    season : int
    week : int
    season_type : "regular" or "postseason"
    api_key : str, optional
        Falls back to CFBD_API_KEY environment variable if not provided.
    """
    api_key = api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise ValueError(
            "No CFBD API key provided. Pass api_key=, or set the "
            "CFBD_API_KEY environment variable. Get a free key at "
            "https://collegefootballdata.com/key"
        )

    plays = _get(
        "/plays",
        params={"year": season, "week": week, "seasonType": season_type, "classification": "fbs"},
        api_key=api_key,
    )
    df = pd.DataFrame(plays)

    # Being courteous to the free-tier rate limit if this is called in a loop
    # across many weeks — CFBD asks for reasonable request pacing.
    time.sleep(0.25)

    return df


def load_season(season: int, weeks: range = range(1, 16), api_key: str = None) -> pd.DataFrame:
    """
    Load a full season by looping over weeks. NOTE: this will burn through
    the free tier's 1,000 calls/month fast if run repeatedly during
    development — cache the result to disk rather than re-fetching.
    """
    frames = []
    for wk in weeks:
        print(f"[cfb_pbp] fetching week {wk}...")
        try:
            frames.append(load_week(season, wk, api_key=api_key))
        except requests.HTTPError as e:
            print(f"[cfb_pbp] week {wk} failed: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def map_to_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CFBD's field names onto the same column names nfl_pbp.load_season()
    produces, so downstream model/ratings.py code (bucketing, scoring,
    opponent adjustment) works identically for both leagues without
    league-specific branching.

    NOTE: field name mapping below is based on CFBD's documented /plays
    schema but has not been validated against a live response in this
    sandbox — double check against real output before relying on it.
    """
    rename_map = {
        "offense": "posteam",
        "defense": "defteam",
        "down": "down",
        "distance": "ydstogo",
        "yardsToGoal": "yardline_100",
        "yardsGained": "yards_gained",
        "playType": "play_type",
        "week": "week",
        "gameId": "game_id",
    }
    available = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=available)

    # CFBD's playType is a free-text description (e.g. "Rush", "Pass
    # Reception", "Pass Incompletion") rather than nflverse's clean
    # "run"/"pass" enum — this needs a real mapping table built against
    # actual observed values, not guessed.
    if "play_type" in df.columns:
        df["play_type"] = df["play_type"].str.lower()
        df.loc[df["play_type"].str.contains("rush", na=False), "play_type"] = "run"
        df.loc[df["play_type"].str.contains("pass", na=False), "play_type"] = "pass"

    return df


if __name__ == "__main__":
    # Example usage — requires CFBD_API_KEY to be set.
    data = load_week(2023, 1)
    print(f"Loaded {len(data)} plays")
    print(data.columns.tolist())
