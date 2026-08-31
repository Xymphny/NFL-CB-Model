"""
NFL play-by-play ingestion.

Source: nflverse-data GitHub releases (nflfastR-derived play-by-play).
This is real, public, free data — no API key required. Hosted as
release assets on github.com, which is reachable from this environment.
"""

import pandas as pd

NFLVERSE_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"

# Columns we actually need for Layer 1 — the full file has 370+ columns,
# most of which are irrelevant to this model. Trimming at load time keeps
# memory and load time down.
NEEDED_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "home_team", "away_team",
    "down", "ydstogo", "yardline_100", "play_type",
    "yards_gained", "first_down", "touchdown", "interception",
    "fumble_lost", "sack", "penalty",
    "qtr", "game_seconds_remaining", "score_differential",
    "posteam_score", "defteam_score", "wp",
    "special_teams_play", "punt_attempt", "field_goal_attempt",
    "kickoff_attempt", "extra_point_attempt",
]


def load_season(season: int) -> pd.DataFrame:
    """
    Load one season of NFL play-by-play data.

    Parameters
    ----------
    season : int
        e.g. 2023

    Returns
    -------
    pd.DataFrame
        Trimmed play-by-play data for the season.
    """
    url = NFLVERSE_PBP_URL.format(season=season)
    df = pd.read_csv(url, compression="gzip", low_memory=False)

    available = [c for c in NEEDED_COLUMNS if c in df.columns]
    missing = set(NEEDED_COLUMNS) - set(available)
    if missing:
        print(f"[nfl_pbp] warning: columns not found in source data: {missing}")

    df = df[available].copy()

    # Keep only actual scrimmage plays for Layer 1 (special teams handled
    # separately in Section 3.7 of the spec, not built yet in this pass).
    df = df[df["play_type"].isin(["pass", "run"])].copy()

    return df


if __name__ == "__main__":
    data = load_season(2023)
    print(f"Loaded {len(data)} scrimmage plays for 2023")
    print(data.head())
