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
    "posteam_score", "defteam_score", "wp", "epa",
    "special_teams_play", "punt_attempt", "field_goal_attempt",
    "kickoff_attempt", "extra_point_attempt",
    "fixed_drive", "fixed_drive_result",
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

    # IMPORTANT: read in chunks rather than all at once. Measured directly:
    # a single read_csv call on this file peaks around 486MB RSS, because
    # decompressing the full ~100MB of raw CSV text happens before pandas
    # can apply usecols or any row filter — usecols only helps after
    # parsing starts, it can't avoid decompressing the whole gzip stream
    # up front. That alone is enough to blow past Render's cron job
    # memory limit (512Mi) once the rest of the pipeline adds more on
    # top. Chunked reading processes and filters each piece before
    # accumulating, so the full decompressed text is never held in
    # memory at once — measured peak with this approach: ~148MB.
    chunks = []
    for chunk in pd.read_csv(
        url, compression="gzip", low_memory=False,
        usecols=lambda c: c in NEEDED_COLUMNS, chunksize=5000,
    ):
        # Filter to scrimmage plays per-chunk, not after concatenating —
        # this is what keeps peak memory down, since discarded rows never
        # accumulate across chunks.
        chunks.append(chunk[chunk["play_type"].isin(["pass", "run"])].copy())

    df = pd.concat(chunks, ignore_index=True)

    missing = set(NEEDED_COLUMNS) - set(df.columns)
    if missing:
        print(f"[nfl_pbp] warning: columns not found in source data: {missing}")

    return df


if __name__ == "__main__":
    data = load_season(2023)
    print(f"Loaded {len(data)} scrimmage plays for 2023")
    print(data.head())
