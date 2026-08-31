"""
NFL schedule ingestion.

Source: nflverse-data GitHub releases, same ecosystem as nfl_pbp.py.
This is a separate file from play-by-play — it has one row per game,
not one row per play — and carries exactly the fields the points-
prediction layer needs that play-by-play doesn't have: rest days,
closing lines, weather, and neutral-site location.
"""

import pandas as pd

NFLVERSE_SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

NEEDED_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday",
    "away_team", "away_score", "home_team", "home_score", "result", "total",
    "location",              # "Home" or "Neutral" — Section 11.2's neutral-site flag
    "away_rest", "home_rest",
    "spread_line", "total_line",   # closing lines — Section 9.3's historical backtest source
    "temp", "wind", "roof",
]


def load_schedules(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Load NFL schedule/game-level data.

    Parameters
    ----------
    seasons : list of int, optional
        If given, filters to these seasons. Otherwise returns all seasons
        in the source file.
    """
    df = pd.read_csv(NFLVERSE_SCHEDULES_URL, low_memory=False, usecols=lambda c: c in NEEDED_COLUMNS)

    missing = set(NEEDED_COLUMNS) - set(df.columns)
    if missing:
        print(f"[nfl_schedules] warning: columns not found in source data: {missing}")

    df = df.copy()

    if seasons is not None:
        df = df[df["season"].isin(seasons)].copy()

    # Regular season only for now — playoff seeding/simulation (Section 12)
    # is a separate, later piece of work.
    df = df[df["game_type"] == "REG"].copy()

    df["is_neutral_site"] = df["location"].eq("Neutral")

    return df


def get_current_week(season: int) -> int:
    """
    Auto-detect the current week for a season, so the weekly cron job
    doesn't need a manually-updated CURRENT_WEEK env var every week.

    Definition: the most recent week that has at least one completed
    game (gameday has passed AND a score exists). Falls back to week 1
    if the season hasn't started yet.
    """
    from datetime import date

    df = load_schedules(seasons=[season])
    df["gameday"] = pd.to_datetime(df["gameday"])
    today = pd.Timestamp(date.today())

    completed = df[(df["gameday"] <= today) & df["home_score"].notna()]
    if completed.empty:
        return 1
    return int(completed["week"].max())


if __name__ == "__main__":
    data = load_schedules(seasons=[2023])
    print(f"Loaded {len(data)} regular-season games for 2023")
    print(f"Neutral-site games: {data['is_neutral_site'].sum()}")
    print(data[["home_team", "away_team", "home_rest", "away_rest", "spread_line", "total_line"]].head())
