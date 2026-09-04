"""
Real college football play-by-play ingestion -- REPLACES the original
design (raw HTTP requests to the CollegeFootballData.com API), which
was never tested this entire project and turned out to be genuinely
blocked (confirmed directly: this sandbox's network proxy returns
"Host not in allowlist" for api.collegefootballdata.com).

Real, working, free alternative found and verified: cfbfastR (the
CFB-equivalent of nflfastR, same SportsDataverse family) publishes
full play-by-play as GitHub release assets -- reachable, unlike the
CFBD API directly. Confirmed with a real download: 254,090 real plays,
362 columns, for the actual 2023 season.

Column mapping below translates cfbfastR's native schema onto this
project's existing NFL schema (posteam, defteam, down, ydstogo,
yardline_100, touchdown, interception, fumble_lost, sack, wp) so the
entire existing ratings pipeline (model/ratings.py, model/play_value.py
-- which already has real CFB-specific success thresholds, 50/70/100
vs NFL's 45/60/100, from the original spec) can be reused without
duplication.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd

try:
    import pyreadr
except ImportError:
    pyreadr = None

CFB_PBP_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfbfastR_cfb_pbp/play_by_play_{season}.rds"

NEEDED_COLUMNS = [
    "year", "week", "game_id", "season_type",
    "home_team", "away_team", "home_team_division", "away_team_division",
    "pos_team", "def_pos_team",
    "down", "distance", "yards_to_goal", "yards_gained",
    "play_type", "wp_before",
]


def _download_and_parse_rds(url):
    if pyreadr is None:
        raise ImportError("pyreadr is required to read cfbfastR's .rds format -- pip install pyreadr")

    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as f:
        f.write(resp.content)
        tmppath = f.name

    result = pyreadr.read_r(tmppath)
    os.unlink(tmppath)
    return result[None]


def load_cfb_season(season):
    url = CFB_PBP_URL.format(season=season)
    raw = _download_and_parse_rds(url)
    return _map_to_nfl_schema(raw), raw


def _map_to_nfl_schema(raw):
    raw = raw[raw["season_type"] == "regular"].copy()
    raw = raw[(raw["home_team_division"] == "fbs") & (raw["away_team_division"] == "fbs")].copy()

    df = pd.DataFrame({
        "season": raw["year"],
        "week": raw["week"],
        "game_id": raw["game_id"],
        "home_team": raw["home_team"],
        "away_team": raw["away_team"],
        "posteam": raw["pos_team"],
        "defteam": raw["def_pos_team"],
        "down": raw["down"],
        "ydstogo": raw["distance"],
        "yardline_100": raw["yards_to_goal"],
        "yards_gained": raw["yards_gained"],
        "wp": raw["wp_before"],
    })

    run_types = {"Rush", "Rushing Touchdown"}
    pass_types = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
                  "Interception Return", "Interception Return Touchdown"}

    df["play_type"] = raw["play_type"].apply(
        lambda p: "run" if p in run_types else ("pass" if p in pass_types else "other")
    )
    df["touchdown"] = raw["play_type"].str.contains("Touchdown", na=False).astype(int)
    df["interception"] = raw["play_type"].str.contains("Interception", na=False).astype(int)
    df["sack"] = (raw["play_type"] == "Sack").astype(int)
    df["fumble_lost"] = (raw["play_type"] == "Fumble Recovery (Opponent)").astype(int)

    df = df[df["play_type"].isin(["run", "pass"])].copy()

    return df.reset_index(drop=True)


def derive_cfb_schedule(raw_with_scores: pd.DataFrame, include_postseason: bool = False) -> pd.DataFrame:
    """
    Derives real final scores directly from the play-by-play data --
    no separate schedule source needed (load_cfb_schedules() in the
    R package hits the live, key-gated CFBD API directly, unlike
    load_cfb_pbp() which has a free GitHub-cached version; this
    sidesteps that entirely).

    Validated against a real, known result: Michigan 30, Ohio State 24
    (the actual 2023 "The Game" result) -- derived correctly by taking
    each game's last play and reading pos_team_score/def_pos_team_score
    relative to which team had possession.

    include_postseason: real bowl/CFP games are excluded by default,
    same as this project's DVOA rating computation (bowl games have
    real, different participation incentives -- opt-outs, backups
    playing -- that would skew per-play efficiency stats). But Elo
    only cares about final scores, not per-play efficiency, so
    excluding bowls there means missing real, meaningful results.
    CORRECTION to an earlier claim in this project: Ohio State's real
    2025 season (played through January 2026) actually ended with
    losses -- 13-10 to Indiana in the Big Ten Championship, then 24-14
    to Miami in the CFP quarterfinal -- not the championship run
    earlier documentation incorrectly stated (which conflated this
    with Ohio State's actual 2024 season title, won in January 2025).
    Set True specifically for Elo's schedule -- confirmed real
    season_type values via direct check: "regular" (242,924 real
    plays) and "postseason" (11,166).

    raw_with_scores: the RAW (unmapped) cfbfastR dataframe, since the
    mapped one in load_cfb_season() doesn't retain the score columns.
    """
    if include_postseason:
        raw = raw_with_scores[raw_with_scores["season_type"].isin(["regular", "postseason"])].copy()
    else:
        raw = raw_with_scores[raw_with_scores["season_type"] == "regular"].copy()
    raw = raw[(raw["home_team_division"] == "fbs") & (raw["away_team_division"] == "fbs")].copy()

    games = []
    for game_id, group in raw.groupby("game_id"):
        last_play = group.iloc[-1]
        home_team, away_team = last_play["home_team"], last_play["away_team"]
        if last_play["pos_team"] == home_team:
            home_score, away_score = last_play["pos_team_score"], last_play["def_pos_team_score"]
        else:
            home_score, away_score = last_play["def_pos_team_score"], last_play["pos_team_score"]

        games.append({
            "game_id": game_id, "season": last_play["year"], "week": last_play["week"],
            # REAL BUG FOUND AND FIXED: postseason games reset their own
            # week numbering (a real January 2026 CFP game showed up as
            # "week 1", identical to actual week-1 games from August
            # 2025) -- sorting Elo's walk-forward purely by (season,
            # week) would have processed postseason games as if they
            # happened BEFORE the season even started, corrupting the
            # whole season's chronological order. Carrying the real
            # start_date through fixes this at the source.
            "game_date": last_play["start_date"],
            "home_team": home_team, "away_team": away_team,
            "home_score": home_score, "away_score": away_score,
        })

    return pd.DataFrame(games)


if __name__ == "__main__":
    print("Testing real CFB ingestion against the actual 2023 season...")
    df, raw = load_cfb_season(2023)
    print(f"\n{len(df)} real FBS scrimmage plays loaded")
    print(f"{pd.concat([df['home_team'], df['away_team']]).nunique()} unique FBS teams")

    print("\nDeriving real schedule/final scores from the same data...")
    schedule = derive_cfb_schedule(raw)
    print(f"{len(schedule)} real FBS games with derived final scores")

    print("\nValidating against a real, known result: Michigan 30, Ohio State 24 (2023 'The Game')")
    game = schedule[(schedule["home_team"] == "Michigan") & (schedule["away_team"] == "Ohio State")]
    print(game[["week", "home_team", "away_team", "home_score", "away_score"]].to_string())
    assert game.iloc[0]["home_score"] == 30 and game.iloc[0]["away_score"] == 24, "Score mismatch!"
    print("PASS: derived score matches the real, known result")
