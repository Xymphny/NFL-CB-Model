#!/usr/bin/env python3
"""NFL final scores from nflverse, to cross-check a cache nobody had checked.

WHY
ADR 0021 repaired 315 corrupt CFB scores found only because a second source
was pulled, and closed by naming the remaining gap: the NFL walk-forward cache
has no second source here and has never been cross-checked at all. That cache
produces MARGIN_SD = 13.2979, backs the ATS validation, and backs ADR 0009's
finding that NFL dispersion varies by season at p = 0.014.

The CFB fault was invisible to every statistical check and to a tie check:
9.19% of rows carried a FROZEN score, and only the 4.39% that happened to land
level could be seen. There is no reason to assume a different pipeline for a
different league behaved better, and no reason to assume it behaved worse.
Either way the answer is one join away.

nflverse's games.csv is the same file model/ngs_coefficient_replay.py already
reads for actual results, so this adds no new dependency -- only a check that
was never run.

Writes data/raw/nfl/nflverse_games.parquet.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "nfl" / "nflverse_games.parquet"
URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

COLS = ["game_id", "season", "game_type", "week", "gameday",
        "home_team", "away_team", "home_score", "away_score", "result",
        "overtime"]


def fetch() -> pd.DataFrame:
    with urllib.request.urlopen(URL, timeout=60) as r:
        raw = r.read()
    df = pd.read_csv(BytesIO(raw))
    df = df[[c for c in COLS if c in df.columns]]
    # Played games only. A scheduled row carries no score and is an absent
    # result, not a 0-0 one -- the distinction ADR 0020 was written about.
    df = df[df.home_score.notna() & df.away_score.notna()]
    df["home_score"] = df.home_score.astype(int)
    df["away_score"] = df.away_score.astype(int)
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if OUT.exists() and not a.force:
        print(f"{OUT.name}: present, skipping")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_parquet(OUT, index=False)
    reg = df[df.game_type == "REG"]
    print(f"{len(df)} played games, {len(reg)} regular season, "
          f"seasons {int(df.season.min())}-{int(df.season.max())} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
