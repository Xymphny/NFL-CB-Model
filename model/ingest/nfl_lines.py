#!/usr/bin/env python3
"""NFL closing lines and prices from nflverse, retained as raw data.

WHY A SEPARATE FILE
model/ingest/nfl_nflverse.py keeps final scores only, and data/raw/nfl/
nflverse_games.parquet is an audited frame other checks depend on -- widening
it would change a file those checks were written against. The lines are a
different object (the market, not the result) and get their own file.

WHAT THEY ARE FOR
The market-relative grade (model/grade_market_weight.py): does the model know
anything the closing price does not? Answering that needs the price as well as
the line, so the spread and moneyline ODDS are kept, not just the numbers --
a devigged probability cannot be computed from a line alone.

nflverse's convention, checked against this model's in model/
test_market_blending.py: spread_line is positive when the HOME team is
favoured. A home handicap in the usual betting sense is therefore
-spread_line.

Writes data/raw/nfl/nflverse_lines.parquet.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "nfl" / "nflverse_lines.parquet"
URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

COLS = ["game_id", "season", "game_type", "week", "gameday", "location",
        "home_team", "away_team", "home_score", "away_score",
        "spread_line", "home_spread_odds", "away_spread_odds",
        "home_moneyline", "away_moneyline",
        "total_line", "over_odds", "under_odds"]


def select(raw: pd.DataFrame) -> pd.DataFrame:
    """Played games with a spread. Pure."""
    df = raw[[c for c in COLS if c in raw.columns]]
    df = df[df.home_score.notna() & df.away_score.notna() & df.spread_line.notna()]
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if OUT.exists() and not a.force:
        print(f"{OUT.name} present; --force to refetch")
        return 0
    with urllib.request.urlopen(URL, timeout=60) as r:
        df = select(pd.read_csv(BytesIO(r.read())))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    priced = df.home_spread_odds.notna().mean()
    print(f"{len(df)} games {df.season.min()}-{df.season.max()}, "
          f"{priced:.1%} with spread prices -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
