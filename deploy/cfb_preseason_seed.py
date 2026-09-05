"""
Preseason CFB ratings seed -- the CFB equivalent of the NFL week-00
prior, existing to break a chicken-and-egg: the CFB odds watch needs a
ratings snapshot to price anything, the weekly job needs 2026
play-by-play to compute real ratings, and that data isn't published
yet at season start.

Method: 2025 final ratings regressed 50% toward zero (league mean).
Half regression is the standard prior for a sport with ~40% annual
roster churn -- and the evidence for humility is already enforced
downstream: the held-out backtest graded weeks 1-4 flags BELOW
breakeven, so the board caps every early-season verdict at Lean
regardless of what these seeded numbers claim. The snapshot is
labeled as carryover so nothing downstream can mistake it for
in-season signal. Elo seeds at 1500 flat (no stored 2025 Elo).

The real weekly job overwrites this the moment 2026 pbp publishes.
"""

import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from model.version import METHODOLOGY_VERSION

REGRESSION = 0.5


def build_seed(season=2026, week=1, output_dir="./data"):
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "model", "cfb_2025_final_ratings.csv")
    df = pd.read_csv(src).rename(columns={"Unnamed: 0": "team"})

    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "week": week,
        "methodology_version": METHODOLOGY_VERSION,
        "source": "preseason carryover: 2025 final ratings regressed 50% to mean; Elo 1500 flat",
        "ratings": [
            {
                "team": row["team"],
                "total_rating": round(row["total_rating"] * (1 - REGRESSION), 5),
                "offense_voa": round(row["offense_voa"] * (1 - REGRESSION), 5),
                "defense_voa": round(row["defense_voa"] * (1 - REGRESSION), 5),
                "elo_rating": 1500,
            }
            for _, row in df.iterrows()
        ],
    }
    out_dir = os.path.join(output_dir, "cfb_ratings")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{season}-week-{week:02d}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[cfb_preseason_seed] wrote {out}: {len(payload['ratings'])} teams (carryover prior)")
    return out


if __name__ == "__main__":
    build_seed(int(os.environ.get("SEASON", 2026)), int(os.environ.get("CFB_WEEK", 1)),
               os.environ.get("REPO_DATA_PATH", "./data"))
