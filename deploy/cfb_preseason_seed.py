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
in-season signal.

Elo: computed from the repo's own walk-forward Elo over the real
2023-2025 schedule cache (final external audit caught the original
flat-1500 seeding, which discarded available data and rendered every
Elo tile in the UI as "1500, vs baseline: 0"). Final 2025 values are
regressed 50% toward 1500, mirroring the ratings regression.

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
REGRESSION_KEPT = 0.5  # share of (elo - 1500) carried into the new season


def _carryover_elo():
    """Final 2025 Elo per team from the schedule cache, regressed 50%
    toward 1500. Soft-fail to flat 1500 if the cache is unreadable."""
    try:
        from model.elo_rating import compute_elo_walk_forward
        cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "model", "cfb_schedule_cache.csv")
        schedule = pd.read_csv(cache_path)
        _, final_elo = compute_elo_walk_forward(schedule)
        return {team: round(1500 + REGRESSION_KEPT * (elo - 1500), 1) for team, elo in final_elo.items()}
    except Exception as e:
        print(f"[cfb_preseason_seed] Elo carryover unavailable ({e}); seeding flat 1500")
        return {}


def build_seed(season=2026, week=1, output_dir="./data"):
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "model", "cfb_2025_final_ratings.csv")
    df = pd.read_csv(src).rename(columns={"Unnamed: 0": "team"})
    elo_map = _carryover_elo()
    matched = sum(1 for t in df["team"] if t in elo_map)
    print(f"[cfb_preseason_seed] Elo carryover matched {matched}/{len(df)} teams")

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
                "elo_rating": elo_map.get(row["team"], 1500),
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
