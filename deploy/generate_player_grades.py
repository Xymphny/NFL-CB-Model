"""
Generates player-level Layer 2 grades (QB/WR-TE/RB) for the dashboard
-- the original Section 4/5 design (a real player-evaluation output,
kept separate from team ratings) never had anywhere to actually be
seen until now.

Writes an immutable snapshot, same architecture as ratings/divergence
-- data/player_grades/{season}-week-{week}.json -- so the same
manifest-based "latest wins" logic on the frontend just works without
any new plumbing.
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.layer2_ngs import compute_player_grades


def main(season, week, output_dir="./data"):
    print(f"Computing real player grades for {season}, through week {week}...")
    grades = compute_player_grades(season, through_week=week if week else None)

    def to_records(df, position_label):
        df = df.reset_index().rename(columns={"index": "player_display_name"})
        records = []
        for _, row in df.iterrows():
            records.append({
                "player": row["player_display_name"],
                "position": position_label,
                "grade": round(float(row["grade"]), 1),
                "sample": int(row["sample"]),
            })
        return records

    payload = {
        "season": season,
        "week": week,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "grades": {
            "QB": to_records(grades["QB"], "QB"),
            "WR_TE": to_records(grades["WR_TE"], "WR/TE"),
            "RB": to_records(grades["RB"], "RB"),
        },
    }

    output_subdir = os.path.join(output_dir, "player_grades")
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f"{season}-week-{week:02d}.json")
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {output_file}")
    print(f"  {len(payload['grades']['QB'])} QBs, {len(payload['grades']['WR_TE'])} WR/TE, {len(payload['grades']['RB'])} RBs")
    return output_file


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./data"
    main(season, week, output_dir)
