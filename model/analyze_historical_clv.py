"""
Historical CLV analysis -- the math (model/clv_tracking.py) was built
and tested with synthetic data, but there was no tool to actually
analyze REAL accumulated data once it exists across a full season.
This scans a data directory for every real week present, runs the
existing per-week CLV report for each, and produces one aggregate
summary -- validation rate, broken down by whether the divergence was
originally flagged as significant or not, since that's the real,
useful question ("does flagging actually mean something") rather than
just an overall number.
"""

import sys
import os
import re
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.clv_tracking import compute_clv_report, load_divergence_snapshots


def discover_available_weeks(data_dir, season):
    pattern = os.path.join(data_dir, "divergence", f"{season}-week-*-*.json")
    files = glob.glob(pattern)
    weeks = set()
    for f in files:
        match = re.search(rf"{season}-week-(\d+)-", os.path.basename(f))
        if match:
            weeks.add(int(match.group(1)))
    return sorted(weeks)


def get_flag_status(data_dir, season, week, home_team, away_team):
    snapshots = load_divergence_snapshots(data_dir, season, week)
    if not snapshots:
        return None
    for d in snapshots[0].get("divergences", []):
        if d["home_team"] == home_team and d["away_team"] == away_team:
            return d.get("spread_flagged", False)
    return None


def analyze_season(data_dir, season):
    weeks = discover_available_weeks(data_dir, season)
    if not weeks:
        return {"weeks_analyzed": [], "games": [], "summary": None}

    all_games = []
    for week in weeks:
        week_results = compute_clv_report(data_dir, season, week)
        for result in week_results:
            result["week"] = week
            result["was_flagged"] = get_flag_status(
                data_dir, season, week, result["home_team"], result["away_team"]
            )
            all_games.append(result)

    if not all_games:
        return {"weeks_analyzed": weeks, "games": [], "summary": None}

    validated_count = sum(1 for g in all_games if g["validated"])
    flagged_games = [g for g in all_games if g["was_flagged"]]
    unflagged_games = [g for g in all_games if g["was_flagged"] is False]

    summary = {
        "total_games": len(all_games),
        "overall_validation_rate": validated_count / len(all_games),
        "avg_clv_score": sum(g["clv_score"] for g in all_games) / len(all_games),
        "flagged_games_count": len(flagged_games),
        "flagged_validation_rate": (
            sum(1 for g in flagged_games if g["validated"]) / len(flagged_games)
            if flagged_games else None
        ),
        "unflagged_games_count": len(unflagged_games),
        "unflagged_validation_rate": (
            sum(1 for g in unflagged_games if g["validated"]) / len(unflagged_games)
            if unflagged_games else None
        ),
    }

    return {"weeks_analyzed": weeks, "games": all_games, "summary": summary}


def print_report(result):
    if not result["games"]:
        print("No real CLV data available yet -- needs at least two odds checks on the same "
              "game, across at least one real week, before this can report anything.")
        return

    s = result["summary"]
    print(f"Weeks analyzed: {result['weeks_analyzed']}")
    print(f"Total games with real line movement: {s['total_games']}")
    print(f"Overall validation rate (market moved toward model): {s['overall_validation_rate']:.1%}")
    print(f"Average CLV score: {s['avg_clv_score']:+.2f}")
    print()
    if s['flagged_games_count']:
        print(f"Games we flagged as diverging: {s['flagged_games_count']}, "
              f"validation rate: {s['flagged_validation_rate']:.1%}")
    else:
        print("Games we flagged as diverging: 0")
    if s['unflagged_games_count']:
        print(f"Games we did NOT flag: {s['unflagged_games_count']}, "
              f"validation rate: {s['unflagged_validation_rate']:.1%}")
    else:
        print("Games we did NOT flag: 0")


if __name__ == "__main__":
    import json
    import shutil

    print("Testing against constructed multi-week data (real accumulated data doesn't exist yet)...\n")
    test_dir = "/tmp/clv_analysis_test"
    os.makedirs(f"{test_dir}/divergence", exist_ok=True)

    week1_snap1 = {
        "computed_at": "2026-09-10T12:00:00Z",
        "divergences": [
            {"home_team": "DET", "away_team": "MIN", "market_spread": 7.5, "spread_gap": 2.5, "spread_flagged": True},
            {"home_team": "KC", "away_team": "LAC", "market_spread": 3.0, "spread_gap": 0.3, "spread_flagged": False},
        ],
    }
    week1_snap2 = {
        "computed_at": "2026-09-10T18:00:00Z",
        "divergences": [
            {"home_team": "DET", "away_team": "MIN", "market_spread": 9.0, "spread_gap": 1.0, "spread_flagged": False},
            {"home_team": "KC", "away_team": "LAC", "market_spread": 3.0, "spread_gap": 0.3, "spread_flagged": False},
        ],
    }
    with open(f"{test_dir}/divergence/2026-week-01-20260910T120000Z.json", "w") as f:
        json.dump(week1_snap1, f)
    with open(f"{test_dir}/divergence/2026-week-01-20260910T180000Z.json", "w") as f:
        json.dump(week1_snap2, f)

    result = analyze_season(test_dir, 2026)
    print_report(result)

    shutil.rmtree(test_dir)
