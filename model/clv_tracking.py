"""
Closing-line value (CLV) tracking -- Section 9's original design
reason for keeping every odds-watch snapshot rather than overwriting.
The infrastructure has existed since the snapshot-architecture fix,
but nothing ever actually computed CLV from it until now.

CLV answers: when the model diverged from an early market line, did
the market later move toward the model's view (validating the
model's signal) or away from it (suggesting the divergence was noise,
not edge)? This is the standard way sharp bettors validate a model
without waiting a full season for win/loss sample size to become
meaningful.

NOTE ON TESTING: real accumulated multi-snapshot data doesn't exist
yet -- the season hasn't started, so odds_watch_job.py has never run
enough times against real games to produce the kind of line-movement
history this module is meant to analyze. Tested here against
constructed-but-realistic snapshot scenarios rather than real
accumulated data -- that real validation has to wait for the season
to actually run.
"""

import sys
import os
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_divergence_snapshots(data_dir, season, week):
    pattern = os.path.join(data_dir, "divergence", f"{season}-week-{week:02d}-*.json")
    files = sorted(glob.glob(pattern))

    snapshots = []
    for f in files:
        with open(f) as fh:
            snapshots.append(json.load(fh))
    return snapshots


def compute_clv_for_game(home_team, away_team, snapshots):
    game_appearances = []
    for snap in snapshots:
        for d in snap.get("divergences", []):
            if d["home_team"] == home_team and d["away_team"] == away_team:
                game_appearances.append({"computed_at": snap["computed_at"], **d})

    if len(game_appearances) < 2:
        return None

    earliest, latest = game_appearances[0], game_appearances[-1]

    model_spread = earliest["market_spread"] + earliest["spread_gap"]
    opening_market_spread = earliest["market_spread"]
    closing_market_spread = latest["market_spread"]

    divergence_direction = model_spread - opening_market_spread
    market_movement = closing_market_spread - opening_market_spread

    if divergence_direction == 0:
        clv_score = 0.0
    else:
        clv_score = market_movement * (1 if divergence_direction > 0 else -1)

    return {
        "home_team": home_team, "away_team": away_team,
        "n_snapshots": len(game_appearances),
        "opening_market_spread": opening_market_spread,
        "closing_market_spread": closing_market_spread,
        "model_spread": model_spread,
        "market_movement": market_movement,
        "clv_score": clv_score,
        "validated": clv_score > 0,
    }


def compute_clv_report(data_dir, season, week):
    snapshots = load_divergence_snapshots(data_dir, season, week)
    if len(snapshots) < 2:
        return []

    all_games = set()
    for snap in snapshots:
        for d in snap.get("divergences", []):
            all_games.add((d["home_team"], d["away_team"]))

    results = []
    for home, away in all_games:
        clv = compute_clv_for_game(home, away, snapshots)
        if clv:
            results.append(clv)
    return results


if __name__ == "__main__":
    os.makedirs("/tmp/clv_test/divergence", exist_ok=True)

    snapshot_1 = {
        "computed_at": "2026-09-10T12:00:00Z",
        "divergences": [
            {"home_team": "DET", "away_team": "MIN", "market_spread": 7.5, "market_total": 46.5,
             "spread_gap": 2.5, "total_gap": 1.0, "win_prob_gap": 0.05,
             "spread_flagged": True, "total_flagged": False, "win_prob_flagged": False},
        ],
    }
    snapshot_2 = {
        "computed_at": "2026-09-10T18:00:00Z",
        "divergences": [
            {"home_team": "DET", "away_team": "MIN", "market_spread": 9.0, "market_total": 46.0,
             "spread_gap": 1.0, "total_gap": 1.5, "win_prob_gap": 0.03,
             "spread_flagged": False, "total_flagged": False, "win_prob_flagged": False},
        ],
    }
    with open("/tmp/clv_test/divergence/2026-week-01-20260910T120000Z.json", "w") as f:
        json.dump(snapshot_1, f)
    with open("/tmp/clv_test/divergence/2026-week-01-20260910T180000Z.json", "w") as f:
        json.dump(snapshot_2, f)

    report = compute_clv_report("/tmp/clv_test", 2026, 1)
    print("Test scenario: model favored DET more than the opening line (model_spread=10.0,")
    print("opening market=7.5), and the closing line moved to 9.0 -- toward the model's view.")
    print()
    for game in report:
        print(game)
        assert game["validated"], "Expected this constructed scenario to show positive CLV"
    print("\nPASS: CLV mechanism correctly identifies market movement toward the model's view")

    import shutil
    shutil.rmtree("/tmp/clv_test")
