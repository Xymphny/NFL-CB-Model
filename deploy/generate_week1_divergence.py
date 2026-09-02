"""
Compares Week 1 2026 predictions (from the preseason-informed prior)
against real, currently-posted book lines gathered from ESPN.

Everything here is real: real 2025 season rating, real 2026 preseason
performance (point diff + PRE WK3 yardage/takeaway margin), real Week
1 schedule, real current market lines. The comparison itself follows
Section 9.3's design — model and market stay separate, only the gap
is surfaced.
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.weekly_job import get_or_compute_prior
from ingest.nfl_schedules import load_schedules
from model.prediction import build_week_predictions
from model.market_comparison import american_to_implied_prob, devig_two_way, flag_divergence
from model.week1_2026_lines import WEEK1_2026_CURRENT_LINES, GATHERED_AT

SEASON = 2026


def main(output_dir: str = "./data"):
    print("Computing preseason-informed prior...")
    prior_ratings, _, _ = get_or_compute_prior(SEASON, output_dir)

    # Same fallback pattern as generate_week1_predictions.py — real
    # finding: NGS data lags play-by-play by about a season, so 2025
    # isn't available yet even though it's complete; falls through to
    # 2024 rather than predicting without Layer 2 entirely.
    # Layer 2 deliberately NOT applied here -- see generate_week1_predictions.py's
    # comment for the real, tested finding: cross-season NGS transfer
    # (old rosters as a preseason prior) measurably hurts accuracy
    # rather than helping. Layer 2 activates correctly in
    # weekly_job.py/odds_watch_job.py once 2026 has its own real data.
    ngs_features = None

    sched = load_schedules(seasons=[SEASON])
    week1_games = sched[sched["week"] == 1]
    predictions = build_week_predictions(prior_ratings, week1_games, ngs_features=ngs_features)

    divergences = []
    for away, home, away_ml, home_ml, home_spread, total in WEEK1_2026_CURRENT_LINES:
        if home not in predictions:
            print(f"  no prediction for {away}@{home}, skipping")
            continue

        pred = predictions[home]

        home_raw = american_to_implied_prob(home_ml)
        away_raw = american_to_implied_prob(away_ml)
        home_fair, away_fair = devig_two_way(home_raw, away_raw)

        market_spread = -home_spread

        divergence = flag_divergence(
            model_spread=pred["spread"], model_total=pred["total"], model_win_prob_home=pred["win_prob_home"],
            market_spread=market_spread, market_total=total,
            market_odds_home=home_fair, market_odds_away=away_fair,
        )

        divergences.append({
            "away_team": away, "home_team": home,
            "market_win_prob_home_fair": home_fair,
            "market_spread": market_spread,
            "market_total": total,
            **divergence,
        })

    divergence_dir = os.path.join(output_dir, "divergence")
    os.makedirs(divergence_dir, exist_ok=True)
    output_file = os.path.join(divergence_dir, f"{SEASON}-week-00-preseason-projection.json")

    with open(output_file, "w") as f:
        json.dump({
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "season": SEASON,
            "week": 0,
            "note": f"PRESEASON PROJECTION vs. real current lines (gathered {GATHERED_AT}). "
                    f"Model side uses 2025 rating + real preseason performance, not in-season data "
                    f"(none exists yet). Lines will move before kickoff -- this is a snapshot.",
            "divergences": divergences,
        }, f, indent=2)

    print(f"\nWrote {output_file}")
    return divergences


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    divergences = main(output_dir)

    print(f"\n=== Week 1 2026: Preseason-Informed Model vs. Real Current Lines ===\n")
    flagged = [d for d in divergences if d["spread_flagged"] or d["total_flagged"] or d["win_prob_flagged"]]
    print(f"{len(divergences)} games compared, {len(flagged)} flagged as diverging\n")

    for d in sorted(divergences, key=lambda x: -abs(x["spread_gap"])):
        model_spread = d["market_spread"] + d["spread_gap"]
        flag = " <-- DIVERGES" if (d["spread_flagged"] or d["total_flagged"] or d["win_prob_flagged"]) else ""
        print(f"  {d['away_team']:>3} @ {d['home_team']:<3}  "
              f"market {d['market_spread']:+.1f} / model {model_spread:+.1f}  "
              f"(gap {d['spread_gap']:+.1f}){flag}")
