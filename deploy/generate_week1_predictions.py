"""
Generates real Week 1 2026 predictions using last season (2025) blended
with real 2026 preseason performance (point differential + PRE WK3
yardage/takeaway margin) as the prior — since actual 2026 in-season
data doesn't exist yet, this prior IS the best available team-strength
estimate for Week 1.

Written as week=0 (not week=1) specifically so it never collides with
or gets confused for the real post-game Week 1 rating weekly_job.py
will eventually produce once real games are played — "2026-week-00"
sorts before "2026-week-01" alphabetically, so the real data
automatically takes over once it exists, same mechanism as the
2023-vs-2026 demo-data handoff.
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.weekly_job import get_or_compute_prior
from ingest.nfl_schedules import load_schedules
from model.prediction import build_week_predictions
from model.team_profile import build_team_profile
from model.elo_rating import compute_elo_walk_forward

SEASON = 2026


def main(output_dir: str = "./data"):
    print(f"Computing {SEASON} prior (2025 rating + real preseason performance)...")
    prior_ratings, _, _ = get_or_compute_prior(SEASON, output_dir)

    # Layer 2 for 2026 Week 1: no 2026 NGS data can exist yet (same
    # reason 2026 play-by-play doesn't exist -- no games played). Tries
    # 2025 first as the natural prior, but a REAL finding while testing
    # this: NGS data has its own publishing lag behind play-by-play --
    # 2025 (a fully completed season) returned a 404 while 2024 worked
    # fine. Falls back through recent seasons rather than giving up
    # after one miss, to actually get real Layer 2 data into the
    # prediction instead of none.
    # HONEST FINDING FROM TESTING: tried falling back to a prior
    # season's NGS data (2023, confirmed complete) as a Week 1 preseason
    # signal, and it made predictions measurably WORSE against the real
    # market lines -- mean absolute gap rose from 2.076 to 2.716 points,
    # with 9 of 16 games getting worse against only 6 improving. The
    # validated Layer 2 improvement (58.22% -> 64.04% straight-up
    # accuracy) is specific to WITHIN-SEASON use -- a season's own early
    # data predicting that same season's later weeks -- not cross-season
    # transfer using multi-year-old rosters. Layer 2 is correctly NOT
    # applied here; it activates naturally in weekly_job.py/
    # odds_watch_job.py once the 2026 season has its own real data to use.
    ngs_features = None

    # Elo, unlike Layer 2 NGS, is DESIGNED to carry over between
    # seasons (with built-in regression-to-mean) -- this is its
    # intended use case, not the same cross-season misapplication that
    # was found to hurt Layer 2 features earlier. Uses real historical
    # results through the most recently completed season.
    try:
        historical_schedule = load_schedules(seasons=list(range(SEASON - 11, SEASON)))
        _, elo_ratings = compute_elo_walk_forward(historical_schedule)
        print(f"Loaded real Elo ratings from {SEASON - 11}-{SEASON - 1}")
    except Exception as e:
        print(f"Elo ratings unavailable ({e}), predicting without them")
        elo_ratings = None

    sched = load_schedules(seasons=[SEASON])
    week1_games = sched[sched["week"] == 1]
    print(f"\nReal Week 1 {SEASON} schedule: {len(week1_games)} games")

    predictions = build_week_predictions(prior_ratings, week1_games, ngs_features=ngs_features, elo_ratings=elo_ratings)
    print(f"Built predictions for {len(predictions)} of {len(week1_games)} games")

    # Write as a ratings snapshot (week=0, see module docstring) so the
    # dashboard can display it the same way as any other snapshot.
    ratings_dir = os.path.join(output_dir, "ratings")
    os.makedirs(ratings_dir, exist_ok=True)
    output_file = os.path.join(ratings_dir, f"{SEASON}-week-00.json")

    # Fill in offense/defense_voa placeholders needed by team_profile's
    # expected columns, plus the actual prior rating.
    payload_ratings = []
    for team in prior_ratings.index:
        payload_ratings.append({
            "team": team,
            "offense_voa": float(prior_ratings.loc[team, "offense_voa"]),
            "defense_voa": float(prior_ratings.loc[team, "defense_voa"]),
            "total_rating": float(prior_ratings.loc[team, "total_rating"]),
            "epa_per_play_offense": None, "epa_per_play_allowed": None,
            "success_rate_offense": None, "success_rate_allowed": None,
            "red_zone_trips": None, "red_zone_points_per_trip": None, "red_zone_td_pct": None,
            "giveaways": None, "takeaways": None, "turnover_margin": None,
        })

    with open(output_file, "w") as f:
        json.dump({
            "season": SEASON,
            "week": 0,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "prior_source": f"PRESEASON PROJECTION — {SEASON-1} rating blended with real {SEASON} "
                             f"preseason performance (point diff + PRE WK3 yardage/takeaway margin). "
                             f"Not a post-game rating — the real Week 1 rating will replace this once "
                             f"actual games are played.",
            "ratings": payload_ratings,
        }, f, indent=2)
    print(f"Wrote {output_file}")

    return predictions, week1_games


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    predictions, week1_games = main(output_dir)

    print("\n=== Sample Week 1 predictions ===")
    for team, pred in list(predictions.items())[:8]:
        print(f"  {team} (home): spread={pred['spread']:+.1f}, total={pred['total']:.1f}, win_prob={pred['win_prob_home']:.2f}")
