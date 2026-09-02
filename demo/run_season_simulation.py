"""
Demo: simulate the rest of the 2023 season from a real week-10 cutoff,
using actual results for weeks 1-10 and simulating weeks 11-18.

This lets us sanity-check the simulation against what ACTUALLY happened
in the second half of 2023 — a real, if informal, validation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, team_ratings, filter_garbage_time,
)
from model.market_comparison import bootstrap_rating_uncertainty
from model.season_simulation import simulate_season
from model.injuries_and_var import apply_qb_persistence_adjustment

CUTOFF_WEEK = 10


def main():
    print(f"Loading 2023 data, using weeks 1-{CUTOFF_WEEK} to rate teams...")
    pbp = load_season(2023)
    pbp_through_cutoff = pbp[pbp["week"] <= CUTOFF_WEEK].copy()

    pbp_through_cutoff = add_situation_buckets(pbp_through_cutoff)
    pbp_through_cutoff = score_all_plays(pbp_through_cutoff, use_turnover_luck_adjustment=True)
    pbp_through_cutoff = filter_garbage_time(pbp_through_cutoff)  # was missing here — inconsistent with the rest of the pipeline
    baselines = compute_baselines(pbp_through_cutoff)
    pbp_through_cutoff = compute_raw_voa(pbp_through_cutoff, baselines)
    pbp_through_cutoff = opponent_adjust(pbp_through_cutoff, iterations=3, regression=0.5)
    pbp_through_cutoff = pbp_through_cutoff.drop(columns=["recency_weight"], errors="ignore")

    ratings = team_ratings(pbp_through_cutoff, use_recency_weights=False)
    print(f"  Rated {len(ratings)} teams through week {CUTOFF_WEEK}")

    print("Applying persistent QB adjustment (Section 11.3, now wired to real per-play data)...")
    qb_adjusted = apply_qb_persistence_adjustment(ratings, pbp_through_cutoff)
    print(f"  MIN current starter detected: {qb_adjusted.loc['MIN', 'current_starter']} "
          f"(qb_adjustment: {qb_adjusted.loc['MIN', 'qb_adjustment']:+.3f})")
    ratings_qb_adjusted = qb_adjusted[["offense_voa", "defense_voa", "total_rating"]].copy()
    ratings_qb_adjusted["offense_voa"] = qb_adjusted["offense_voa_qb_adjusted"]
    ratings_qb_adjusted["total_rating"] = qb_adjusted["total_rating_qb_adjusted"]

    print("Running bootstrap uncertainty (30 iterations)...")
    def rating_fn(resample):
        return team_ratings(resample, use_recency_weights=False)
    uncertainty = bootstrap_rating_uncertainty(pbp_through_cutoff, rating_fn, n_bootstrap=30)

    sched = load_schedules(seasons=[2023])
    played = sched[sched["week"] <= CUTOFF_WEEK]
    remaining = sched[sched["week"] > CUTOFF_WEEK].copy()
    remaining["rest_diff"] = remaining["home_rest"] - remaining["away_rest"]
    print(f"  {len(played)} games played, {len(remaining)} games remaining to simulate")

    # Current actual records through the cutoff.
    current_records = {}
    for _, g in played.iterrows():
        if pd.isna(g["home_score"]):
            continue
        home_win = g["home_score"] > g["away_score"]
        current_records[g["home_team"]] = current_records.get(g["home_team"], 0) + (1 if home_win else 0)
        current_records[g["away_team"]] = current_records.get(g["away_team"], 0) + (0 if home_win else 1)
    for team in ratings.index:
        current_records.setdefault(team, 0)

    # Margin coefficients — using the calibrated values from
    # calibrate_points_model.py's actual output (Section 11.4).
    margin_coefficients = {
        "rating_diff": 22.7091,
        "home_field": 2.8321,
        "rest_diff": 0.1310,
        "intercept": -1.1811,
    }

    print("Simulating remaining season, WITHOUT QB adjustment (2000 iterations)...")
    results_no_qb = simulate_season(
        remaining_schedule=remaining,
        current_records=current_records,
        ratings=ratings,
        rating_uncertainty=uncertainty,
        margin_coefficients=margin_coefficients,
        n_simulations=2000,
    )

    print("Simulating remaining season, WITH QB adjustment (2000 iterations)...")
    results_qb = simulate_season(
        remaining_schedule=remaining,
        current_records=current_records,
        ratings=ratings_qb_adjusted,
        rating_uncertainty=uncertainty,
        margin_coefficients=margin_coefficients,
        n_simulations=2000,
    )

    print("\n=== MIN specifically: without vs. with QB adjustment, vs. actual ===")
    final_actual = {}
    for _, g in sched.iterrows():
        if pd.isna(g["home_score"]):
            continue
        home_win = g["home_score"] > g["away_score"]
        final_actual[g["home_team"]] = final_actual.get(g["home_team"], 0) + (1 if home_win else 0)
        final_actual[g["away_team"]] = final_actual.get(g["away_team"], 0) + (0 if home_win else 1)

    print(f"  Without QB adjustment: {results_no_qb.loc['MIN', 'mean_wins']:.1f} projected wins")
    print(f"  With QB adjustment:    {results_qb.loc['MIN', 'mean_wins']:.1f} projected wins")
    print(f"  Actual final record:   {final_actual['MIN']} wins")

    results = results_qb  # use QB-adjusted for the rest of the summary below

    print("\n=== Simulated final win totals (top 10) ===")
    print(results.head(10).to_string(float_format=lambda x: f"{x:.1f}"))

    print("\n=== Sanity check: actual 2023 final records for these teams ===")
    final_actual = {}
    for _, g in sched.iterrows():
        if pd.isna(g["home_score"]):
            continue
        home_win = g["home_score"] > g["away_score"]
        final_actual[g["home_team"]] = final_actual.get(g["home_team"], 0) + (1 if home_win else 0)
        final_actual[g["away_team"]] = final_actual.get(g["away_team"], 0) + (0 if home_win else 1)
    actual_series = pd.Series(final_actual, name="actual_final_wins")
    comparison = results.head(10)[["mean_wins"]].join(actual_series)
    print(comparison.to_string(float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
