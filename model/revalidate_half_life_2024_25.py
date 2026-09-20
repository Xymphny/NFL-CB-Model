"""Does the half-life result replicate on seasons nobody looked at?

WHY THIS EXISTS. model/ratings.py ships half_life_weeks=100 and its
docstring calls the choice "real, held-out walk-forward calibration."
Read the rest of that docstring and it says something different: the
train-set argmin picked a SHORT half-life, that choice was rejected
because it "actively picked the WORST real-world choice," and 100 was
taken because it scored best on the 2023 test set (59.13% straight-up
vs 56.25% for the old default of 6).

That is selection on the test set. The script's own argmin is honest
-- min() over train MAE -- but the operator saw every candidate's test
column printed beside it and overrode the argmin using that column. So
the 2023 number is not a clean held-out estimate of anything, and the
docstring presents it as one.

THE DEFENSE IS THE GOOD ONE, WHICH IS WHY IT IS TESTABLE. The
docstring does not rest on 100 winning; it rests on test MAE improving
MONOTONICALLY across all 8 candidates, which is far harder to get by
chance than a single lucky argmin. A monotone trend across a grid is a
real claim about football: that week-to-week form carries little
signal beyond season-long quality. Real claims replicate.

WHAT THIS RUNS. Every calibrate_* script in this repo touches only
2021-2023 (Elo: 2014-2023), so 2024-25 are clean for a margin
question. They are NOT virgin ground in general, and an earlier draft
of this file wrongly said they were: model/player_projection.py holds
them out as its gate set and has already made ship/withhold decisions
there, pass_yds twice. Different quantity -- player prop shapes, not
team margin error -- but the same seasons, and the engine's team-TD
environment is derived from these same ratings, so the two are not
strictly independent. Recorded because the size of the remaining
holdout budget is exactly the kind of thing that gets rounded up.

This replays the same walk-forward over those two seasons, over the
same 8-candidate grid, and asks three questions:

  1. Does 100 still beat 6?
  2. Is the trend still monotone in half-life?
  3. Is the gap bigger than its own standard error?

WHAT IT WILL NOT DO. It will not re-tune. If 100 loses here, the
honest response is a ledger entry saying the knob did not replicate --
not a fresh argmin on 2024-25, which would restart the identical leak
one season later. Running this at all spends 2024-25 as a selection
set; that is the price of getting one clean read, and it is stated
here so the next person knows the well is now dry.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules
from model.ratings import (add_home_field_and_rest, add_recency_weights,
                           add_situation_buckets, compute_baselines,
                           compute_raw_voa, filter_garbage_time,
                           opponent_adjust, score_all_plays, team_ratings)
from model.prediction import predict_margin

HOLDOUT_SEASONS = [2024, 2025]      # clean for margin work; see docstring
BACKTEST_WEEKS = range(4, 18)
CANDIDATE_HALF_LIVES = [2, 4, 6, 8, 10, 12, 16, 100]
SHIPPED = 100.0
PRIOR_DEFAULT = 6.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "revalidate_half_life_2024_25_results.json")


def build_cached_dataframes():
    """The half-life-independent part, computed once per (season, week)."""
    cache = {}
    for season in HOLDOUT_SEASONS:
        schedules = load_schedules(seasons=[season])
        full_season_df = load_season(season)
        for week in BACKTEST_WEEKS:
            df = full_season_df[full_season_df["week"] < week].copy()
            if df.empty:
                continue
            df = add_situation_buckets(df)
            df = score_all_plays(df, use_turnover_luck_adjustment=True)
            df = filter_garbage_time(df)
            df = add_home_field_and_rest(df, schedules)
            baselines = compute_baselines(df)
            df = compute_raw_voa(df, baselines)
            df = opponent_adjust(df, iterations=3, regression=0.5)
            cache[(season, week)] = (df, schedules)
        print(f"  cached season {season}", flush=True)
    return cache


def run_walk_forward(half_life, cache):
    rows = []
    for (season, week), (df, schedules) in cache.items():
        d = add_recency_weights(df.copy(), half_life_weeks=half_life)
        ratings = team_ratings(d, use_recency_weights=True)
        week_games = schedules[schedules["week"] == week].dropna(
            subset=["home_score", "away_score"])
        for _, game in week_games.iterrows():
            home, away = game["home_team"], game["away_team"]
            if home not in ratings.index or away not in ratings.index:
                continue
            rating_diff = (ratings.loc[home, "total_rating"]
                           - ratings.loc[away, "total_rating"])
            pred = predict_margin(rating_diff, game.get("is_neutral_site", False),
                                  game["home_rest"] - game["away_rest"])
            actual = game["home_score"] - game["away_score"]
            rows.append({"season": season, "week": week, "pred_margin": pred,
                         "actual_margin": actual, "actual_home_win": actual > 0,
                         "key": f"{season}-{week}-{away}@{home}"})
    return pd.DataFrame(rows)


def main():
    print("Building cached dataframes for the holdout seasons...", flush=True)
    cache = build_cached_dataframes()

    results, preds = {}, {}
    for hl in CANDIDATE_HALF_LIVES:
        df = run_walk_forward(hl, cache)
        mae = float(np.mean(np.abs(df["pred_margin"] - df["actual_margin"])))
        acc = float(((df["pred_margin"] > 0) == df["actual_home_win"]).mean())
        results[hl] = {"mae": mae, "straight_up": acc, "n": int(len(df))}
        preds[hl] = df.set_index("key")
        print(f"  half_life={hl:<4} MAE={mae:.3f}  straight-up={acc:.4f}  n={len(df)}",
              flush=True)

    # Q1: does the shipped value still beat the one it replaced?
    ship, prior = results[SHIPPED], results[PRIOR_DEFAULT]
    beats = ship["straight_up"] > prior["straight_up"]

    # Q3: paired standard error on the accuracy gap. The two configs are
    # graded on the SAME games, so pair them rather than treating the
    # seasons as two independent samples.
    a = preds[SHIPPED]
    b = preds[PRIOR_DEFAULT].reindex(a.index)
    hit_a = ((a["pred_margin"] > 0) == a["actual_home_win"]).astype(float)
    hit_b = ((b["pred_margin"] > 0) == b["actual_home_win"]).astype(float)
    d = (hit_a - hit_b).dropna()
    gap = float(d.mean())
    gap_se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")

    # Q2: monotone in half-life, the claim the docstring actually rests on?
    accs = [results[hl]["straight_up"] for hl in CANDIDATE_HALF_LIVES]
    maes = [results[hl]["mae"] for hl in CANDIDATE_HALF_LIVES]
    mono_acc = all(x <= y for x, y in zip(accs, accs[1:]))
    mono_mae = all(x >= y for x, y in zip(maes, maes[1:]))

    # Reported, NOT acted on. See the module docstring.
    argmin_here = min(results, key=lambda hl: results[hl]["mae"])

    print("\n--- the three questions ---")
    print(f"1. 100 still beats 6:            {beats}   "
          f"({ship['straight_up']:.4f} vs {prior['straight_up']:.4f})")
    print(f"2. monotone in half-life:        accuracy {mono_acc}, MAE {mono_mae}")
    print(f"3. gap vs its own error:         {gap:+.4f} +/- {gap_se:.4f} "
          f"(paired, n={len(d)})")
    print(f"\nargmin on these seasons would be {argmin_here} -- REPORTED, NOT ADOPTED. "
          f"The knob stays at {SHIPPED}.")

    out = {
        "_provenance": {
            "script": "model/revalidate_half_life_2024_25.py",
            "generated": __import__("datetime").date.today().isoformat(),
            "holdout_seasons": HOLDOUT_SEASONS,
            "why_these_seasons": ("Every calibrate_* script trains and tests only on "
                                  "2021-2023 (Elo 2014-2023), so these two are clean for a "
                                  "MARGIN question. They are not unused in general: "
                                  "model/player_projection.py gates its markets here and has "
                                  "withheld pass_yds on them twice. Different quantity, same "
                                  "seasons, and not strictly independent -- the engine's "
                                  "team-TD environment comes from these ratings."),
            "what_is_being_tested": ("half_life=100 was selected on the 2023 TEST set, "
                                     "overriding a train argmin that favored short "
                                     "half-lives. model/ratings.py describes that as "
                                     "held-out calibration. This grades it on data that "
                                     "actually was."),
            "selection_spent": ("Publishing this burns 2024-25 as a future selection set. "
                                "Stated so the next person knows."),
        },
        "shipped_value": SHIPPED,
        "prior_default": PRIOR_DEFAULT,
        "per_candidate": {str(hl): results[hl] for hl in CANDIDATE_HALF_LIVES},
        "questions": {
            "shipped_beats_prior_default": bool(beats),
            "monotone_in_half_life_accuracy": bool(mono_acc),
            "monotone_in_half_life_mae": bool(mono_mae),
            "paired_accuracy_gap": round(gap, 5),
            "paired_gap_standard_error": round(gap_se, 5),
            "gap_exceeds_one_se": bool(abs(gap) > gap_se) if gap_se == gap_se else None,
            "argmin_on_holdout_reported_not_adopted": argmin_here,
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
