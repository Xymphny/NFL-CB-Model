"""
Real injury impact -- closes the loop between ingest/injuries.py's real
weekly report data and the already-validated QB VAR system
(model/injuries_and_var.py). If a team's actual current starter is
officially ruled Out for the upcoming week, this replaces them with
replacement level in the team's offensive rating for that
prediction -- a real, live version of the same "persistent QB
adjustment" concept already built for slower-moving starter changes,
now driven by the market's own real-time information instead of
inferred from snap counts after the fact.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingest.injuries import get_players_out
from model.injuries_and_var import compute_qb_var, identify_current_starter


def apply_injury_adjustment(ratings, df, season, upcoming_week):
    qb_var = compute_qb_var(df)
    adjusted = ratings.copy()
    adjusted["injury_adjustment"] = 0.0
    adjusted["injury_note"] = None

    try:
        out_players = get_players_out(season, upcoming_week)
    except Exception as e:
        print(f"[injury_impact] could not load injury report for {season} week {upcoming_week}: {e}")
        return adjusted

    for team in adjusted.index:
        starter = identify_current_starter(df, team)
        if starter is None:
            continue

        team_out = out_players[out_players["team"] == team]
        for _, row in team_out.iterrows():
            full_name = row["full_name"]
            last_name = full_name.split()[-1]
            first_initial = full_name[0]
            short_form = f"{first_initial}.{last_name}"

            if short_form == starter and row["position"] == "QB":
                if starter in qb_var.index:
                    replacement_var = 0.0
                    starter_var = qb_var.loc[starter, "var"]
                    adjusted.loc[team, "injury_adjustment"] = replacement_var - starter_var
                    adjusted.loc[team, "injury_note"] = f"{full_name} (Out, {row['report_primary_injury']}) -- reverting to replacement level"

    adjusted["offense_voa_injury_adjusted"] = adjusted["offense_voa"] + adjusted["injury_adjustment"]
    adjusted["total_rating_injury_adjusted"] = adjusted["offense_voa_injury_adjusted"] - adjusted["defense_voa"]
    return adjusted


if __name__ == "__main__":
    from ingest.nfl_pbp import load_season
    from model.ratings import (
        add_situation_buckets, score_all_plays, compute_baselines,
        compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
    )

    print("Testing against the real 2023 Deshaun Watson case (CLE, ruled Out week 8)...")
    df = load_season(2023)
    df_through_w7 = df[df["week"] <= 7].copy()
    df_through_w7 = add_situation_buckets(df_through_w7)
    df_through_w7 = score_all_plays(df_through_w7)
    df_through_w7 = filter_garbage_time(df_through_w7)
    baselines = compute_baselines(df_through_w7)
    df_through_w7 = compute_raw_voa(df_through_w7, baselines)
    df_through_w7 = opponent_adjust(df_through_w7)
    ratings = team_ratings(df_through_w7, use_recency_weights=False)

    adjusted = apply_injury_adjustment(ratings, df_through_w7, season=2023, upcoming_week=8)
    print(f"CLE injury_adjustment: {adjusted.loc['CLE', 'injury_adjustment']:.4f}")
    print(f"CLE injury_note: {adjusted.loc['CLE', 'injury_note']}")
    print(f"CLE offense_voa: {adjusted.loc['CLE', 'offense_voa']:.4f} -> adjusted: {adjusted.loc['CLE', 'offense_voa_injury_adjusted']:.4f}")
