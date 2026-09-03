"""
Properly co-calibrates the full production feature set -- DVOA
rating_diff, home_field, rest_diff, Layer 2 NGS features, AND the new
Elo ensemble term -- together in one regression, on the largest range
where all of them are real and available (2016-2023, since NGS data
starts in 2016). Avoids the inconsistency of mixing coefficients fit
on different data ranges at different times.

Train: 2016-2021 (6 seasons). Test: 2022-2023 (2 seasons, fully held
out).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingest.nfl_schedules import load_schedules
from model.elo_rating import compute_elo_walk_forward
from model.layer2_ngs import compute_team_ngs_features, load_ngs_data

CACHE_PATH = os.path.join(os.path.dirname(__file__), "expanded_walk_forward_cache.csv")
TRAIN_SEASONS = list(range(2016, 2022))
TEST_SEASONS = [2022, 2023]
ALL_SEASONS = TRAIN_SEASONS + TEST_SEASONS


def build_combined_dataset():
    dvoa = pd.read_csv(CACHE_PATH)
    dvoa = dvoa[dvoa["season"].isin(ALL_SEASONS)].copy()

    schedule = load_schedules(seasons=list(range(2014, 2024)))
    elo_df, _ = compute_elo_walk_forward(schedule)

    combined = dvoa.merge(
        elo_df[["season", "week", "home_team", "away_team", "elo_diff"]],
        on=["season", "week", "home_team", "away_team"], how="inner",
    )

    # Add Layer 2 NGS features, walk-forward (through the same week
    # cutoff as the DVOA rating), matching the original validated
    # Layer 2 methodology exactly.
    ngs_rows = []
    for season in ALL_SEASONS:
        print(f"  loading NGS for {season}...")
        ngs_preloaded = {
            "passing": load_ngs_data(season, "passing"),
            "receiving": load_ngs_data(season, "receiving"),
            "rushing": load_ngs_data(season, "rushing"),
        }
        season_games = combined[combined["season"] == season]
        for week in season_games["week"].unique():
            try:
                ngs = compute_team_ngs_features(season, through_week=week, preloaded_data=ngs_preloaded)
            except ValueError:
                continue
            week_games = season_games[season_games["week"] == week]
            for _, game in week_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in ngs.index or away not in ngs.index:
                    continue
                ngs_rows.append({
                    "season": season, "week": week, "home_team": home, "away_team": away,
                    "cpoe_diff": ngs.loc[home, "team_cpoe"] - ngs.loc[away, "team_cpoe"],
                    "separation_diff": ngs.loc[home, "team_avg_separation"] - ngs.loc[away, "team_avg_separation"],
                    "yac_oe_diff": ngs.loc[home, "team_yac_over_expected"] - ngs.loc[away, "team_yac_over_expected"],
                    "ryoe_diff": ngs.loc[home, "team_ryoe"] - ngs.loc[away, "team_ryoe"],
                })

    ngs_df = pd.DataFrame(ngs_rows)
    combined = combined.merge(ngs_df, on=["season", "week", "home_team", "away_team"], how="inner")
    return combined


def fit_and_score(train, test, feature_cols):
    train = train.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)
    X_train = np.column_stack([train[feature_cols].values, np.ones(len(train))])
    y_train = train["actual_margin"].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack([test[feature_cols].values, np.ones(len(test))])
    y_test = test["actual_margin"].values
    pred_test = X_test @ coef
    mae = np.mean(np.abs(pred_test - y_test))
    acc = ((pred_test > 0) == test["actual_home_win"]).mean()
    return coef, mae, acc


if __name__ == "__main__":
    print("Building fully combined dataset (DVOA + Elo + Layer 2 NGS)...")
    df = build_combined_dataset()
    print(f"\n{len(df)} real games with all features present\n")

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    print(f"Train: {len(train)} games (2016-2021), Test: {len(test)} games (2022-2023, fully held out)\n")

    base_cols = ["rating_diff", "home_field", "rest_diff"]
    layer2_cols = base_cols + ["cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"]
    full_cols = layer2_cols + ["elo_diff"]

    _, mae_base, acc_base = fit_and_score(train, test, base_cols)
    print(f"Base (rating only):        MAE={mae_base:.2f}, acc={acc_base:.4f}")

    _, mae_l2, acc_l2 = fit_and_score(train, test, layer2_cols)
    print(f"+ Layer 2 NGS:             MAE={mae_l2:.2f}, acc={acc_l2:.4f}")

    coef, mae_full, acc_full = fit_and_score(train, test, full_cols)
    print(f"+ Elo ensemble (full):     MAE={mae_full:.2f}, acc={acc_full:.4f}")
    print(f"\nFinal co-calibrated coefficients:")
    for name, val in zip(full_cols + ["intercept"], coef):
        print(f"  {name}: {val:.4f}")
