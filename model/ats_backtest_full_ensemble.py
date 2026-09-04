"""
The question this whole project ultimately turns on: does the FULL
production ensemble (DVOA rating + Layer 2 NGS + Elo, the exact
deployed MARGIN_COEFFICIENTS) beat the closing spread often enough to
clear the 52.4% ATS breakeven -- and at what edge threshold?

Methodology mirrors test_full_ensemble.py exactly (same dataset
builder, same walk-forward NGS/Elo construction, zero lookahead in
features). Grading is strictly held out: MARGIN_COEFFICIENTS was fit
on 2016-2021, so only 2022-2023 games are scored -- outcomes those
coefficients never saw. Closing lines: nflverse's real spread_line
(positive = home favored, empirically verified sign convention in
test_market_blending.py).

An earlier ratings-only version of this test (2023 held out) came in
at 46.7% ATS overall / 45.2% at the 2-point flag threshold -- below
breakeven. This script answers whether the full ensemble changes that.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from model.test_full_ensemble import build_combined_dataset
from model.prediction import predict_margin

HELD_OUT_SEASONS = [2022, 2023]
BREAKEVEN = 0.5238


def grade(df, label):
    edge = df["model_margin"] - df["spread_line"]
    pushes = df["actual_margin"] == df["spread_line"]
    home_covers = df["actual_margin"] > df["spread_line"]
    picked_home = edge > 0
    graded = ~pushes

    win = (picked_home == home_covers)[graded]
    print(f"\n=== {label} ({len(df)} games, {graded.sum()} graded, {pushes.sum()} pushes) ===")
    print(f"  Model MAE vs actual: {np.mean(np.abs(df['model_margin'] - df['actual_margin'])):.2f}")
    print(f"  Market MAE vs actual: {np.mean(np.abs(df['spread_line'] - df['actual_margin'])):.2f}")
    print(f"  ALL games ATS: {win.mean()*100:.2f}% ({win.sum()}/{len(win)})  [breakeven {BREAKEVEN*100:.1f}%]")

    rows = []
    for th in [1, 2, 2.5, 3, 4, 5, 6]:
        mask = graded & (edge.abs() >= th)
        w = (picked_home == home_covers)[mask]
        if len(w) < 10:
            continue
        pct = w.mean()
        rows.append({"threshold": th, "ats_pct": pct, "n": int(len(w)), "wins": int(w.sum())})
        marker = " <-- clears breakeven" if pct >= BREAKEVEN else ""
        print(f"  |edge|>={th}: {pct*100:.2f}% ({w.sum()}/{len(w)}){marker}")
    return rows


def main():
    print("Building the combined walk-forward dataset (same builder as test_full_ensemble)...")
    combined = build_combined_dataset()

    games = pd.read_csv("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
    lines = games[["season", "week", "home_team", "away_team", "spread_line"]].dropna()
    combined = combined.merge(lines, on=["season", "week", "home_team", "away_team"], how="inner")

    combined["model_margin"] = combined.apply(
        lambda r: predict_margin(
            r["rating_diff"], False, r["rest_diff"],
            cpoe_diff=r["cpoe_diff"], separation_diff=r["separation_diff"],
            yac_oe_diff=r["yac_oe_diff"], ryoe_diff=r["ryoe_diff"],
            elo_diff=r["elo_diff"],
        ),
        axis=1,
    )

    held_out = combined[combined["season"].isin(HELD_OUT_SEASONS)]
    grade(held_out, "HELD OUT 2022-2023 (coefficients never saw these outcomes)")
    for season in HELD_OUT_SEASONS:
        grade(held_out[held_out["season"] == season], f"Held out {season} alone")

    out_path = os.path.join(os.path.dirname(__file__), "ats_backtest_results.csv")
    held_out.to_csv(out_path, index=False)
    print(f"\nPer-game results saved to {out_path}")


if __name__ == "__main__":
    main()
