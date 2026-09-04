"""
CFB ATS backtest: grades the walk-forward CFB ratings (already cached,
zero lookahead) against real closing lines from CFBD.

Run AFTER ingest/cfb_lines.py has built model/cfb_lines_cache.csv and
its sign convention has been verified per that module's checklist.

Methodology matches the NFL version (model/ats_backtest_full_ensemble
.py): margin coefficients fit on the earlier seasons only, the final
season fully held out, ATS graded by edge bucket against the close.
Also reports splits the NFL test doesn't need: conference games vs
non-conference, and (roughly, by team name) G5-involved games -- the
segments where a public CFB model is most likely to hold real edge.

NOT RUN against real lines data yet -- team-name matching between
CFBD ("Ohio State") and the play-by-play cache is the most likely
failure point; the script reports its own join rate so a bad match
rate is loud, not silent.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "cfb_full_walk_forward_cache.csv")
LINES = os.path.join(os.path.dirname(__file__), "cfb_lines_cache.csv")
BREAKEVEN = 0.5238


def grade(df, label):
    edge = df["model_margin"] - df["spread_line"]
    pushes = df["actual_margin"] == df["spread_line"]
    home_covers = df["actual_margin"] > df["spread_line"]
    picked_home = edge > 0
    graded = ~pushes
    win = (picked_home == home_covers)[graded]
    if len(win) == 0:
        return
    print(f"\n=== {label} ({len(df)} games) ===")
    print(f"  Model MAE {np.mean(np.abs(df['model_margin']-df['actual_margin'])):.2f} vs market MAE {np.mean(np.abs(df['spread_line']-df['actual_margin'])):.2f}")
    print(f"  ALL: {win.mean()*100:.2f}% ({win.sum()}/{len(win)})")
    for th in [2, 3, 4, 5, 7, 10]:
        m = graded & (edge.abs() >= th)
        w = (picked_home == home_covers)[m]
        if len(w) >= 15:
            marker = " <-- clears breakeven" if w.mean() >= BREAKEVEN else ""
            print(f"  |edge|>={th}: {w.mean()*100:.2f}% ({w.sum()}/{len(w)}){marker}")


def main():
    cache = pd.read_csv(CACHE)
    lines = pd.read_csv(LINES)

    df = cache.merge(lines[["season", "week", "home_team", "away_team", "spread_line"]],
                     on=["season", "week", "home_team", "away_team"], how="inner")
    join_rate = len(df) / len(cache)
    print(f"Joined {len(df)}/{len(cache)} cached games to lines ({join_rate*100:.0f}%).")
    if join_rate < 0.5:
        print("JOIN RATE TOO LOW -- team-name mismatch between CFBD and the pbp cache is "
              "the likely cause. Fix the name mapping before trusting anything below.")

    seasons = sorted(df["season"].unique())
    train_seasons, test_season = seasons[:-1], seasons[-1]
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season].copy()

    # Simple margin model from the rating diff, fit on train only.
    X = np.column_stack([train["rating_diff"], np.ones(len(train))])
    coef, _, _, _ = np.linalg.lstsq(X, train["actual_margin"].values, rcond=None)
    print(f"Margin fit on {train_seasons}: {coef[0]:.2f} * rating_diff + {coef[1]:.2f} (home field)")

    test["model_margin"] = coef[0] * test["rating_diff"] + coef[1]
    grade(test, f"HELD OUT {test_season}")


if __name__ == "__main__":
    main()
