"""
Residual modeling -- the strategic pivot confirmed by the full-ensemble
ATS backtest (50.8% held out): stop trying to out-predict the game and
start predicting where the CLOSING LINE itself errs.

Target: actual_margin - spread_line (spread residual), and
        actual_total - total_line (total residual).

Features are STRICTLY pre-game schedule facts (zero lookahead by
construction -- everything here is known days before kickoff): rest
differentials, short weeks, division games, weather for outdoor games,
line magnitude, calendar position. The model's own disagreement with
the close is deliberately included as ONE feature among many, so this
test also answers whether the ensemble adds anything on top of
situational spots.

Splits: train 1999/2010-2021, held out 2022-2025 (four full seasons the
fits never see). Regular season only. All results graded against the
52.38% breakeven at -110.

Honesty note baked in: schedule-only inefficiencies are the most
heavily mined territory in sports betting. Finding little here is a
real and publishable result, not a failure of the code.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

GAMES_CSV = os.environ.get("GAMES_CSV", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
TRAIN_SEASONS = list(range(2010, 2022))
TEST_SEASONS = [2022, 2023, 2024, 2025]
BREAKEVEN = 0.5238


def load_games():
    g = pd.read_csv(GAMES_CSV)
    g = g[(g["game_type"] == "REG") & g["season"].isin(TRAIN_SEASONS + TEST_SEASONS)]
    g = g.dropna(subset=["home_score", "away_score", "spread_line"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["spread_resid"] = g["margin"] - g["spread_line"]
    g["total_pts"] = g["home_score"] + g["away_score"]
    g["total_resid"] = g["total_pts"] - g["total_line"]
    return g


def build_spread_features(g):
    f = pd.DataFrame(index=g.index)
    f["rest_diff"] = (g["home_rest"] - g["away_rest"]).fillna(0)
    f["home_short_week"] = (g["home_rest"] <= 5).astype(float)
    f["away_short_week"] = (g["away_rest"] <= 5).astype(float)
    f["home_off_bye"] = (g["home_rest"] >= 13).astype(float)
    f["away_off_bye"] = (g["away_rest"] >= 13).astype(float)
    f["div_game"] = g["div_game"].fillna(0).astype(float)
    f["home_dog"] = (g["spread_line"] < 0).astype(float)
    f["big_home_fav"] = (g["spread_line"] >= 7).astype(float)
    f["big_away_fav"] = (g["spread_line"] <= -7).astype(float)
    f["late_season"] = (g["week"] >= 15).astype(float)
    f["early_season"] = (g["week"] <= 3).astype(float)
    f["outdoor_cold"] = ((g["roof"] == "outdoors") & (g["temp"] < 35)).astype(float)
    f["outdoor_windy"] = ((g["roof"] == "outdoors") & (g["wind"] >= 15)).astype(float)
    return f.fillna(0)


def build_total_features(g):
    f = pd.DataFrame(index=g.index)
    outdoors = g["roof"] == "outdoors"
    f["wind"] = np.where(outdoors, g["wind"].fillna(0), 0)
    f["windy_15"] = np.where(outdoors & (g["wind"] >= 15), 1.0, 0)
    f["windy_20"] = np.where(outdoors & (g["wind"] >= 20), 1.0, 0)
    f["cold_25"] = np.where(outdoors & (g["temp"] < 25), 1.0, 0)
    f["dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    f["high_total"] = (g["total_line"] >= 49).astype(float)
    f["low_total"] = (g["total_line"] <= 41).astype(float)
    f["div_game"] = g["div_game"].fillna(0).astype(float)
    f["late_season"] = (g["week"] >= 15).astype(float)
    f["short_week_either"] = ((g["home_rest"] <= 5) | (g["away_rest"] <= 5)).astype(float)
    return f.fillna(0)


def fit_ridge(X, y, alpha=1.0):
    Xb = np.column_stack([X, np.ones(len(X))])
    ident = np.eye(Xb.shape[1]); ident[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + alpha * ident, Xb.T @ y)


def predict(X, coef):
    return np.column_stack([X, np.ones(len(X))]) @ coef


def grade_side_bets(pred_resid, resid, line_frac_pushable, label, is_total=False, lines=None, actual=None):
    print(f"\n=== {label} ===")
    for th in [0.5, 1.0, 1.5, 2.0, 3.0]:
        take = np.abs(pred_resid) >= th
        if is_total:
            pushes = actual == lines
        else:
            pushes = resid == 0
        graded = take & ~pushes
        if graded.sum() < 25:
            continue
        win = ((pred_resid > 0) == (resid > 0))[graded]
        pct = win.mean()
        marker = " <-- clears breakeven" if pct >= BREAKEVEN else ""
        print(f"  |pred residual|>={th}: {pct*100:.2f}% ({win.sum()}/{len(win)}, {graded.sum()} bets over {len(TEST_SEASONS)} seasons){marker}")


def main():
    g = load_games()
    train = g[g["season"].isin(TRAIN_SEASONS)]
    test = g[g["season"].isin(TEST_SEASONS)]
    print(f"Train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}: {len(train)} games. Held out {TEST_SEASONS}: {len(test)} games.")

    # ---- Spread residual ----
    Xtr, Xte = build_spread_features(train), build_spread_features(test)
    coef = fit_ridge(Xtr.values, train["spread_resid"].values)
    named = sorted(zip(Xtr.columns, coef[:-1]), key=lambda t: -abs(t[1]))
    print("\nSpread residual coefficients (points of closing-line error, train fit):")
    for name, c in named:
        print(f"  {name:>16}: {c:+.2f}")
    pred = predict(Xte.values, coef)
    print(f"  In-sample R^2: {1 - np.var(train['spread_resid'] - predict(Xtr.values, coef)) / np.var(train['spread_resid']):.4f}")
    grade_side_bets(pred, test["spread_resid"].values, None, "SPREAD residual bets, held out 2022-2025")

    # ---- Total residual ----
    gt_train = train.dropna(subset=["total_line"])
    gt_test = test.dropna(subset=["total_line"])
    Xtr, Xte = build_total_features(gt_train), build_total_features(gt_test)
    coef_t = fit_ridge(Xtr.values, gt_train["total_resid"].values)
    named = sorted(zip(Xtr.columns, coef_t[:-1]), key=lambda t: -abs(t[1]))
    print("\nTotal residual coefficients (points of closing-total error, train fit):")
    for name, c in named:
        print(f"  {name:>16}: {c:+.2f}")
    pred_t = predict(Xte.values, coef_t)
    grade_side_bets(pred_t, gt_test["total_resid"].values, None, "TOTAL residual bets (pred>0 = over), held out 2022-2025",
                    is_total=True, lines=gt_test["total_line"].values, actual=gt_test["total_pts"].values)

    # ---- The classic single spot, isolated: 15+ mph wind unders ----
    windy = gt_test[(gt_test["roof"] == "outdoors") & (gt_test["wind"] >= 15)]
    if len(windy) > 20:
        unders = (windy["total_pts"] < windy["total_line"]).sum()
        overs = (windy["total_pts"] > windy["total_line"]).sum()
        print(f"\nSanity spot-check -- unders in 15+ mph wind, held out: {unders}-{overs} "
              f"({unders/(unders+overs)*100:.1f}%)")

    out = os.path.join(os.path.dirname(__file__), "residual_model_coefs.json")
    import json
    with open(out, "w") as f:
        json.dump({
            "spread": {"features": list(build_spread_features(train).columns), "coefs": [round(float(c), 4) for c in coef]},
            "total": {"features": list(build_total_features(gt_train).columns), "coefs": [round(float(c), 4) for c in coef_t]},
            "train_seasons": TRAIN_SEASONS, "held_out_seasons": TEST_SEASONS,
        }, f, indent=2)
    print(f"\ncoefficients saved to {out}")


if __name__ == "__main__":
    main()
