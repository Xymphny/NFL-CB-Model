"""
Empirical proxy for the PFF question: how much held-out accuracy does
TRENCH/PRESSURE information add to the existing full ensemble?

PFF's genuinely orthogonal contribution to this stack is line play --
pass-block/pass-rush grades and pressure charting. The NGS Layer 2
covers QB accuracy, receiver separation/YAC, and rushing over expected,
but nothing about the trenches. Before paying for PFF, this measures
what the same CLASS of information is worth, using the pressure events
already in play-by-play (sacks + QB hits per dropback) built strictly
walk-forward: each game sees only prior weeks of the same season,
credibility-blended toward league average early (same k-style shrinkage
as the preseason prior).

Design: stacking regressions fit on 2016-2021, graded on held-out
2022-2023 -- the identical split as the full-ensemble ATS backtest, so
numbers are directly comparable.
  Baseline:  actual_margin ~ ensemble_prediction
  Augmented: actual_margin ~ ensemble_prediction + pressure features
If PFF-class trench data moves this model, it must show up here first;
pbp pressure events are a noisier version of PFF's charting, so this
is a LOWER bound on PFF's value, but a lower bound of ~zero would say
the ensemble already absorbs trench effects through EPA.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from model.test_full_ensemble import build_combined_dataset
from model.prediction import predict_margin

SEASONS = list(range(2016, 2024))
TRAIN = list(range(2016, 2022))
TEST = [2022, 2023]
CRED_K = 60  # dropbacks of shrinkage toward league mean
BREAKEVEN = 0.5238
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet"


def build_pressure_table():
    """Per (season, week, team): walk-forward pressure rates using only
    weeks strictly before `week` of the same season."""
    rows = []
    for s in SEASONS:
        pbp = pd.read_parquet(PBP_URL.format(s=s), columns=["season", "week", "posteam", "defteam", "qb_dropback", "sack", "qb_hit"])
        pbp = pbp[(pbp["qb_dropback"] == 1) & pbp["posteam"].notna()]
        pbp["pressured"] = ((pbp["sack"].fillna(0) + pbp["qb_hit"].fillna(0)) > 0).astype(float)
        league_rate = pbp["pressured"].mean()

        off = pbp.groupby(["week", "posteam"]).agg(db=("pressured", "size"), pr=("pressured", "sum")).reset_index()
        deф = pbp.groupby(["week", "defteam"]).agg(db=("pressured", "size"), pr=("pressured", "sum")).reset_index()

        for wk in sorted(pbp["week"].unique()):
            o_prior = off[off["week"] < wk].groupby("posteam")[["db", "pr"]].sum()
            d_prior = deф[deф["week"] < wk].groupby("defteam")[["db", "pr"]].sum()
            teams = set(o_prior.index) | set(d_prior.index)
            for t in teams:
                odb, opr = (o_prior.loc[t] if t in o_prior.index else (0, 0))
                ddb, dpr = (d_prior.loc[t] if t in d_prior.index else (0, 0))
                # Credibility blend toward league mean, then center.
                off_rate = (opr + CRED_K * league_rate) / (odb + CRED_K) - league_rate
                def_rate = (dpr + CRED_K * league_rate) / (ddb + CRED_K) - league_rate
                rows.append({"season": s, "week": wk, "team": t,
                             "press_allowed": off_rate, "press_created": def_rate})
    return pd.DataFrame(rows)


def grade_ats(model_margin, spread_line, actual_margin, label):
    edge = model_margin - spread_line
    pushes = actual_margin == spread_line
    home_covers = actual_margin > spread_line
    picked_home = edge > 0
    graded = ~pushes
    win = (picked_home == home_covers)[graded]
    print(f"  {label}: MAE {np.mean(np.abs(model_margin - actual_margin)):.3f} | "
          f"ALL ATS {win.mean()*100:.2f}% ({win.sum()}/{len(win)})", end="")
    for th in [2, 4]:
        m = graded & (np.abs(edge) >= th)
        w = (picked_home == home_covers)[m]
        if len(w) >= 15:
            print(f" | >={th}: {w.mean()*100:.2f}% (n={len(w)})", end="")
    print()


def main():
    print("Building combined ensemble dataset...")
    combined = build_combined_dataset()
    combined = combined[combined["season"].isin(SEASONS)]

    games = pd.read_csv(os.environ.get("GAMES_CSV", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))
    lines = games[["season", "week", "home_team", "away_team", "spread_line"]].dropna()
    combined = combined.merge(lines, on=["season", "week", "home_team", "away_team"], how="inner")

    combined["ensemble"] = combined.apply(
        lambda r: predict_margin(r["rating_diff"], False, r["rest_diff"],
                                 cpoe_diff=r["cpoe_diff"], separation_diff=r["separation_diff"],
                                 yac_oe_diff=r["yac_oe_diff"], ryoe_diff=r["ryoe_diff"],
                                 elo_diff=r["elo_diff"]), axis=1)

    print("Building walk-forward pressure table (8 seasons of pbp)...")
    press = build_pressure_table()
    combined = combined.merge(press.rename(columns={"team": "home_team", "press_allowed": "h_allowed", "press_created": "h_created"}),
                              on=["season", "week", "home_team"], how="left")
    combined = combined.merge(press.rename(columns={"team": "away_team", "press_allowed": "a_allowed", "press_created": "a_created"}),
                              on=["season", "week", "away_team"], how="left")
    combined = combined.dropna(subset=["h_allowed", "a_allowed", "h_created", "a_created"])
    # Net pressure edge for the home side: our rush vs their protection,
    # minus their rush vs our protection.
    combined["press_edge"] = (combined["h_created"] + combined["a_allowed"]) - (combined["a_created"] + combined["h_allowed"])
    combined = combined.dropna(subset=["ensemble", "actual_margin", "spread_line", "press_edge"])

    train = combined[combined["season"].isin(TRAIN)]
    test = combined[combined["season"].isin(TEST)]
    print(f"Train {len(train)} games ({TRAIN[0]}-{TRAIN[-1]}), held out {len(test)} ({TEST}).")
    print(f"press_edge vs spread_resid corr (train): {np.corrcoef(train['press_edge'], train['actual_margin'] - train['spread_line'])[0,1]:+.4f}")

    def stack(features):
        X = np.column_stack([train[f] for f in features] + [np.ones(len(train))])
        coef, _, _, _ = np.linalg.lstsq(X, train["actual_margin"].values, rcond=None)
        Xt = np.column_stack([test[f] for f in features] + [np.ones(len(test))])
        return coef, Xt @ coef

    print("\nHeld-out 2022-2023, identical split as the ensemble ATS backtest:")
    coef_b, pred_b = stack(["ensemble"])
    grade_ats(pred_b, test["spread_line"].values, test["actual_margin"].values, "Baseline (ensemble only)   ")
    coef_a, pred_a = stack(["ensemble", "press_edge", "h_created", "a_allowed"])
    grade_ats(pred_a, test["spread_line"].values, test["actual_margin"].values, "With pressure features     ")
    print(f"\n  pressure coefficients (train): press_edge {coef_a[1]:+.1f}, h_created {coef_a[2]:+.1f}, a_allowed {coef_a[3]:+.1f}")
    print(f"  (units: points of margin per unit of centered pressure rate; a strong team is ~+/-0.03)")


if __name__ == "__main__":
    main()
