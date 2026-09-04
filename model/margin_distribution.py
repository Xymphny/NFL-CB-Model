"""
Two empirical replacements for the normal-CDF approximation, exported
as data/margin_dist.json for both backend and dashboard use:

1. ATS residual distribution: (actual margin - closing spread) over
   every nflverse game with a real closing line, 2010-2024. This is
   the real shape of scoring -- mass spiked on key margins -- and
   drives push probabilities and key-number math.

2. Calibrated edge -> cover probability: a logistic fit of ACTUAL
   cover outcomes against the model's edge vs the close, fit on the
   held-out full-ensemble backtest games (2022-2023). This is the
   honest answer to "when the model sees a 4-point edge, how often
   does that side really cover" -- and it is far flatter than the
   normal approximation, which is exactly why Kelly stakes must use
   it. Refit as real graded seasons accumulate.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DIST_SEASONS = range(2010, 2025)
BACKTEST_RESULTS = os.path.join(os.path.dirname(__file__), "ats_backtest_results.csv")


def build_residual_distribution(games=None):
    if games is None:
        games = pd.read_csv(GAMES_URL)
    g = games[games["season"].isin(DIST_SEASONS)].dropna(subset=["home_score", "away_score", "spread_line"])
    residual = (g["home_score"] - g["away_score"]) - g["spread_line"]

    lo, hi = -40, 40
    clipped = residual.clip(lo, hi)
    # Bin at 0.5 granularity (half-point lines make residuals half-integer).
    bins = np.round(clipped * 2) / 2
    pmf = bins.value_counts(normalize=True).sort_index()

    margins = (g["home_score"] - g["away_score"]).abs()
    margin_pmf = margins.value_counts(normalize=True).sort_index()

    return {
        "n_games": int(len(g)),
        "seasons": [int(DIST_SEASONS.start), int(DIST_SEASONS.stop - 1)],
        "residual_pmf": {str(k): round(float(v), 6) for k, v in pmf.items()},
        "residual_std": round(float(residual.std()), 3),
        "key_margin_mass": {str(int(m)): round(float(margin_pmf.get(m, 0)), 4) for m in [1, 2, 3, 4, 6, 7, 8, 10, 14]},
        "push_prob_integer_line": round(float((residual == 0).mean() / max((g["spread_line"] % 1 == 0).mean(), 1e-9) * (g["spread_line"] % 1 == 0).mean() / max((g["spread_line"] % 1 == 0).mean(), 1e-9)), 6),
    }


def fit_edge_calibration():
    """Logistic fit: P(model's side covers) vs |edge| on held-out games."""
    df = pd.read_csv(BACKTEST_RESULTS)
    edge = df["model_margin"] - df["spread_line"]
    pushes = df["actual_margin"] == df["spread_line"]
    home_covers = df["actual_margin"] > df["spread_line"]
    picked_home = edge > 0
    d = pd.DataFrame({
        "edge": edge.abs(),
        "covered": (picked_home == home_covers).astype(float),
    })[~pushes]

    # 1-parameter logistic through 0.5 at edge=0: p = 1/(1+exp(-b*edge)).
    # Symmetry is forced deliberately: with no edge the pick is a coin
    # flip by construction, so the intercept is not a free parameter --
    # and 372 games can support one honest parameter, not two.
    from scipy.optimize import minimize_scalar

    def nll(b):
        p = 1 / (1 + np.exp(-b * d["edge"]))
        p = p.clip(1e-6, 1 - 1e-6)
        return -(d["covered"] * np.log(p) + (1 - d["covered"]) * np.log(1 - p)).sum()

    res = minimize_scalar(nll, bounds=(0.0, 0.2), method="bounded")
    b = float(res.x)

    # Report the fit against reality in buckets so the number can be
    # sanity-checked by eye, not just trusted.
    buckets = []
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 99)]:
        m = (d["edge"] >= lo) & (d["edge"] < hi)
        if m.sum() >= 15:
            buckets.append({
                "edge_range": [lo, hi if hi < 99 else None],
                "n": int(m.sum()),
                "actual_cover_pct": round(float(d.loc[m, "covered"].mean()), 4),
                "fitted_at_midpoint": round(float(1 / (1 + np.exp(-b * d.loc[m, "edge"].mean()))), 4),
            })

    return {"edge_coef": round(b, 5), "n_games": int(len(d)), "fit_check": buckets,
            "source": "held-out 2022-2023 full-ensemble backtest",
            "note": "P(cover) = 1/(1+exp(-edge_coef*|edge|)). Refit when real graded seasons accumulate."}


def main():
    dist = build_residual_distribution()
    calib = fit_edge_calibration()
    out = {"residual_distribution": dist, "edge_calibration": calib}
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "margin_dist.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    print(f"  residual dist over {dist['n_games']} games (2010-2024), std {dist['residual_std']}")
    print(f"  key margin mass: 3 -> {dist['key_margin_mass']['3']}, 7 -> {dist['key_margin_mass']['7']}")
    print(f"  edge calibration: coef {calib['edge_coef']} over {calib['n_games']} held-out games")
    for bkt in calib["fit_check"]:
        print(f"    edge {bkt['edge_range']}: actual {bkt['actual_cover_pct']*100:.1f}% vs fitted {bkt['fitted_at_midpoint']*100:.1f}% (n={bkt['n']})")
    print(f"  For comparison, the normal approximation claims a 4-pt edge covers "
          f"{100*0.5*(1+__import__("math").erf((4/13.86)/np.sqrt(2))):.1f}% -- the calibrated answer is "
          f"{100/(1+np.exp(-calib['edge_coef']*4)):.1f}%.")


if __name__ == "__main__":
    main()
