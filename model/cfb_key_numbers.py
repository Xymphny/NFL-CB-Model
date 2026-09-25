#!/usr/bin/env python3
"""Measure CFB key-number weights and grade them once on a held-out season.

Writes data/cfb_key_numbers.json. The method is model/nfl_key_numbers.py's,
unchanged, so the two artifacts mean the same thing:

    w(k) = (empirical count at k) / (summed rounded-normal mass at k)

with each game's normal centred on its own closing spread, weights applied
then renormalised. See that file for why the correction is multiplicative.

WHAT IS DIFFERENT FOR CFB
The margin is much wider (sd ~16 around the close against NFL's ~13) and
lopsided games are common, so the support runs to +-80 and the 3/7 excess is
expected to be diluted, not absent. Whether it is large enough to matter is
the question the holdout answers.

GRADED AT THE SIGMA THE MODEL SHIPS. The weights are fitted around the market
line at the market's own residual sigma, but CFBModel prices with MARGIN_SD
(17.8047), which is wider. A table that helps at one width and not the other
would be mis-sold by a grade at the convenient one, so the holdout is scored
at both and `supported` requires both.

DISCIPLINE
Lines: model/cfb_lines_cache.csv (CFBD closing lines, positive = home
favoured, with final scores), 2021-2023. Fit on 2021-2022. Graded ONCE on
2023, on mean log-likelihood of the realised margin. Re-running against a
different holdout spends that window; say so in the ledger if you do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LINES = ROOT / "model" / "cfb_lines_cache.csv"
OUT = ROOT / "data" / "cfb_key_numbers.json"

TRAIN_SEASONS = (2021, 2022)
HOLDOUT_SEASONS = (2023, 2023)
SUPPORT = range(-80, 81)
#: As NFL: a weight is only estimated where the normal predicts enough mass.
MIN_PREDICTED_MASS = 20.0
#: The width CFBModel prices with (src/coverline/leagues/cfb/model.py).
SHIPPED_SIGMA = 17.8047


def load_games(path: Path = LINES) -> pd.DataFrame:
    g = pd.read_csv(path).dropna(subset=["home_score", "away_score", "spread_line"])
    g = g.drop_duplicates(["season", "week", "home_team", "away_team"]).copy()
    g["margin"] = (g["home_score"] - g["away_score"]).astype(int)
    return g


def _season_slice(g: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return g[(g["season"] >= lo) & (g["season"] <= hi)]


def predicted_mass(df: pd.DataFrame, sigma: float) -> np.ndarray:
    mu = df["spread_line"].values[:, None]
    k = np.array(list(SUPPORT))[None, :]
    return (stats.norm.cdf((k + 0.5 - mu) / sigma)
            - stats.norm.cdf((k - 0.5 - mu) / sigma))


def fit_weights(train: pd.DataFrame, sigma: float) -> dict[int, float]:
    pred = predicted_mass(train, sigma).sum(axis=0)
    emp = train["margin"].value_counts().reindex(list(SUPPORT)).fillna(0).values
    return {int(k): round(float(e / p), 4)
            for k, e, p in zip(SUPPORT, emp, pred) if p >= MIN_PREDICTED_MASS}


def grade(holdout: pd.DataFrame, sigma: float, weights: dict[int, float]) -> dict:
    idx = {k: i for i, k in enumerate(SUPPORT)}
    keep = holdout["margin"].isin(idx).values
    rows = np.array([idx[m] for m in holdout["margin"][keep]])
    P = predicted_mass(holdout[keep], sigma)
    plain = P / P.sum(axis=1, keepdims=True)
    w = np.array([weights.get(k, 1.0) for k in SUPPORT])
    rw = P * w[None, :]
    rw = rw / rw.sum(axis=1, keepdims=True)

    n = len(rows)
    lp = np.log(np.clip(plain[np.arange(n), rows], 1e-12, None))
    lr = np.log(np.clip(rw[np.arange(n), rows], 1e-12, None))
    d = lr - lp
    gain, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    checks = {str(k): {
        "plain": round(float(plain[:, idx[k]].mean()), 4),
        "reweighted": round(float(rw[:, idx[k]].mean()), 4),
        "empirical_holdout": round(float(np.mean(holdout["margin"].values == k)), 4),
    } for k in (0, 3, 6, 7, 10, 14, 17, 21)}
    return {
        "n": n,
        "sigma": round(sigma, 4),
        "plain_rounded_normal": round(float(lp.mean()), 5),
        "key_number_reweighted": round(float(lr.mean()), 5),
        "paired_gain": round(gain, 5),
        "standard_error": round(se, 5),
        "t": round(gain / se, 2),
        "supported": bool(gain / se > 2),
        "push_probability_check": checks,
    }


def build(g: pd.DataFrame, generated: str) -> dict:
    train = _season_slice(g, *TRAIN_SEASONS)
    hold = _season_slice(g, *HOLDOUT_SEASONS)
    sigma = float((train["margin"] - train["spread_line"]).std())
    weights = fit_weights(train, sigma)
    at_fit = grade(hold, sigma, weights)
    at_shipped = grade(hold, SHIPPED_SIGMA, weights)
    checks = at_shipped.pop("push_probability_check")
    at_fit.pop("push_probability_check")
    supported = at_fit["supported"] and at_shipped["supported"]
    return {
        "_provenance": {
            "script": "model/cfb_key_numbers.py",
            "generated": generated,
            "source": "model/cfb_lines_cache.csv: CFBD closing lines with final scores",
            "train_seasons": list(TRAIN_SEASONS),
            "holdout_seasons": list(HOLDOUT_SEASONS),
            "graded_once": True,
            "method": "as model/nfl_key_numbers.py: w(k) = empirical / summed "
                      "rounded-normal mass, centred per game on the closing spread, "
                      "applied then renormalised",
            "min_predicted_mass_to_estimate": MIN_PREDICTED_MASS,
            "sigma_used": round(sigma, 4),
            "n_train": int(len(train)),
            "n_holdout": at_shipped["n"],
            "sign_check_corr_line_margin": round(float(np.corrcoef(
                g["spread_line"], g["margin"])[0, 1]), 4),
        },
        "weights": {str(k): v for k, v in sorted(weights.items())},
        "holdout_grade": {
            "metric": "mean log-likelihood of the realised margin",
            "at_fit_sigma": at_fit,
            "at_shipped_sigma": at_shipped,
            # Headline numbers are the shipped width's: that is what prices.
            "paired_gain": at_shipped["paired_gain"],
            "t": at_shipped["t"],
            "supported": supported,
        },
        "push_probability_check": checks,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated", required=True, help="date to stamp, YYYY-MM-DD")
    a = ap.parse_args(argv)
    art = build(load_games(), a.generated)
    OUT.write_text(json.dumps(art, indent=1) + "\n")
    hg = art["holdout_grade"]
    print(f"wrote {OUT.relative_to(ROOT)}: {len(art['weights'])} weights")
    for k in ("at_fit_sigma", "at_shipped_sigma"):
        r = hg[k]
        print(f"  {k} ({r['sigma']}): gain {r['paired_gain']:+.4f} "
              f"se {r['standard_error']:.4f} t {r['t']}")
    print(f"  supported = {hg['supported']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
