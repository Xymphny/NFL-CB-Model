#!/usr/bin/env python3
"""Measure NFL key-number weights and grade them once on held-out seasons.

Writes data/nfl_key_numbers.json.

WHY THIS FILE EXISTS AT ALL
The first version of this measurement was run as an inline script and thrown
away, leaving a committed artifact whose _provenance named a generator that
was not in the repo. That is precisely the failure the project carries a
permanent ledger row about -- eight shipped constants citing a grid search
whose output exists in no committed file. The artifact guard caught it, which
is the only reason it is not a second instance of the same loss.

WHAT IS MEASURED
Football margins carry excess mass on 3 and 7 and almost none on 0. A rounded
normal does not reproduce that. The correction is multiplicative, not an
absolute table: P(margin = 3) must depend on the spread, so what is constant
across games is the RATIO of true mass to normal mass at each margin.

    w(k) = (empirical count at k) / (summed rounded-normal mass at k)

with each game's normal centred on its own closing spread. Weights are applied
then renormalised, so they change shape and cannot change total probability.

DISCIPLINE
Fit on 2010-2021. Graded ONCE on 2022-2025, on mean log-likelihood of the
realised margin -- a proper scoring rule, so it cannot be gamed by making the
distribution more confident. Re-running this script against a different
holdout window spends that window; say so in the ledger if you do.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "nfl_key_numbers.json")

TRAIN_SEASONS = (2010, 2021)
HOLDOUT_SEASONS = (2022, 2025)
SUPPORT = range(-40, 41)

#: A weight is only estimated where the normal predicts enough mass for the
#: ratio to mean anything. Below this, empirical counts are too noisy and the
#: weight would be fitting single games.
MIN_PREDICTED_MASS = 20.0


def load_games(url: str = GAMES_URL) -> pd.DataFrame:
    g = pd.read_csv(url)
    g = g[g["game_type"] == "REG"].dropna(
        subset=["home_score", "away_score", "spread_line"])
    g = g.copy()
    g["margin"] = g["home_score"] - g["away_score"]
    return g


def _season_slice(g: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return g[(g["season"] >= lo) & (g["season"] <= hi)]


def predicted_mass(df: pd.DataFrame, sigma: float) -> np.ndarray:
    """Rounded-normal mass at each k, per game, centred on the closing spread."""
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
    """Mean log-likelihood of the realised margin, reweighted versus plain."""
    idx = {k: i for i, k in enumerate(SUPPORT)}
    keep = np.array([m in idx for m in holdout["margin"]])
    rows = np.array([idx[m] for m in holdout["margin"] if m in idx])
    P = predicted_mass(holdout[keep], sigma)

    plain = P / P.sum(axis=1, keepdims=True)
    w = np.array([weights.get(k, 1.0) for k in SUPPORT])
    rw = P * w[None, :]
    rw = rw / rw.sum(axis=1, keepdims=True)

    n = len(rows)
    lp = np.log(np.clip(plain[np.arange(n), rows], 1e-12, None))
    lr = np.log(np.clip(rw[np.arange(n), rows], 1e-12, None))
    d = lr - lp
    gain = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))

    checks = {}
    for k in (0, 3, 6, 7, 10, 14):
        i = idx[k]
        checks[str(k)] = {
            "plain": round(float(plain[:, i].mean()), 4),
            "reweighted": round(float(rw[:, i].mean()), 4),
            "empirical_holdout": round(float(np.mean(holdout["margin"].values == k)), 4),
        }

    return {
        "n": n,
        "grade": {
            "metric": "mean log-likelihood of the realised margin",
            "plain_rounded_normal": round(float(lp.mean()), 5),
            "key_number_reweighted": round(float(lr.mean()), 5),
            "paired_gain": round(gain, 5),
            "standard_error": round(se, 5),
            "t": round(gain / se, 2),
            "supported": bool(gain / se > 2),
        },
        "push_probability_check": checks,
    }


def main() -> int:
    g = load_games()
    train = _season_slice(g, *TRAIN_SEASONS)
    hold = _season_slice(g, *HOLDOUT_SEASONS)
    sigma = float((train["margin"] - train["spread_line"]).std())

    weights = fit_weights(train, sigma)
    graded = grade(hold, sigma, weights)

    art = {
        "_provenance": {
            "script": "model/nfl_key_numbers.py",
            "generated": "2026-09-21",
            "source": "nflverse nfldata games.csv, REG season games with a closing spread",
            "train_seasons": list(TRAIN_SEASONS),
            "holdout_seasons": list(HOLDOUT_SEASONS),
            "graded_once": True,
            "method": (
                "multiplicative weights on a rounded normal centred per game on the "
                "closing spread. w(k) = empirical count at k / summed predicted mass "
                "at k. Weights are applied then renormalised, so they adjust SHAPE "
                "and cannot change total probability."),
            "why_multiplicative": (
                "an absolute mass table would be wrong for a conditional "
                "distribution: P(margin=3) must depend on how big the spread is. The "
                "excess at key numbers is a property of how football scores, not of "
                "the game."),
            "min_predicted_mass_to_estimate": MIN_PREDICTED_MASS,
            "sigma_used": round(sigma, 4),
            "n_train": int(len(train)),
            "n_holdout": graded["n"],
        },
        "weights": {str(k): v for k, v in sorted(weights.items())},
        "holdout_grade": graded["grade"],
        "push_probability_check": graded["push_probability_check"],
        "note": (
            "The plain rounded normal understates P(margin=3) by nearly 3x. Any push "
            "price or key-number calculation made without these weights was wrong by "
            "that factor."),
    }

    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=1)
    t = art["holdout_grade"]["t"]
    print(f"wrote {OUT}")
    print(f"  {len(weights)} weights, held-out t = {t}, "
          f"supported = {art['holdout_grade']['supported']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
