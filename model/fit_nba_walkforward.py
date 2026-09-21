#!/usr/bin/env python3
"""NBA ratings that update within a season, graded once on a clean holdout.

WHY THIS, AFTER THE LAST ATTEMPT FAILED
model/fit_nba.py fitted static ratings on 2021-2023 and graded them on
2024-2025: t = -6.96, decisively worse than predicting the league-average
margin. Three-year-old ratings are actively misleading in a sport with this
much roster turnover. The obvious repair is to rate WITHIN season and walk
forward, which is what the NFL and MLB models here already do.

HOLDOUT ACCOUNTING, STATED BECAUSE IT IS EASY TO LOSE TRACK OF
2024 and 2025 are SPENT -- graded once already for the static question. Using
them again would be test-set reuse, and each reuse raises the chance of a
spurious pass whatever the code says.

So: hyperparameters are chosen on 2021-2022, and 2023 is graded ONCE and has
never been graded before. 2024-2025 stay spent and are not touched here, which
leaves nothing clean for a third question -- a real cost of having asked the
static one first.

WHAT IS FITTED
An Elo-style rating in points of margin, updated after every game from the
prediction error, with a between-season regression toward the mean. Predicting
game N uses only games 1..N-1, so there is no lookahead by construction rather
than by discipline.

Writes model/nba_walkforward_results.json.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fit_data_checks import NBA, check_scores  # noqa: E402
from model.fit_nba import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nba_walkforward_results.json")

TUNE_SEASONS = (2021, 2022)
HOLDOUT_SEASON = 2023
SPENT_SEASONS = (2024, 2025)

#: Hyperparameter grid. Deliberately small: every extra combination is another
#: effective trial, and Bailey/Lopez de Prado's bound says the expected best
#: in-sample result from N pure-noise trials rises with N.
GRID = {
    "k": [0.02, 0.05, 0.10],
    "home_adv": [2.0, 2.5, 3.0],
    "carryover": [0.50, 0.75],
}


def walk_forward(df: pd.DataFrame, k: float, home_adv: float,
                 carryover: float, state: dict | None = None) -> pd.DataFrame:
    """Predict every game from prior games only.

    Pass `state` to receive the final ratings. See the note in the NHL
    script's walk_forward: a rating that lives only as a loop local is a
    number the artifact cannot justify.
    """
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    prev_season = None
    rows = []

    for g in df.itertuples():
        if g.season != prev_season:
            # New season: regress every rating toward the mean rather than
            # resetting. Resetting throws away real information; carrying
            # unchanged is the mistake the static fit made.
            ratings = {t: r * carryover for t, r in ratings.items()}
            prev_season = g.season

        rh = ratings.get(g.home, 0.0)
        ra = ratings.get(g.away, 0.0)
        mu = rh - ra + home_adv
        err = g.margin - mu

        ratings[g.home] = rh + k * err
        ratings[g.away] = ra - k * err
        rows.append({"season": g.season, "mu": mu, "margin": g.margin})

    if state is not None:
        state.update({"ratings": dict(ratings), "k": k,
                      "home_adv": home_adv, "carryover": carryover})
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame, seasons, sd: float | None = None):
    """Mean log-likelihood, and the sd used, on the given seasons."""
    p = pred[pred.season.isin(seasons)]
    resid = p.margin.values - p.mu.values
    s = sd if sd is not None else float(resid.std(ddof=1))
    ll = stats.norm.logpdf(p.margin.values, p.mu.values, s)
    return float(ll.mean()), s, p


def _noise_ceiling(trials: int, observed_t: float) -> dict:
    """The t a pure-noise search of this size is expected to produce.

    Bailey and Lopez de Prado's bound: with N independent trials on random
    data, the expected best in-sample t is roughly

        (1-g)*z(1 - 1/N) + g*z(1 - 1/(N*e)),  g = Euler-Mascheroni

    A result that does not clear its own search's noise ceiling is not a
    result, however small its p-value looks.
    """
    g = 0.5772156649
    ceiling = ((1 - g) * stats.norm.ppf(1 - 1 / trials)
               + g * stats.norm.ppf(1 - 1 / (trials * np.e)))
    return {
        "hyperparameter_trials": trials,
        "expected_best_t_from_pure_noise": round(float(ceiling), 3),
        "observed_holdout_t": round(observed_t, 2),
        "clears_noise_ceiling": bool(observed_t > ceiling),
        "note": ("the trials were spent choosing hyperparameters on the TUNING "
                 "seasons; the holdout was graded once with the chosen "
                 "values, so the ceiling bounds the selection, not the grade "
                 "directly. Reported because a search this size can "
                 "manufacture a tuning result that then fails to replicate."),
    }


def main() -> int:
    df = load((2021, 2022, 2023))
    check_scores(df, NBA, label="nba walk-forward")
    print(f"{len(df)} games, seasons {sorted(df.season.unique())}")
    print(f"tuning on {TUNE_SEASONS}, grading ONCE on {HOLDOUT_SEASON}")
    print(f"{SPENT_SEASONS} are spent by the static fit and are NOT used here")

    # --- choose hyperparameters on the tuning seasons only --------------
    best, trials = None, 0
    for k, ha, co in itertools.product(GRID["k"], GRID["home_adv"], GRID["carryover"]):
        pred = walk_forward(df, k, ha, co)
        ll, sd, _ = score(pred, TUNE_SEASONS)
        trials += 1
        if best is None or ll > best["ll"]:
            best = {"k": k, "home_adv": ha, "carryover": co, "ll": ll, "sd": sd}
    print(f"\n{trials} hyperparameter combinations tried on the tuning seasons")
    print(f"chosen: k={best['k']} home_adv={best['home_adv']} "
          f"carryover={best['carryover']}  (tuning ll {best['ll']:.5f})")

    # --- grade ONCE on the clean season ---------------------------------
    pred = walk_forward(df, best["k"], best["home_adv"], best["carryover"])
    ll_model, sd_used, hold = score(pred, [HOLDOUT_SEASON], sd=best["sd"])

    tune = df[df.season.isin(TUNE_SEASONS)]
    base_mu, base_sd = float(tune.margin.mean()), float(tune.margin.std(ddof=1))
    ll_base = float(stats.norm.logpdf(hold.margin.values, base_mu, base_sd).mean())

    d = (stats.norm.logpdf(hold.margin.values, hold.mu.values, sd_used)
         - stats.norm.logpdf(hold.margin.values, base_mu, base_sd))
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = float(d.mean() / se)

    art = {
        "_provenance": {
            "script": "model/fit_nba_walkforward.py", "generated": "2026-09-21",
            "tune_seasons": list(TUNE_SEASONS),
            "holdout_season": HOLDOUT_SEASON,
            "graded_once": True,
            "spent_seasons_not_used": list(SPENT_SEASONS),
            "spent_because": ("2024-2025 were graded once already for the "
                              "static cross-season question in fit_nba.py; "
                              "reusing them would be test-set reuse"),
            "hyperparameter_trials": trials,
            "metric": "mean log-likelihood of the realised margin",
            "baseline": "league-average margin and sd from the tuning seasons",
            "n_holdout": int(len(hold)),
        },
        "chosen_hyperparameters": {kk: best[kk] for kk in
                                   ("k", "home_adv", "carryover")},
        "sd_used": round(sd_used, 4),
        "holdout_grade": {
            "model": round(ll_model, 5),
            "baseline": round(ll_base, 5),
            "paired_gain": round(float(d.mean()), 5),
            "standard_error": round(se, 5),
            "t": round(t, 2),
            "supported": bool(t > 2),
        },
        "multiple_testing": _noise_ceiling(trials, t),
        "versus_static": {
            "static_t": -6.96,
            "note": ("the static cross-season fit measured t = -6.96 against "
                     "the same kind of baseline on a different holdout; the "
                     "two are not directly comparable but the sign change is "
                     "the point"),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=1)

    print(json.dumps(art["holdout_grade"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
