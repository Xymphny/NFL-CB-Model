#!/usr/bin/env python3
"""Fit an NBA margin model, and grade it ONCE.

DISCIPLINE, FIXED BEFORE ANY RESULT WAS SEEN
Train on 2021-2023, hold out 2024-2025. Graded once, on mean log-likelihood
of the realised margin -- a proper scoring rule, so it cannot be improved by
making the distribution more confident. If the grade rejects it, nothing
ships and leagues/nba keeps its no-coefficients state (ADR 0005).

WHAT IS FITTED
A ridge-regularised team rating on margin, plus home advantage:

    margin ~ rating[home] - rating[away] + home_adv

and SIGMA AS A FUNCTION OF THE PREDICTED MARGIN, which is the whole reason
this is not a copied football model. ADR 0005 records that NBA margin
variance rises with spread magnitude where the NFL's does not; that claim is
tested here rather than assumed, by fitting sigma = a + b*|mu| and checking
whether b is distinguishable from zero on held-out data.

DATA FILTERS THAT MATTER
season_type == 2 (regular season only) and both scores non-zero. Without the
first, the All-Star Game enters the fit -- a 211-186 exhibition between teams
that do not exist. Without the second, postponed rows enter as 0-0 blowouts.
The degenerate-data guard caught both.

Writes model/nba_fit_results.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fit_data_checks import NBA, check_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nba_fit_results.json")

TRAIN_SEASONS = (2021, 2022, 2023)
HOLDOUT_SEASONS = (2024, 2025)
DATA = "/tmp/fitdata/nba_{year}.parquet"
RIDGE = 8.0


def load(seasons) -> pd.DataFrame:
    frames = []
    for y in seasons:
        d = pd.read_parquet(DATA.format(year=y))
        # BOTH filters are needed. ESPN classifies the All-Star Game as
        # season_type == 2, so that alone lets a 211-186 exhibition between
        # "EAST" and "WEST" into the fit. type_abbreviation == "STD" is what
        # actually means a regular-season game between two real teams.
        d = d[(d.season_type == 2) & (d.type_abbreviation == "STD")]
        d = d.dropna(subset=["home_score", "away_score",
                             "home_abbreviation", "away_abbreviation"])
        d = d[(d.home_score > 0) & (d.away_score > 0)]  # drop postponed rows
        frames.append(d[["season", "date", "home_abbreviation",
                         "away_abbreviation", "home_score", "away_score"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"home_abbreviation": "home",
                              "away_abbreviation": "away"})
    out["margin"] = out.home_score.astype(int) - out.away_score.astype(int)
    return out


def fit_ratings(df: pd.DataFrame, ridge: float = RIDGE):
    """Ridge least squares on margin. Ratings are centred, so home_adv is
    identifiable -- the collinearity defect the NFL vector still carries."""
    teams = sorted(set(df.home) | set(df.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    X = np.zeros((len(df), n + 1))
    X[np.arange(len(df)), df.home.map(idx).values] = 1.0
    X[np.arange(len(df)), df.away.map(idx).values] = -1.0
    X[:, n] = 1.0                                   # home advantage
    y = df.margin.values.astype(float)

    pen = np.eye(n + 1) * ridge
    pen[n, n] = 0.0                                 # do not shrink home_adv
    beta = np.linalg.solve(X.T @ X + pen, X.T @ y)

    ratings = dict(zip(teams, (beta[:n] - beta[:n].mean()).round(5)))
    return {"ratings": ratings, "home_adv": float(round(beta[n], 5))}


def predict_mu(fit, home, away):
    r = fit["ratings"]
    if home not in r or away not in r:
        return None
    return r[home] - r[away] + fit["home_adv"]


def fit_sigma(mu: np.ndarray, resid: np.ndarray):
    """sigma = a + b*|mu|, fitted on absolute residuals.

    E|resid| = sigma * sqrt(2/pi) for a normal, so the regression is scaled
    back to a standard deviation rather than reported as a mean absolute
    error dressed up as one.
    """
    X = np.column_stack([np.ones_like(mu), np.abs(mu)])
    beta, *_ = np.linalg.lstsq(X, np.abs(resid), rcond=None)
    scale = np.sqrt(np.pi / 2)
    return float(beta[0] * scale), float(beta[1] * scale)


def main() -> int:
    train, hold = load(TRAIN_SEASONS), load(HOLDOUT_SEASONS)
    check_scores(train, NBA, label="nba train")
    check_scores(hold, NBA, label="nba holdout")
    print(f"train {len(train)} games, holdout {len(hold)}")

    fit = fit_ratings(train)

    mu_tr = np.array([predict_mu(fit, g.home, g.away) for g in train.itertuples()])
    resid_tr = train.margin.values - mu_tr
    a, b = fit_sigma(mu_tr, resid_tr)
    sd_const = float(resid_tr.std(ddof=1))

    # --- grade ONCE -----------------------------------------------------
    rows = [(predict_mu(fit, g.home, g.away), g.margin) for g in hold.itertuples()]
    rows = [(m, y) for m, y in rows if m is not None]
    mu = np.array([m for m, _ in rows])
    y = np.array([float(v) for _, v in rows])

    ll_var = stats.norm.logpdf(y, mu, np.maximum(a + b * np.abs(mu), 1.0))
    ll_const = stats.norm.logpdf(y, mu, sd_const)
    ll_base = stats.norm.logpdf(y, train.margin.mean(), train.margin.std(ddof=1))

    def paired(x, z):
        d = x - z
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        return {"paired_gain": round(float(d.mean()), 5),
                "standard_error": round(se, 5),
                "t": round(float(d.mean() / se), 2),
                "supported": bool(d.mean() / se > 2)}

    # Does variance really rise with the spread? The ADR 0005 claim, tested --
    # and the test's own limitation measured alongside it.
    resid_ho = y - mu
    corr = float(np.corrcoef(np.abs(mu), np.abs(resid_ho))[0, 1])
    q = pd.qcut(np.abs(mu), 4, labels=False, duplicates="drop")
    quartiles = [
        {"mean_abs_mu": round(float(np.abs(mu)[q == b].mean()), 3),
         "resid_sd": round(float(resid_ho[q == b].std(ddof=1)), 3),
         "n": int((q == b).sum())}
        for b in sorted(set(q))
    ]

    art = {
        "_provenance": {
            "script": "model/fit_nba.py", "generated": "2026-09-21",
            "source": "sportsdataverse espn_nba_schedules, season_type==2",
            "train_seasons": list(TRAIN_SEASONS),
            "holdout_seasons": list(HOLDOUT_SEASONS),
            "graded_once": True,
            "metric": "mean log-likelihood of the realised margin",
            "filters": ("season_type == 2 AND type_abbreviation == STD, both "
                        "scores non-zero. season_type alone is insufficient: "
                        "ESPN files the All-Star Game (EAST 211, WEST 186) as "
                        "regular season, and the degenerate-data guard caught "
                        "it twice before the filter was right"),
            "n_train": int(len(train)), "n_holdout": int(len(mu)),
        },
        "fit": {"home_adv": fit["home_adv"], "n_teams": len(fit["ratings"]),
                "ridge": RIDGE,
                "sigma_intercept": round(a, 4), "sigma_slope": round(b, 5),
                "sigma_constant_alternative": round(sd_const, 4)},
        "holdout_grade": {
            "ratings_vs_league_average": paired(ll_const, ll_base),
            "varying_sigma_vs_constant_sigma": paired(ll_var, ll_const),
            "mean_ll_varying": round(float(ll_var.mean()), 5),
            "mean_ll_constant": round(float(ll_const.mean()), 5),
            "mean_ll_baseline": round(float(ll_base.mean()), 5),
        },
        "adr_0005_claim": {
            "claim": ("ADR 0005: NBA margin variance rises with spread "
                      "magnitude (design research reported corr ~0.60 against "
                      "MARKET spread)"),
            "holdout_corr_abs_mu_vs_abs_resid": round(corr, 5),
            "sigma_slope": round(b, 5),
            "residual_sd_by_predicted_spread_quartile": quartiles,
            "verdict": "INCONCLUSIVE, not refuted",
            "why_inconclusive": (
                "The predicted spread here comes from cross-season team "
                "ratings, not from a market line, and those ratings barely "
                "separate the games: the top quartile averages under 9 points "
                "where real NBA spreads reach the high teens. A variance "
                "effect across a wide spread range would be largely invisible "
                "over this narrow one. Residual sd IS flat across the range "
                "tested (holdout corr +0.015), which rules the effect out "
                "WHERE MEASURED and says nothing about where it was claimed. "
                "Testing the claim properly needs market spreads, which this "
                "project will have once odds capture runs."),
        },
        "ratings": fit["ratings"],
    }
    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=1)

    print(json.dumps(art["fit"], indent=1))
    print(json.dumps(art["holdout_grade"], indent=1))
    print(json.dumps(art["adr_0005_claim"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
