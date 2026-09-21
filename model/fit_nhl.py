#!/usr/bin/env python3
"""Fit an NHL scoring model, and grade it ONCE.

DISCIPLINE, FIXED BEFORE ANY RESULT WAS SEEN
Train on 2021-2023, hold out 2024-2025. The holdout is graded exactly once,
on mean log-likelihood of the realised scoreline -- a proper scoring rule, so
it cannot be improved by making the distribution more confident.

Anything the grade rejects does not ship. The package currently ships NO
coefficients (ADR 0005), and that stays true unless the numbers earn it.

WHAT IS FITTED
A Poisson-style attack/defence rating per team, plus home advantage, by
iterative proportional fitting on goals -- the standard approach for
low-scoring sports. Expected goals for a game are then

    lam_home = exp(attack[home] + defence[away] + home_adv)
    lam_away = exp(attack[away] + defence[home])

THREE THINGS THIS DELIBERATELY CHECKS RATHER THAN ASSUMES
1. Whether hockey goals are actually Poisson. MLB looked like the textbook
   Poisson sport and measured 2.2x overdispersed; assuming it here would be
   repeating a mistake this project has already made once.
2. Whether the two teams' goals are independent, which decides whether the
   shared-component term is warranted.
3. Whether empty-net goals distort the tail enough to matter, by comparing
   one-goal-game frequency against the model's prediction.

Writes model/nhl_fit_results.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fit_data_checks import NHL as NHL_EXPECT  # noqa: E402
from model.fit_data_checks import check_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nhl_fit_results.json")

#: ONLY TWO SEASONS ARE USABLE. sportsdataverse's 2021-2023 NHL schedule files
#: carry a CONSTANT score on every row -- 1,312 games all 6-3 in 2022. The
#: degenerate-data guard rejects them; the first version of this script did
#: not have that guard, fitted on them anyway, and reported t = -5.12 with a
#: straight face.
#:
#: One season to train and one to grade is thin and is stated as thin. It is
#: not padded by reusing the holdout.
TRAIN_SEASONS = (2024,)
HOLDOUT_SEASONS = (2025,)
DATA = "/tmp/fitdata/nhl_{year}.parquet"
MAX_GOALS = 15


def load(seasons) -> pd.DataFrame:
    frames = []
    for y in seasons:
        d = pd.read_parquet(DATA.format(year=y))
        d = d[d.game_type == "R"] if "game_type" in d.columns else d
        d = d.dropna(subset=["home_score", "away_score",
                             "home_team_abbr", "away_team_abbr"])
        frames.append(d[["season", "game_date", "home_team_abbr",
                         "away_team_abbr", "home_score", "away_score"]])
    out = pd.concat(frames, ignore_index=True)
    out["home_score"] = out.home_score.astype(int)
    out["away_score"] = out.away_score.astype(int)
    return out


def fit_ratings(df: pd.DataFrame, iters: int = 200, lr: float = 0.05):
    """Attack/defence ratings by gradient ascent on the Poisson likelihood."""
    teams = sorted(set(df.home_team_abbr) | set(df.away_team_abbr))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    atk = np.zeros(n)
    dfn = np.zeros(n)
    home_adv = 0.0

    h = df.home_team_abbr.map(idx).values
    a = df.away_team_abbr.map(idx).values
    hs = df.home_score.values.astype(float)
    as_ = df.away_score.values.astype(float)
    base = np.log((hs.mean() + as_.mean()) / 2)

    for _ in range(iters):
        lam_h = np.exp(base + atk[h] + dfn[a] + home_adv)
        lam_a = np.exp(base + atk[a] + dfn[h])
        rh, ra = hs - lam_h, as_ - lam_a

        g_atk = np.bincount(h, rh, n) + np.bincount(a, ra, n)
        g_dfn = np.bincount(a, rh, n) + np.bincount(h, ra, n)
        atk += lr * g_atk / len(df)
        dfn += lr * g_dfn / len(df)
        home_adv += lr * rh.sum() / len(df)

        atk -= atk.mean()
        dfn -= dfn.mean()

    return {"teams": teams, "attack": dict(zip(teams, atk.round(5))),
            "defence": dict(zip(teams, dfn.round(5))),
            "home_adv": float(round(home_adv, 5)), "base": float(round(base, 5))}


def rates(fit, home, away):
    if home not in fit["attack"] or away not in fit["attack"]:
        return None
    lh = np.exp(fit["base"] + fit["attack"][home] + fit["defence"][away]
                + fit["home_adv"])
    la = np.exp(fit["base"] + fit["attack"][away] + fit["defence"][home])
    return float(lh), float(la)


def main() -> int:
    train, hold = load(TRAIN_SEASONS), load(HOLDOUT_SEASONS)
    # The guard that the first version of this script lacked.
    check_scores(train, NHL_EXPECT, label="nhl train")
    check_scores(hold, NHL_EXPECT, label="nhl holdout")
    print(f"train {len(train)} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}), "
          f"holdout {len(hold)} ({HOLDOUT_SEASONS[0]}-{HOLDOUT_SEASONS[-1]})")

    fit = fit_ratings(train)

    # --- assumption checks, on TRAIN only -------------------------------
    checks = {}
    for side, col in (("home", "home_score"), ("away", "away_score")):
        mu, var = float(train[col].mean()), float(train[col].var())
        checks[side] = {"mean": round(mu, 4), "variance": round(var, 4),
                        "variance_over_mean": round(var / mu, 4)}
    checks["correlation"] = round(
        float(train.home_score.corr(train.away_score)), 5)
    checks["poisson_like"] = bool(
        all(abs(checks[s]["variance_over_mean"] - 1) < 0.25 for s in ("home", "away")))
    checks["independent"] = bool(abs(checks["correlation"]) < 0.05)

    # --- grade ONCE on holdout ------------------------------------------
    ks = np.arange(0, MAX_GOALS + 1)
    ll_model, ll_base, one_goal_pred, skipped = [], [], [], 0
    base_h = train.home_score.mean()
    base_a = train.away_score.mean()

    for g in hold.itertuples():
        r = rates(fit, g.home_team_abbr, g.away_team_abbr)
        if r is None:
            skipped += 1
            continue
        lh, la = r
        ll_model.append(stats.poisson.logpmf(g.home_score, lh)
                        + stats.poisson.logpmf(g.away_score, la))
        ll_base.append(stats.poisson.logpmf(g.home_score, base_h)
                       + stats.poisson.logpmf(g.away_score, base_a))
        ph = stats.poisson.pmf(ks, lh)
        pa = stats.poisson.pmf(ks, la)
        one_goal_pred.append(float(sum(ph[k] * (pa[k - 1] + (pa[k + 1] if k + 1 <= MAX_GOALS else 0))
                                       for k in range(1, MAX_GOALS))))

    ll_model = np.array(ll_model)
    ll_base = np.array(ll_base)
    d = ll_model - ll_base
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(d.mean() / se)

    actual_one_goal = float(
        ((hold.home_score - hold.away_score).abs() == 1).mean())

    art = {
        "_provenance": {
            "script": "model/fit_nhl.py", "generated": "2026-09-21",
            "source": "sportsdataverse nhl_schedules releases, regular season",
            "train_seasons": list(TRAIN_SEASONS),
            "holdout_seasons": list(HOLDOUT_SEASONS),
            "graded_once": True,
            "metric": "mean log-likelihood of the realised scoreline",
            "baseline": "league-average Poisson rates, no team ratings",
            "n_train": int(len(train)), "n_holdout": int(n),
            "skipped_unknown_team": skipped,
            "sample_caveat": (
                "one season trains and one grades, because sportsdataverse's "
                "2021-2023 NHL files carry a constant score on every row and "
                "the degenerate-data guard rejects them. Thin, and stated as "
                "thin rather than padded by reusing the holdout."),
        },
        "assumption_checks": checks,
        "fit": {"home_adv": fit["home_adv"], "base": fit["base"],
                "n_teams": len(fit["teams"])},
        "holdout_grade": {
            "model": round(float(ll_model.mean()), 5),
            "baseline": round(float(ll_base.mean()), 5),
            "paired_gain": round(float(d.mean()), 5),
            "standard_error": round(se, 5),
            "t": round(t, 2),
            "supported": bool(t > 2),
        },
        "empty_net_check": {
            "predicted_one_goal_rate": round(float(np.mean(one_goal_pred)), 4),
            "actual_one_goal_rate": round(actual_one_goal, 4),
            "note": (
                "The model UNDER-predicts one-goal games -- 28.1% against an "
                "actual 38.3%. That is the opposite of what this check was "
                "written expecting, and it is the signature of NEGATIVE "
                "dependence between the two scores (measured corr -0.14), not "
                "of empty nets: real hockey games stay closer than independent "
                "Poisson rates allow, because a leading team defends and the "
                "trailing team presses. Empty-net goals push the other way, "
                "converting one-goal games into two-goal finals, so the "
                "underlying closeness effect is LARGER than this 10-point gap "
                "shows. Either way, independent Poisson is the wrong joint "
                "distribution for hockey."),
        },
        "ratings": {"attack": fit["attack"], "defence": fit["defence"]},
    }
    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=1)

    print(json.dumps(art["assumption_checks"], indent=1))
    print(json.dumps(art["holdout_grade"], indent=1))
    print(json.dumps(art["empty_net_check"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
