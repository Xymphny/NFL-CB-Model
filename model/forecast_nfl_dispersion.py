#!/usr/bin/env python3
"""Can next season's NFL dispersion be forecast? Declared before it was run.

THE OPEN ITEM
ADR 0009 found that NFL residual dispersion varies across seasons -- 11.73 to
15.15, Bartlett p = 0.014 -- and did NOT change MARGIN_SD, on the grounds that
"knowing the value moves is not the same as being able to forecast it. A
season-varying sd has to predict next season's dispersion and be graded on
that, and no such forecast exists."

This is that forecast, and the point of the exercise is that it is allowed to
fail. If it does, the shipped constant becomes the right choice for a MEASURED
reason instead of an unexamined one.

WHY IT PROBABLY FAILS, WORKED OUT BEFORE GRADING
Variance that MOVES is not variance that is PREDICTABLE. Decomposing the
between-season spread: the season sds vary with a standard deviation of 1.048,
and sampling alone on ~195 games a season accounts for 0.695 of that, leaving a
real signal of about 0.78. So the variation is real -- which is what Bartlett
detected -- and the lag-1 autocorrelation of the season sd is -0.32 on nine
pairs, against a standard error near 0.33. Negative and indistinguishable from
zero. A quantity whose own history carries no sign of itself cannot be
forecast from that history.

CANDIDATES, FIXED A PRIORI, NO SEARCH
  constant        the shipped 13.2979 -- the incumbent
  last_season     last season's realised sd
  expanding       the mean of every prior season's sd
  shrunk_half     halfway between the two, at a lambda fixed at 0.5

Four candidates, zero tuned parameters, so there is no search to inflate and
no noise ceiling to clear.

HOLDOUT, FIXED BEFORE RUNNING
  train    2014-2019   six seasons, used only to seed the estimators
  holdout  2020-2023   four seasons, graded ONCE

Writes model/nfl_dispersion_forecast.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coverline.leagues.nfl.model import (  # noqa: E402
    MARGIN_COEFFICIENTS_V1_RATING_ONLY as C, MARGIN_SD,
)

OUT = HERE / "nfl_dispersion_forecast.json"
TRAIN = tuple(range(2014, 2020))
HOLDOUT = (2020, 2021, 2022, 2023)
SHRINK_LAMBDA = 0.5


def residuals() -> pd.DataFrame:
    d = pd.read_csv(HERE / "expanded_walk_forward_cache.csv")
    mu = (C["intercept"] + C["rating_diff"] * d.rating_diff
          + C["home_field"] * d.home_field + C["rest_diff"] * d.rest_diff)
    return d.assign(mu=mu, resid=d.actual_margin - mu)


def main() -> int:
    d = residuals()
    by = d.groupby("season").resid.agg(n="size",
                                       sd=lambda x: float(x.std(ddof=1)))
    sd_of = by.sd.to_dict()

    rows = []
    for season in HOLDOUT:
        prior = [sd_of[s] for s in sorted(sd_of) if s < season]
        expanding = float(np.mean(prior))
        last = float(prior[-1])
        estimates = {
            "constant": MARGIN_SD,
            "last_season": last,
            "expanding": expanding,
            "shrunk_half": expanding + SHRINK_LAMBDA * (last - expanding),
        }
        sub = d[d.season == season]
        ll = {name: float(stats.norm.logpdf(sub.actual_margin, sub.mu, s).mean())
              for name, s in estimates.items()}
        rows.append({"season": int(season), "realised_sd": round(sd_of[season], 4),
                     "estimates": {k: round(v, 4) for k, v in estimates.items()},
                     "mean_log_likelihood": {k: round(v, 5) for k, v in ll.items()}})

    # Paired against the incumbent, game by game across the whole holdout.
    hold = d[d.season.isin(HOLDOUT)].copy()
    est_by_season = {r["season"]: r["estimates"] for r in rows}
    base = stats.norm.logpdf(hold.actual_margin, hold.mu, MARGIN_SD)
    grades = {}
    for name in ("last_season", "expanding", "shrunk_half"):
        s_hat = hold.season.map(lambda x: est_by_season[x][name]).to_numpy(float)
        alt = stats.norm.logpdf(hold.actual_margin, hold.mu, s_hat)
        diff = np.asarray(alt - base)
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        grades[name] = {
            "paired_gain_vs_constant": round(float(diff.mean()), 6),
            "standard_error": round(se, 6),
            "t": round(float(diff.mean() / se), 2),
            "beats_the_constant": bool(diff.mean() / se > 2),
        }

    s = by.sd.to_numpy()
    n = by.n.to_numpy()
    sampling = float(s.mean() / np.sqrt(2 * n.mean()))
    report = {
        "_provenance": {
            "script": "model/forecast_nfl_dispersion.py",
            "train_seasons": list(TRAIN), "holdout_seasons": list(HOLDOUT),
            "graded_once": True, "hyperparameter_trials": 0,
            "candidates_fixed_a_priori": True,
            "shrink_lambda": SHRINK_LAMBDA,
            "metric": "mean log-likelihood of the realised margin",
            "incumbent": f"MARGIN_SD = {MARGIN_SD}",
            "n_holdout": int(len(hold)),
        },
        "why_it_was_expected_to_fail": {
            "sd_of_season_sds": round(float(s.std(ddof=1)), 4),
            "sampling_se_of_one_season": round(sampling, 4),
            "implied_real_signal_sd": round(
                float(np.sqrt(max(s.var(ddof=1) - sampling ** 2, 0.0))), 4),
            "lag1_autocorrelation": round(
                float(np.corrcoef(s[:-1], s[1:])[0, 1]), 4),
            "autocorrelation_standard_error": round(1 / np.sqrt(len(s) - 1), 4),
            "reading": ("the variation is real -- Bartlett detected it -- and "
                        "its own history carries no sign of it. A quantity "
                        "like that cannot be forecast from that history."),
        },
        "by_season": rows,
        "holdout_grades": grades,
        "verdict": None,
    }
    winner = [k for k, v in grades.items() if v["beats_the_constant"]]
    report["verdict"] = (
        f"no forecast beats the constant; MARGIN_SD stays at {MARGIN_SD}"
        if not winner else
        f"beaten by {sorted(winner)}, which is a model change and needs its "
        "own record"
    )
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"lag-1 autocorrelation of season sd "
          f"{report['why_it_was_expected_to_fail']['lag1_autocorrelation']:+.4f} "
          f"(SE {report['why_it_was_expected_to_fail']['autocorrelation_standard_error']:.4f})")
    for name, g in grades.items():
        print(f"  {name:14s} gain vs constant {g['paired_gain_vs_constant']:+.6f}"
              f"  t {g['t']:+.2f}  beats: {g['beats_the_constant']}")
    print(report["verdict"])
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
