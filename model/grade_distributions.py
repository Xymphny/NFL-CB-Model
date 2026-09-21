#!/usr/bin/env python3
"""Is each league's shipped distribution actually the right shape?

THE ASYMMETRY THIS CLOSES
NHL and NBA were graded on the likelihood of realised scorelines against a
league-average baseline. NFL, CFB and MLB never were. What they have instead
is parity with the legacy pipeline, which says the rewrite reproduces the old
numbers and nothing about whether the old numbers were well specified, and --
for NFL -- an ATS validation, which asks whether the model beats a book.

Those are different questions. A model can be honestly negative against the
market and still be badly calibrated, and the calibration is what every stake
divides by. If MARGIN_SD is too small, every probability is overconfident and
every Kelly fraction is too large, whatever the edge sign turns out to be.
That failure is invisible to an ATS record.

WHAT IS IN SAMPLE HERE, SAID PLAINLY
The RATINGS in these caches were produced walk-forward, so the mean of each
prediction uses only prior games. The COEFFICIENTS and the dispersion
constants were fitted on the same games being scored. So this is in-sample for
everything except the ratings.

That makes it a ONE-SIDED test, and it is run for the side it can answer: a
distribution that is miscalibrated IN SAMPLE is miscalibrated, full stop, and
no out-of-sample test will rescue it. A distribution that looks calibrated
here has proved nothing, and this script says so rather than claiming a pass.

THE THREE THINGS MEASURED
  1. DISPERSION RATIO -- realised residual sd over the shipped sd. The direct
     test of the constant every probability divides by. One is right; above
     one is overconfident.
  2. COVERAGE -- how often the outcome lands inside the model's own 50, 80 and
     95 percent intervals. A ratio can hide a shape problem; coverage cannot.
  3. LIKELIHOOD against a league-average baseline, the same metric NHL and
     NBA were graded on, so the five leagues can be read on one scale.

Writes model/distribution_grades.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = HERE / "distribution_grades.json"

#: Nominal interval levels checked for coverage.
LEVELS = (0.50, 0.80, 0.95)


def _coverage(resid: np.ndarray, sd: float) -> dict:
    out = {}
    for lv in LEVELS:
        z = stats.norm.ppf(0.5 + lv / 2)
        out[f"{int(lv * 100)}"] = round(float((np.abs(resid) <= z * sd).mean()), 4)
    return out


def _grade(name: str, mu: np.ndarray, actual: np.ndarray, sd: float,
           note: str) -> dict:
    resid = actual - mu
    realised_sd = float(resid.std(ddof=1))
    ll_model = stats.norm.logpdf(actual, mu, sd)
    # Baseline: the league's own average margin and spread, which is the same
    # baseline NHL and NBA were graded against.
    ll_base = stats.norm.logpdf(actual, actual.mean(), actual.std(ddof=1))
    d = np.asarray(ll_model - ll_base)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {
        "n": int(len(actual)),
        "shipped_sd": round(float(sd), 4),
        "realised_residual_sd": round(realised_sd, 4),
        "dispersion_ratio": round(realised_sd / sd, 4),
        "mean_residual": round(float(resid.mean()), 4),
        "coverage": _coverage(resid, sd),
        "nominal_coverage": {str(int(lv * 100)): lv for lv in LEVELS},
        "mean_log_likelihood": round(float(ll_model.mean()), 5),
        "baseline_log_likelihood": round(float(ll_base.mean()), 5),
        "paired_gain": round(float(d.mean()), 5),
        "standard_error": round(se, 5),
        "t": round(float(d.mean() / se), 2),
        "in_sample_for": note,
    }


def _split_check(mu: np.ndarray, actual: np.ndarray, season: np.ndarray,
                 cut: int) -> dict:
    """Re-estimate the dispersion on early seasons, check it on later ones.

    WHY THIS IS NOT OPTIONAL. CFB's MARGIN_SD was measured as the residual sd
    of this exact cache, so the dispersion ratio above comes out at exactly
    1.0000 by construction. That is a tautology, not a pass, and reporting it
    beside NFL's 1.0337 would invite it to be read as the better number.

    Splitting breaks the circle: the sd is estimated on seasons before `cut`
    and judged on seasons from `cut` onwards, which it has never seen.
    """
    early, late = season < cut, season >= cut
    if early.sum() < 50 or late.sum() < 50:
        return {"available": False,
                "reason": f"too few games either side of {cut}"}
    sd_early = float((actual[early] - mu[early]).std(ddof=1))
    resid_late = actual[late] - mu[late]
    return {
        "available": True,
        "cut_season": cut,
        "sd_estimated_on": f"< {cut}",
        "n_estimate": int(early.sum()),
        "n_check": int(late.sum()),
        "sd_from_early_seasons": round(sd_early, 4),
        "realised_sd_on_later_seasons": round(float(resid_late.std(ddof=1)), 4),
        "dispersion_ratio": round(float(resid_late.std(ddof=1)) / sd_early, 4),
        "coverage": _coverage(resid_late, sd_early),
        "mean_residual": round(float(resid_late.mean()), 4),
    }


def _by_season(mu: np.ndarray, actual: np.ndarray,
               season: np.ndarray) -> dict:
    """Residual dispersion season by season.

    A single sd constant assumes the league holds still. This is the cheapest
    way to see whether it does, and it is what turns 'the pooled ratio is
    1.03' into something actionable or not.
    """
    out = {}
    for yr in sorted(set(int(x) for x in season)):
        m = season == yr
        if m.sum() < 30:
            continue
        r = actual[m] - mu[m]
        out[str(yr)] = {"n": int(m.sum()),
                        "residual_sd": round(float(r.std(ddof=1)), 4),
                        "mean_residual": round(float(r.mean()), 4)}
    return out


def _homogeneity(by_season: dict) -> dict:
    """Does the league hold still, or do these seasons only look different?

    Two questions, because they have different answers and different
    consequences. Bartlett asks whether the residual SPREAD is the same across
    seasons -- if it is not, a single sd constant is wrong in principle. The
    second asks whether the mean residual is the same -- a systematic BIAS is
    a different defect and is corrected differently.

    ONE TEST EACH, CHOSEN AFTER LOOKING AT THE NUMBERS. That is a garden of
    forking paths and the p-values should be read as suggestive rather than
    established, which is why both are reported with the raw season table
    beside them rather than as a verdict.
    """
    ns = np.array([v["n"] for v in by_season.values()], float)
    sds = np.array([v["residual_sd"] for v in by_season.values()], float)
    mus = np.array([v["mean_residual"] for v in by_season.values()], float)
    k = len(ns)
    if k < 2:
        return {"available": False}
    dfs = ns - 1
    pooled_var = float((dfs * sds ** 2).sum() / dfs.sum())
    stat = dfs.sum() * np.log(pooled_var) - (dfs * np.log(sds ** 2)).sum()
    corr = 1 + ((1 / dfs).sum() - 1 / dfs.sum()) / (3 * (k - 1))
    bartlett = float(stat / corr)

    se = np.sqrt(pooled_var / ns)
    grand = float(np.average(mus, weights=1 / se ** 2))
    q = float((((mus - grand) / se) ** 2).sum())

    return {
        "seasons": k,
        "dispersion_varies_by_season": {
            "test": "Bartlett", "chi2": round(bartlett, 2), "df": k - 1,
            "p": round(float(1 - stats.chi2.cdf(bartlett, k - 1)), 4),
            "sd_range": [round(float(sds.min()), 4), round(float(sds.max()), 4)],
            "sampling_se_of_one_seasons_sd": round(
                float(sds.mean() / np.sqrt(2 * ns.mean())), 3),
        },
        "bias_varies_by_season": {
            "test": "weighted chi-square on season mean residuals",
            "q": round(q, 2), "df": k - 1,
            "p": round(float(1 - stats.chi2.cdf(q, k - 1)), 5),
            "mean_residual_range": [round(float(mus.min()), 3),
                                    round(float(mus.max()), 3)],
            "pooled_mean_residual": round(grand, 4),
        },
    }


def grade_nfl() -> dict:
    from coverline.leagues.nfl.model import (
        MARGIN_COEFFICIENTS_V1_RATING_ONLY as C, MARGIN_SD,
    )

    d = pd.read_csv(HERE / "expanded_walk_forward_cache.csv")
    mu = (C["intercept"] + C["rating_diff"] * d.rating_diff
          + C["home_field"] * d.home_field + C["rest_diff"] * d.rest_diff)
    return _grade("nfl", mu.to_numpy(float), d.actual_margin.to_numpy(float),
                  MARGIN_SD,
                  "coefficients and MARGIN_SD; ratings are walk-forward") | {
        "split_check": _split_check(
            mu.to_numpy(float), d.actual_margin.to_numpy(float),
            d.season.to_numpy(int), cut=2022),
        "by_season": _by_season(mu.to_numpy(float),
                                d.actual_margin.to_numpy(float),
                                d.season.to_numpy(int)),
        "path": "V1_RATING_ONLY -- the cache carries no NGS columns, so this "
                "grades the no-NGS path, which is what a caller without NGS "
                "gets and is the one the board falls back to",
        "seasons": sorted(int(s) for s in d.season.unique()),
    }


def grade_cfb() -> dict:
    from coverline.leagues.cfb.model import (
        MARGIN_COEFFICIENTS_DVOA_ONLY as C, MARGIN_SD,
    )

    d = pd.read_csv(HERE / "cfb_full_walk_forward_cache.csv")
    mu = C["intercept"] + C["rating_diff"] * d.rating_diff
    return _grade("cfb", mu.to_numpy(float), d.actual_margin.to_numpy(float),
                  MARGIN_SD,
                  "coefficients and MARGIN_SD; ratings are walk-forward") | {
        "dispersion_ratio_is_circular": (
            "MARGIN_SD was MEASURED as the residual sd of this exact cache by "
            "model/cfb_margin_sd.py, so the ratio above is 1.0000 by "
            "construction and is not evidence of anything. See split_check."
        ),
        "split_check": _split_check(
            mu.to_numpy(float), d.actual_margin.to_numpy(float),
            d.season.to_numpy(int), cut=2023),
        "by_season": _by_season(mu.to_numpy(float),
                                d.actual_margin.to_numpy(float),
                                d.season.to_numpy(int)),
        "path": "DVOA_ONLY -- the cache carries no elo",
        "seasons": sorted(int(s) for s in d.season.unique()),
    }


def grade_mlb() -> dict:
    """MLB is counts, so the normal machinery above does not apply.

    Graded on the same likelihood scale as NHL: log P(realised scoreline)
    under the shipped negative binomials against league-average rates.
    """
    from coverline.leagues.mlb.model import R_AWAY, R_HOME
    from model.mlb_model import run_walk_forward

    sch = pd.read_csv(HERE / "mlb_schedule_cache.csv")
    pit = pd.read_csv(HERE / "mlb_pitching_cache.csv")
    w = run_walk_forward(sch, pit).dropna(
        subset=["home_score", "away_score", "exp_home", "exp_away"])
    w = w[(w.home_score >= 0) & (w.away_score >= 0)]

    def _nb(x, mu, r):
        p = r / (r + mu)
        return stats.nbinom.logpmf(x, r, p)

    hs = w.home_score.to_numpy(float)
    as_ = w.away_score.to_numpy(float)
    ll_model = (_nb(hs, w.exp_home.to_numpy(float), R_HOME)
                + _nb(as_, w.exp_away.to_numpy(float), R_AWAY))
    ll_base = (_nb(hs, hs.mean(), R_HOME) + _nb(as_, as_.mean(), R_AWAY))
    d = np.asarray(ll_model - ll_base)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))

    margin = hs - as_
    resid = margin - (w.exp_home.to_numpy(float) - w.exp_away.to_numpy(float))
    return {
        "n": int(len(w)),
        "family": "independent negative binomial",
        "r_home": R_HOME,
        "r_away": R_AWAY,
        "realised_margin_sd": round(float(margin.std(ddof=1)), 4),
        "realised_residual_sd": round(float(resid.std(ddof=1)), 4),
        "mean_residual": round(float(resid.mean()), 4),
        "mean_log_likelihood": round(float(ll_model.mean()), 5),
        "baseline_log_likelihood": round(float(ll_base.mean()), 5),
        "paired_gain": round(float(d.mean()), 5),
        "standard_error": round(se, 5),
        "t": round(float(d.mean() / se), 2),
        "in_sample_for": "r_home and r_away; expected runs are walk-forward",
        "seasons": sorted(int(s) for s in w.season.unique()),
    }


def main() -> int:
    grades = {
        "_provenance": {
            "script": "model/grade_distributions.py",
            "one_sided": ("in-sample for coefficients and dispersion "
                          "constants, walk-forward for ratings. A failure "
                          "here is decisive; a pass proves nothing."),
            "metrics": ["dispersion_ratio", "coverage", "likelihood vs "
                        "league-average baseline"],
        },
        "nfl": grade_nfl(),
        "cfb": grade_cfb(),
        "mlb": grade_mlb(),
    }
    for lg in ("nfl", "cfb"):
        grades[lg]["homogeneity"] = _homogeneity(grades[lg]["by_season"])
    OUT.write_text(json.dumps(grades, indent=2) + "\n")

    for lg in ("nfl", "cfb"):
        g = grades[lg]
        print(f"{lg.upper():4s} n={g['n']:5d}  shipped sd {g['shipped_sd']:7.4f}  "
              f"realised {g['realised_residual_sd']:7.4f}  "
              f"ratio {g['dispersion_ratio']:.4f}  "
              f"mean resid {g['mean_residual']:+.3f}  t={g['t']:+.2f}")
        print(f"      coverage {g['coverage']}  nominal 50/80/95")
        sc = g.get("split_check", {})
        if sc.get("available"):
            print(f"      split at {sc['cut_season']}: sd {sc['sd_from_early_seasons']:.4f} "
                  f"-> realised {sc['realised_sd_on_later_seasons']:.4f} "
                  f"ratio {sc['dispersion_ratio']:.4f}  coverage {sc['coverage']}")
        h = grades[lg]["homogeneity"]
        print(f"      dispersion varies by season: p={h['dispersion_varies_by_season']['p']}"
              f"   bias varies: p={h['bias_varies_by_season']['p']}")
    g = grades["mlb"]
    print(f"MLB  n={g['n']:5d}  residual sd {g['realised_residual_sd']:.4f}  "
          f"mean resid {g['mean_residual']:+.3f}  t={g['t']:+.2f}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
