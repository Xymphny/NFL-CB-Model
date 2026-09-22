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
    # 76 rows carry an impossible tied margin -- see the note on MARGIN_SD.
    # Grading calibration through them measured a distribution against 4.39%
    # of outcomes that cannot occur.
    from model.fit_data_checks import check_no_impossible_ties
    d = d[d.actual_margin != 0].reset_index(drop=True)
    check_no_impossible_ties(d, "cfb", label="cfb grading frame",
                             margin="actual_margin")
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


def grade_nba() -> dict:
    """NBA on the same scale as the rest, and the sigma question re-opened.

    HOLDOUT ACCOUNTING. 2021-2022 tuned the shipped hyperparameters and 2023
    graded them once; 2024-2025 were spent earlier on the static question.
    Everything here is therefore either in sample or already graded, and this
    asks a DIFFERENT question of the same grade -- is the distribution the
    right shape -- which can embarrass it and cannot promote it. The 2023 row
    is the one to read.

    THE SIGMA QUESTION ADR 0005 LEFT OPEN. That record claimed NBA margin
    variance correlates about 0.60 with spread magnitude, which would make a
    constant sigma wrong. ADR 0006 recorded the claim as INCONCLUSIVE rather
    than confirmed: the 0.60 was measured against MARKET spreads, and these
    ratings separate games across a much narrower range, so the effect is
    ruled out where it can be measured and untested where it was claimed.
    This computes the by-bucket dispersion so the range is visible rather
    than asserted.
    """
    from coverline.leagues.nba.model import load_fitted
    from model import fit_nba_walkforward as nbawf
    from model.fit_nba import load as load_nba

    art = load_fitted()
    hp = art["hyperparameters"]
    sd = float(art["sigma_constant"])
    tune = list(art["_provenance"]["tune_seasons"])
    hold = int(art["_provenance"]["holdout_season"])

    df = load_nba(tuple(tune) + (hold,))
    pred = nbawf.walk_forward(df, hp["k"], hp["home_adv"], hp["carryover"])
    pred = pred.assign(resid=pred.margin - pred.mu)

    def _block(sub: pd.DataFrame, label: str) -> dict:
        mu = sub.mu.to_numpy(float)
        act = sub.margin.to_numpy(float)
        g = _grade(label, mu, act, sd, "sigma_constant and hyperparameters")
        # Dispersion by predicted-spread magnitude. Quartiles rather than
        # fixed cut points, because the interesting number is the RANGE the
        # ratings actually span.
        q = np.quantile(np.abs(mu), [0.25, 0.5, 0.75])
        buckets = {}
        edges = [0.0, *q, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (np.abs(mu) >= lo) & (np.abs(mu) < hi)
            if m.sum() < 30:
                continue
            buckets[f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(m.sum()),
                "mean_abs_predicted_spread": round(float(np.abs(mu[m]).mean()), 3),
                "residual_sd": round(float((act[m] - mu[m]).std(ddof=1)), 4),
            }
        g["dispersion_by_predicted_spread"] = buckets
        g["predicted_spread_range"] = [round(float(np.abs(mu).min()), 3),
                                       round(float(np.abs(mu).max()), 3)]
        g["correlation_abs_spread_with_abs_residual"] = round(
            float(np.corrcoef(np.abs(mu), np.abs(act - mu))[0, 1]), 4)
        return g

    # HOW BIG A BUCKETED CORRELATION DOES CONSTANT VARIANCE PRODUCE?
    #
    # ADR 0005 cited "about 0.60" between margin variance and spread
    # magnitude. That is a correlation over a handful of BUCKET means, and a
    # handful of points correlate strongly by accident. This simulates
    # constant-variance noise through the same bucketing and reports what it
    # gives, which is the only way to know whether 0.60 was ever evidence.
    rng = np.random.default_rng(0)
    a = np.abs(pred.mu.to_numpy(float))
    r = pred.resid.to_numpy(float)
    pooled_sd = float(r.std(ddof=1))
    bucket_noise = {}
    for nb in (5, 8, 10, 20):
        q = np.quantile(a, np.linspace(0, 1, nb + 1))
        masks = [(a >= q[i]) & (a < (q[i + 1] if i < nb - 1 else np.inf))
                 for i in range(nb)]
        masks = [m for m in masks if m.sum() >= 20]
        xs = np.array([a[m].mean() for m in masks])
        observed = float(np.corrcoef(
            xs, [r[m].std(ddof=1) for m in masks])[0, 1])
        sims = []
        for _ in range(2000):
            noise = rng.normal(0.0, pooled_sd, size=len(a))
            sims.append(abs(float(np.corrcoef(
                xs, [noise[m].std(ddof=1) for m in masks])[0, 1])))
        bucket_noise[str(nb)] = {
            "observed": round(observed, 3),
            "median_abs_from_constant_variance": round(float(np.median(sims)), 3),
            "p90_abs_from_constant_variance": round(
                float(np.quantile(sims, 0.9)), 3),
        }

    out = {
        "graded_holdout_season": hold,
        "tune_seasons": tune,
        "bucketed_correlation_noise_ceiling": bucket_noise,
        "per_game_correlation": {
            "abs_spread_with_abs_residual": round(
                float(np.corrcoef(a, np.abs(r))[0, 1]), 4),
            "abs_spread_with_squared_residual": round(
                float(np.corrcoef(a, r ** 2)[0, 1]), 4),
            "n": int(len(a)),
            "predicted_spread_range": [round(float(a.min()), 3),
                                       round(float(a.max()), 3)],
            "note": ("the per-game correlation is the one carrying "
                     "information; it uses every game instead of a handful "
                     "of bucket means"),
        },
        "holdout": _block(pred[pred.season == hold], "nba-holdout"),
        "tune_in_sample": _block(pred[pred.season.isin(tune)], "nba-tune"),
        "sigma_question": (
            "ADR 0005 claimed margin variance correlates about 0.60 with "
            "spread magnitude, measured against MARKET spreads. These ratings "
            "do not span that range, so the by-bucket table below rules the "
            "effect out where it can be measured and leaves it untested where "
            "it was claimed. Market spreads are what settles it."
        ),
    }
    return out


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


def grade_totals() -> dict:
    """The three sd_total constants nobody had checked, and two are wrong.

    All three leagues withhold totals from primary_markets and all three were
    still answering total_mean() with a placeholder until the distribution was
    taught to refuse. This measures how much that would have cost.

    CFB is the exact case: its mu_total is a CONSTANT, so the residual and the
    unconditional dispersion are the same quantity and the comparison needs no
    model. NFL's comes from its own withholding artifact. NBA's cannot be done
    here at all, because the residual needs the model's predicted totals and
    the feature source that produces them is not in this repository -- the
    unconditional sd bounds nothing, and saying so is the honest output.
    """
    out = {}

    cfb = pd.read_csv(HERE / "cfb_schedule_cache.csv").dropna(
        subset=["home_score", "away_score"])
    tot = (cfb.home_score + cfb.away_score).to_numpy(float)
    from coverline.leagues.cfb.model import (
        TOTAL_MEAN_PLACEHOLDER, TOTAL_SD_UNVALIDATED,
    )
    resid = tot - TOTAL_MEAN_PLACEHOLDER
    out["cfb"] = {
        "n": int(len(tot)),
        "seasons": sorted(int(s) for s in cfb.season.unique()),
        "shipped_mu": TOTAL_MEAN_PLACEHOLDER,
        "actual_mean": round(float(tot.mean()), 4),
        "mean_bias": round(float(resid.mean()), 4),
        "mean_bias_t": round(
            float(resid.mean() / (tot.std(ddof=1) / np.sqrt(len(tot)))), 2),
        "shipped_sd": TOTAL_SD_UNVALIDATED,
        "actual_sd": round(float(tot.std(ddof=1)), 4),
        "dispersion_ratio": round(float(tot.std(ddof=1)) / TOTAL_SD_UNVALIDATED, 4),
        "coverage": _coverage(resid, TOTAL_SD_UNVALIDATED),
        "exact_because": ("mu_total is a constant, so residual and "
                          "unconditional dispersion are the same number"),
        "verdict": ("the number labelled PLACEHOLDER is accurate; the one "
                    "labelled UNVALIDATED understates dispersion by a third"),
    }

    val = json.loads((ROOT / "data" / "totals_validation.json").read_text())
    from coverline.leagues.nfl.model import TOTAL_SD_UNVALIDATED as NFL_SD
    rmse = float(val["accuracy"]["model"]["rmse"])
    out["nfl"] = {
        "shipped_sd": NFL_SD,
        "model_rmse": rmse,
        "actual_total_sd": float(val["accuracy"]["actual_sd"]),
        "dispersion_ratio": round(rmse / NFL_SD, 4),
        "source": "data/totals_validation.json -- the artifact that withheld "
                  "this market records the number that contradicts its sd",
        "verdict": "understates residual dispersion by a third",
    }

    from model.fit_nba import load as load_nba
    nba = load_nba((2021, 2022, 2023, 2024, 2025))
    ntot = (nba.home_score + nba.away_score).to_numpy(float)
    out["nba"] = {
        "n": int(len(ntot)),
        "shipped_sd": 18.0,
        "unconditional_sd": round(float(ntot.std(ddof=1)), 4),
        "residual_sd": None,
        "not_measurable_here": (
            "sd_total should be the residual around the model's predicted "
            "total, and the feature source that produces those totals is not "
            "in this repository. The unconditional sd bounds a residual only "
            "if the model has skill, which is the thing in question. "
            "Unmeasured, not passing."
        ),
    }
    return out


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
        "nba": grade_nba(),
        "totals": grade_totals(),
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
    t = grades["totals"]
    print(f"TOTALS  cfb sd {t['cfb']['shipped_sd']} vs {t['cfb']['actual_sd']} "
          f"(ratio {t['cfb']['dispersion_ratio']}, 95% covers "
          f"{t['cfb']['coverage']['95']})")
    print(f"        nfl sd {t['nfl']['shipped_sd']} vs rmse {t['nfl']['model_rmse']} "
          f"(ratio {t['nfl']['dispersion_ratio']})")
    print(f"        nba sd {t['nba']['shipped_sd']} -- not measurable here")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
