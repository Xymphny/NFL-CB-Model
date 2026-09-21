#!/usr/bin/env python3
"""Calibration for distributions over counts, where intervals do not work.

WHY NOT THE COVERAGE CHECK ALREADY IN model/grade_distributions.py
That one asks how often the outcome lands inside a 50, 80 or 95 percent
interval. It assumes a continuous, symmetric distribution -- the 95% interval
of a Skellam is not "mu plus or minus 1.96 sd" in any useful sense -- and ADR
0009 recorded MLB as graded on likelihood but NOT on coverage for exactly that
reason. This is the count equivalent it named.

WHAT THE PIT IS, AND WHY THE NON-RANDOMISED ONE
If a forecast distribution F is right, then F(Y) is uniform on [0, 1]. For
discrete Y that fails: F(Y) can only take the values F takes, so the histogram
is lumpy even for a perfect forecaster. The usual fix is to randomise inside
the atom, which makes the answer depend on a seed.

Czado, Gneiting and Held's non-randomised version averages the conditional
uniform over each atom instead:

    Fbar(u) = 0                                  u <= F(y-1)
              (u - F(y-1)) / (F(y) - F(y-1))     F(y-1) < u < F(y)
              1                                  u >= F(y)

A calibrated forecaster has mean Fbar(u) equal to u for every u. The shape of
the departure says what is wrong: a hump in the middle means the forecasts are
too WIDE, a U means too NARROW, and a slope means biased.

WHAT IS GRADED HERE AND WHAT IS NOT
NHL is checked on 2022-2023 -- the same holdout the rules layer was graded on,
with the same parameters, so this is a second QUESTION about one grade rather
than a second grade. It spends nothing extra and it can only embarrass the
layer, never promote it: a likelihood win with a bent PIT is a model that is
better than the baseline and still the wrong shape.

MLB is in sample, like everything in model/grade_distributions.py, and says so.

Writes model/calibration_pit.json.
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

OUT = HERE / "calibration_pit.json"

#: Grid on [0, 1]. Ten is the conventional histogram; the statistic below is
#: computed on a finer grid because a Kolmogorov-type distance on ten points
#: understates.
BINS = 10
GRID = np.linspace(0.0, 1.0, 201)


def pit(cdf_at_y: np.ndarray, cdf_below_y: np.ndarray) -> dict:
    """Non-randomised PIT summary for one set of forecasts.

    Takes F(y) and F(y-1) per observation, which is all the method needs and
    is the only thing every distribution in this repository can answer
    cheaply.
    """
    lo = np.clip(cdf_below_y, 0.0, 1.0)
    hi = np.clip(cdf_at_y, 0.0, 1.0)
    width = np.maximum(hi - lo, 1e-12)

    # Fbar(u), averaged over observations, on the fine grid.
    u = GRID[None, :]
    frac = np.clip((u - lo[:, None]) / width[:, None], 0.0, 1.0)
    fbar = frac.mean(axis=0)

    # Histogram heights: the mass the forecaster put in each tenth.
    edges = np.linspace(0.0, 1.0, BINS + 1)
    heights = []
    for a, b in zip(edges[:-1], edges[1:]):
        fa = np.clip((a - lo) / width, 0.0, 1.0)
        fb = np.clip((b - lo) / width, 0.0, 1.0)
        heights.append(float((fb - fa).mean() * BINS))

    ks = float(np.abs(fbar - GRID).max())
    n = len(lo)
    return {
        "n": int(n),
        "max_deviation": round(ks, 4),
        # Asymptotic Kolmogorov scale. Reported as a yardstick, not a p-value:
        # the observations are not independent draws from one F and the
        # forecasts share estimated parameters, so this is optimistic.
        "kolmogorov_scale_at_n": round(float(1.36 / np.sqrt(n)), 4),
        "exceeds_kolmogorov_scale": bool(ks > 1.36 / np.sqrt(n)),
        "histogram": [round(h, 4) for h in heights],
        "histogram_note": ("heights are relative to 1.0. A hump in the middle "
                           "means the forecasts are too WIDE, a U means too "
                           "NARROW, a slope means biased."),
        "mean_pit": round(float(((lo + hi) / 2).mean()), 4),
    }


def _discrete_pit_from_pmf(pmf_rows: list[np.ndarray], support_lo: int,
                           observed: np.ndarray) -> dict:
    """PIT from an explicit pmf vector per observation."""
    lo = np.empty(len(observed))
    hi = np.empty(len(observed))
    for i, (p, y) in enumerate(zip(pmf_rows, observed)):
        k = int(y) - support_lo
        c = np.cumsum(p)
        hi[i] = c[k] if 0 <= k < len(c) else (0.0 if k < 0 else 1.0)
        lo[i] = c[k - 1] if k - 1 >= 0 else 0.0
    return pit(hi, lo)


def nhl() -> dict:
    """Baseline against the rules layer, on the layer's own holdout."""
    from coverline.leagues.nhl.rules import NHLFinalScoreDistribution
    from model.export_fitted import export_nhl_rules  # noqa: F401  (kept in sync)
    from model.fit_nhl_rules import (
        MAX_LEAD, REFRESH_HOLDOUT, REFRESH_TUNE, load, measure_overtime,
        measure_pull_table, rate,
    )

    # THE SHIPPED CONFIGURATION, which is the refreshed table graded on
    # 2024-2025 -- not the first one. Checking the calibration of a layer that
    # is no longer in data/nhl_rules.json would be checking nothing.
    TUNE, HOLDOUT = REFRESH_TUNE, REFRESH_HOLDOUT
    from coverline.leagues.nhl.rules import GoaliePullLayer, OvertimeLayer

    tune, hold = load(TUNE), load(HOLDOUT)
    pull = GoaliePullLayer(table=measure_pull_table(tune, max_lead=MAX_LEAD),
                           max_lead=MAX_LEAD)
    overtime = OvertimeLayer(home_win_prob=measure_overtime(tune))

    finals = hold.sort_values(["season", "game_date"]).reset_index(drop=True)
    margin = (finals.home_score - finals.away_score).to_numpy(int)
    total = (finals.home_score + finals.away_score).to_numpy(int)

    base = rate(hold, "home_score", "away_score")
    rules = rate(hold, "nopull_h", "nopull_a")

    span = 40
    base_margin, base_total = [], []
    for r in base.itertuples():
        ks = np.arange(0, span)
        ph = stats.poisson.pmf(ks, r.lam_h)
        pa = stats.poisson.pmf(ks, r.lam_a)
        j = np.outer(ph, pa)
        m = np.add.outer(ks, -ks)
        base_margin.append(np.array(
            [j[m == d].sum() for d in range(-span + 1, span)]))
        t = np.add.outer(ks, ks)
        base_total.append(np.array([j[t == s].sum() for s in range(0, 2 * span - 1)]))

    rules_margin, rules_total = [], []
    for r in rules.itertuples():
        d = NHLFinalScoreDistribution(r.lam_h, r.lam_a, pull, overtime)
        rules_margin.append(np.array(
            [d.margin_pmf(x) for x in range(-span + 1, span)]))
        rules_total.append(np.array([d.total_pmf(x) for x in range(0, 2 * span - 1)]))

    return {
        "holdout_seasons": list(HOLDOUT),
        "spends_nothing_extra": (
            "the same holdout, the same parameters, a different question "
            "about one grade. It can embarrass the layer and cannot promote "
            "it."
        ),
        "baseline_margin": _discrete_pit_from_pmf(base_margin, -span + 1, margin),
        "rules_margin": _discrete_pit_from_pmf(rules_margin, -span + 1, margin),
        "baseline_total": _discrete_pit_from_pmf(base_total, 0, total),
        "rules_total": _discrete_pit_from_pmf(rules_total, 0, total),
    }


def mlb() -> dict:
    """In sample, like every MLB number in model/grade_distributions.py."""
    from coverline.leagues.mlb.model import R_AWAY, R_HOME
    from model.mlb_model import run_walk_forward

    sch = pd.read_csv(HERE / "mlb_schedule_cache.csv")
    pit_df = pd.read_csv(HERE / "mlb_pitching_cache.csv")
    w = run_walk_forward(sch, pit_df).dropna(
        subset=["home_score", "away_score", "exp_home", "exp_away"])
    w = w[(w.home_score >= 0) & (w.away_score >= 0)]

    def _cdf(x, mu, r):
        return stats.nbinom.cdf(x, r, r / (r + mu))

    out = {"n": int(len(w)), "in_sample_for": "r_home and r_away"}
    for side, col, r in (("home", "home_score", R_HOME),
                         ("away", "away_score", R_AWAY)):
        mu = w[f"exp_{side}"].to_numpy(float)
        y = w[col].to_numpy(float)
        out[side] = pit(_cdf(y, mu, r), np.where(y > 0, _cdf(y - 1, mu, r), 0.0))
    return out


def main() -> int:
    report = {
        "_provenance": {
            "script": "model/calibration_pit.py",
            "method": "non-randomised PIT (Czado, Gneiting and Held)",
            "why": ("interval coverage assumes a continuous symmetric "
                    "distribution and does not apply to counts; ADR 0009 "
                    "named this as the missing piece"),
        },
        "nhl": nhl(),
        "mlb": mlb(),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    for league, keys in (("nhl", ("baseline_margin", "rules_margin",
                                  "baseline_total", "rules_total")),
                         ("mlb", ("home", "away"))):
        for k in keys:
            d = report[league][k]
            flag = "  <-- exceeds" if d["exceeds_kolmogorov_scale"] else ""
            print(f"{league} {k:16s} n={d['n']:5d}  max dev {d['max_deviation']:.4f}"
                  f"  scale {d['kolmogorov_scale_at_n']:.4f}{flag}")
            print(f"     histogram {d['histogram']}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
