#!/usr/bin/env python3
"""The grid that was never committed, committed -- without selecting anything.

WHAT THIS CLOSES
migration/ledger.yaml's frozen-threshold-grid row has been open since the
rebuild began: shipped constants in model/player_projection.py cite a grid
search whose output exists in no committed file and whose SPECIFICATION is
also gone. That file's own honesty note names the remedy -- "treat the
constants as asserted-not-shown until a calibrate_player_td_lambda.py exists"
-- and this is that file.

WHAT IT REFUSES TO DO
Select. Not because selecting is wrong, but because the holdout is already
spent: model/player_projection_results.json records the held-out table for
the SHIPPED configuration, taken once on 2024-2025, and model/ratings.py is
the worked example of what happens when a test column is consulted while a
knob is being turned. So NO held-out metric is computed here at all. There is
nothing to leak because the sealed numbers do not exist in this process.

WHAT IT DOES INSTEAD
Sweeps each constant on TRAINING seasons only and reports the surface. Three
things come out of that, none of which needs a holdout:

  1. WHETHER THE CONSTANT MATTERS. If train log-loss is flat across a wide
     sweep, the missing grid never mattered and the wound is cosmetic. If it
     is steep, the constant is load-bearing and being unjustified is serious.
  2. WHETHER THE SHIPPED VALUE IS THE TRAIN ARGMIN. If it is not, somebody
     overrode the training answer -- which is exactly how half_life=100
     shipped, and that took a year and two graded seasons to catch.
  3. A REPRODUCIBLE SPECIFICATION. The grid is now a committed file rather
     than a memory of a scratch harness.

THE TRAP INSIDE THIS SCRIPT, WHICH WAS WORSE THAN I DESIGNED FOR
The first version of this file guarded one constant, TD_MIN_LAMBDA, on the
grounds that it decides which player-weeks enter the sample. The guard fired
on EVERY constant instead, because the eligibility test is
max(lambda, base) >= TD_MIN_LAMBDA and every constant here moves lambda. Each
sweep value was scoring a different population.

That is not a cosmetic problem. Admitting more low-lambda players LOWERS
log-loss for free -- they are easy negatives -- so a configuration looks
better precisely by being more permissive. The apparent winner in the first
run, TIER_K = {0: 12, 1: 8, 2: 4}, scored 0.5213 against the shipped 0.5325
while admitting 1,124 more player-weeks. That gap is the extra rows, not the
constant.

So every configuration is now evaluated on a FIXED EVALUATION SET: the
player-weeks eligible under the SHIPPED constants. Only the probabilities
vary. Coverage is reported per row, because a configuration that cannot
produce a lambda for a shipped-eligible player-week is telling you something
too.

Writes model/player_td_grid.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model import player_projection as pp  # noqa: E402

OUT = HERE / "player_td_grid.json"

#: Train seasons. model/player_projection_results.json records the shipped
#: configuration's holdout as 2024-2025; those seasons do not appear here.
FIRST, LAST = 2016, 2023

#: The sweeps. Shipped value first in each list so the output is easy to read,
#: and ranges chosen to bracket the shipped value rather than to centre on it.
#:
#: SHAPE_ONLY constants do not change the walk-forward at all -- only the
#: recalibration fitted afterwards -- so they are swept without re-running the
#: engine. Everything else costs a full pass.
SHAPE_ONLY = {
    "TD_POWER": [0.55, 0.35, 0.45, 0.65, 0.75, 0.85, 1.0],
}

ENGINE = {
    "OPP_SHRINK_TD": [0.5, 0.0, 0.25, 0.75, 1.0],
    "ENV_DAMP": [0.5, 0.0, 0.25, 0.75, 1.0],
    "USAGE_W": [0.7, 0.0, 0.35, 0.85, 1.0],
    "RZ_OPP_SHRINK": [0.5, 0.0, 0.25, 0.75, 1.0],
    "LONG_CRED": [150.0, 50.0, 100.0, 250.0, 400.0],
    "ENV_CLAMP": [(0.6, 1.5), (0.8, 1.25), (0.7, 1.4), (0.5, 1.75), (0.4, 2.0)],
    "TIER_CUTS": [(8.0, 15.0), (6.0, 12.0), (7.0, 14.0), (10.0, 18.0), (12.0, 20.0)],
    "TIER_K": [{0: 5.0, 1: 3.0, 2: 1.5}, {0: 2.0, 1: 1.5, 2: 1.0},
               {0: 3.0, 1: 2.0, 2: 1.0}, {0: 8.0, 1: 5.0, 2: 2.5},
               {0: 12.0, 1: 8.0, 2: 4.0}],
}

#: Changes the SAMPLE, not just the scores. Swept, reported, and excluded from
#: any comparison of log-losses. See the module docstring.
SAMPLE_CHANGING = {
    "TD_MIN_LAMBDA": [0.15, 0.05, 0.10, 0.20, 0.30],
}


def _td_rows(preds: pd.DataFrame) -> pd.DataFrame:
    td = preds[preds["mkt"] == "anytime_td"]
    return td[td["season"] > td["season"].min() + pp.BURN_IN_SEASONS - 1]


def _keys(td: pd.DataFrame) -> pd.Index:
    return pd.MultiIndex.from_frame(td[["season", "week", "player"]])


def _train_logloss(preds: pd.DataFrame,
                   eval_keys: pd.Index | None = None) -> dict:
    """Train log-loss and Brier for anytime_td under the current constants.

    build_shape reads the module constants, so the per-tier recalibration
    refits for each configuration -- correct, because `a` is part of the
    configuration and not a number the sweep should hold still.

    eval_keys FIXES THE POPULATION. Without it every configuration scores a
    different set of player-weeks and the comparison rewards permissiveness
    rather than accuracy. See the module docstring.
    """
    shapes = pp.build_shape(preds)
    td = _td_rows(preds)
    if not len(td):
        return {"train_log_loss": float("nan"), "train_brier": float("nan"),
                "n": 0, "coverage": 0.0}
    if eval_keys is not None:
        td = td.set_index(_keys(td))
        td = td[~td.index.duplicated()]
        td = td.reindex(eval_keys).dropna(subset=["proj", "actual"])
        coverage = len(td) / len(eval_keys)
    else:
        coverage = 1.0
    tiers = (td["tier"].values if "tier" in td.columns
             else np.array([None] * len(td)))
    p = [pp.prob_score(shapes, lam, t)
         for lam, t in zip(td["proj"].values, tiers)]
    ll, brier = pp._logloss_brier(p, td["actual"].values)
    return {"train_log_loss": round(ll, 6), "train_brier": round(brier, 6),
            "n": int(len(td)), "coverage": round(float(coverage), 4)}


def _sweep(name: str, values: list, data: pd.DataFrame, rerun: bool,
           cached_preds: pd.DataFrame | None, eval_keys: pd.Index) -> dict:
    shipped = getattr(pp, name)
    rows = []
    for v in values:
        setattr(pp, name, dict(v) if isinstance(v, dict) else v)
        try:
            preds = (pp.walk_forward(FIRST, LAST, data=data)[0]
                     if rerun else cached_preds)
            m = _train_logloss(preds, eval_keys)
        finally:
            setattr(pp, name, shipped)
        m["value"] = list(v) if isinstance(v, tuple) else v
        rows.append(m)
        print(f"    {name} = {v!r:28s} ll {m['train_log_loss']:.6f}  "
              f"brier {m['train_brier']:.6f}  n {m['n']}  "
              f"coverage {m['coverage']:.3f}", flush=True)

    lls = [r["train_log_loss"] for r in rows]
    best = rows[int(np.nanargmin(lls))]
    low_cov = [r["value"] for r in rows if r["coverage"] < 0.95]
    return {
        "shipped": list(shipped) if isinstance(shipped, tuple) else shipped,
        "rows": rows,
        "train_argmin": best["value"],
        "shipped_is_train_argmin": best["value"] == rows[0]["value"],
        "spread_of_train_log_loss": round(
            float(np.nanmax(lls) - np.nanmin(lls)), 6),
        "evaluated_on_fixed_population": True,
        "values_with_incomplete_coverage": low_cov,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sweep the TD constants on train")
    ap.add_argument("--cache", default=os.environ.get("PP_MERGED_CACHE", ""),
                    help="pickle of the merged training frame, to skip loading")
    a = ap.parse_args(argv)

    if a.cache and Path(a.cache).exists():
        import pickle
        data = pickle.load(open(a.cache, "rb"))
        print(f"loaded merged frame from {a.cache}: {len(data)} rows")
    else:
        stats = pp.load_seasons(FIRST, LAST)
        data = pp.merge_rz(stats, pp.load_rz(FIRST, LAST))

    t0 = time.time()
    base_preds, _ = pp.walk_forward(FIRST, LAST, data=data)
    print(f"baseline walk-forward: {len(base_preds)} rows in "
          f"{time.time() - t0:.1f}s")
    eval_keys = _keys(_td_rows(base_preds))
    eval_keys = eval_keys[~eval_keys.duplicated()]
    base = _train_logloss(base_preds, eval_keys)
    print(f"shipped configuration: train log-loss {base['train_log_loss']:.6f} "
          f"on {base['n']} player-weeks, which is the fixed evaluation set")

    report = {
        "_provenance": {
            "script": "model/calibrate_player_td_lambda.py",
            "closes": "migration/ledger.yaml frozen-threshold-grid",
            "train_seasons": [FIRST, LAST],
            "burn_in_seasons": pp.BURN_IN_SEASONS,
            "metric": "log-loss of anytime_td on TRAINING seasons",
            "no_holdout_metric_computed": (
                "deliberately. 2024-2025 are spent on the shipped "
                "configuration's grade in model/player_projection_results.json,"
                " and model/ratings.py is the worked example of a test column "
                "consulted mid-selection. Nothing sealed exists in this "
                "process, so nothing can leak from it."
            ),
            "selects_nothing": (
                "the train argmin is reported and NOT adopted. Adopting it "
                "would be a new selection that the next holdout has to pay "
                "for, and the point of this file is to document the surface, "
                "not to move along it."
            ),
        },
        "shipped_configuration": base,
        "fixed_evaluation_set": {
            "n": int(len(eval_keys)),
            "definition": "the anytime_td player-weeks eligible under the "
                          "SHIPPED constants, after the burn-in season",
            "why": ("eligibility is max(lambda, base) >= TD_MIN_LAMBDA and "
                    "every constant here moves lambda, so a free-floating "
                    "sample rewards permissiveness: admitting more "
                    "low-lambda players lowers log-loss because they are "
                    "easy negatives"),
        },
        "sweeps": {},
    }

    for name, values in SHAPE_ONLY.items():
        print(f"  [shape-only] {name}")
        report["sweeps"][name] = _sweep(name, values, data, False,
                                        base_preds, eval_keys)
    for name, values in ENGINE.items():
        print(f"  [full re-run] {name}")
        report["sweeps"][name] = _sweep(name, values, data, True, None,
                                        eval_keys)
    for name, values in SAMPLE_CHANGING.items():
        print(f"  [sample-changing] {name}")
        s = _sweep(name, values, data, True, None, eval_keys)
        s["note"] = (
            "this constant decides eligibility directly, so a value above "
            "the shipped one cannot cover the fixed evaluation set at all. "
            "Read its coverage column before its log-loss column."
        )
        report["sweeps"][name] = s

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\nsummary:")
    for name, s in report["sweeps"].items():
        mark = ("" if s["shipped_is_train_argmin"]
                else "   <-- shipped is NOT the train argmin")
        cov = ("" if not s["values_with_incomplete_coverage"]
               else f"   ({len(s['values_with_incomplete_coverage'])} values "
                    "with incomplete coverage)")
        print(f"  {name:16s} spread {s['spread_of_train_log_loss']:.6f}{cov}{mark}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
