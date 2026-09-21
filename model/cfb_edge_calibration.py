"""Refit the CFB edge -> cover curve from the committed caches.

WHY THIS EXISTS. The CFB board sized its cover probabilities from
`edgeCoefOverride={0.01828}` -- a bare constant in JSX that no
committed code produced. The 2026-09-20 audit could not reproduce it;
refitting from the caches in this repo gives 0.01623. The difference
is small, but a magic number in a display file is not evidence, and
this project's whole claim is that its numbers are checkable.

This script regenerates the value, its uncertainty, and the realized
bucket table into data/cfb_edge_calibration.json, which the frontend
reads instead. Run it after any cfb-backtest-job refresh.

WHAT IT FINDS (2026-09-20). Unlike the NFL curve, which is
sign-unstable and loses to a coin flip out of sample, CFB's realized
cover rate rises monotonically with edge -- 50.5% / 53.1% / 57.7%
across the 0-5 / 5-10 / 10-20 buckets. That is a real gradient. It is
still not significant at 95% (the interval grazes zero), so the value
ships WITH its uncertainty and the board discloses it, rather than
either hiding the caveat or throwing away a signal that looks real.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cfb_full_walk_forward_cache.csv")
LINES = os.path.join(HERE, "cfb_lines_cache.csv")
# Site-facing artifact, so it lives in data/ beside margin_dist.json
# rather than in model/ with the caches. It was written to model/ until
# 2026-09-20, where nothing shipped it: the board fetched
# /data/cfb_edge_calibration.json, got a 404, fell through to the NFL
# coefficient -- which is withheld as unsupported -- and quietly showed
# no cover probability for the one league whose curve is monotonic.
OUT = os.path.join(os.path.dirname(HERE), "data", "cfb_edge_calibration.json")
TRAIN_MAX = 2022        # margin fit trains here; 2023 is the held-out grade


def _fit_logistic(edge, covered):
    from scipy.optimize import minimize_scalar

    x, y = np.asarray(edge), np.asarray(covered)

    def nll(b):
        p = np.clip(1 / (1 + np.exp(-b * x)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    b = float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)
    p = 1 / (1 + np.exp(-b * x))
    info = float(np.sum(p * (1 - p) * x * x))
    return b, (float(1 / np.sqrt(info)) if info > 0 else float("nan"))


def main():
    for path in (CACHE, LINES):
        if not os.path.exists(path):
            raise SystemExit(f"missing {os.path.basename(path)} -- re-run cfb-backtest-job "
                             f"on Render to regenerate the caches, then run this again.")

    m = pd.read_csv(CACHE).merge(pd.read_csv(LINES),
                                 on=["season", "week", "home_team", "away_team"], how="inner")

    # The margin fit the backtest artifact documents, refit here so the
    # whole chain is reproducible rather than quoted.
    tr = m[m["season"] <= TRAIN_MAX]
    slope, intercept = np.polyfit(tr["rating_diff"], tr["actual_margin"], 1)

    te = m[m["season"] > TRAIN_MAX].copy()
    te["model_margin"] = slope * te["rating_diff"] + intercept
    te["edge"] = te["model_margin"] - te["spread_line"]
    pushes = te["actual_margin"] == te["spread_line"]
    covered = ((te["edge"] > 0) == (te["actual_margin"] > te["spread_line"])).astype(float)
    d = pd.DataFrame({"edge": te["edge"].abs(), "covered": covered})[~pushes]

    b, se = _fit_logistic(d["edge"].values, d["covered"].values)
    ci = [round(b - 1.96 * se, 5), round(b + 1.96 * se, 5)]

    buckets = []
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 999)]:
        sel = d[(d["edge"] >= lo) & (d["edge"] < hi)]
        if len(sel) >= 20:
            buckets.append({"edge_range": [lo, hi if hi < 999 else None], "n": int(len(sel)),
                            "actual_cover_pct": round(float(sel["covered"].mean()), 4)})
    realized = [bk["actual_cover_pct"] for bk in buckets]
    monotonic = all(x <= y for x, y in zip(realized, realized[1:]))

    out = {
        "_provenance": {
            "script": "model/cfb_edge_calibration.py",
            "generated": __import__("datetime").date.today().isoformat(),
            "margin_fit": f"{slope:.2f} * rating_diff + {intercept:.2f} (train <= {TRAIN_MAX})",
            "graded_on": f"{TRAIN_MAX + 1} held out",
            "replaces": ("the hardcoded edgeCoefOverride={0.01828} in frontend/src/App.jsx, "
                         "which no committed code reproduced"),
        },
        "edge_coef": round(b, 5),
        "standard_error": round(se, 5),
        "ci95": ci,
        "n_games": int(len(d)),
        "significant_at_95": bool(ci[0] > 0),
        "realized_monotonic_in_edge": bool(monotonic),
        "fit_check": buckets,
        "note": ("Realized cover rises with edge here, unlike the NFL curve -- a real gradient "
                 "on one held-out season. Not significant at 95%, so the board shows the "
                 "probability WITH its caveat rather than as a settled number."),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  coef {out['edge_coef']} +/- {out['standard_error']}  CI {ci}  n={out['n_games']}")
    print(f"  monotonic in edge: {monotonic}   significant at 95%: {out['significant_at_95']}")
    for bk in buckets:
        print(f"    edge {bk['edge_range']}: {bk['actual_cover_pct']*100:.1f}% (n={bk['n']})")


if __name__ == "__main__":
    main()
