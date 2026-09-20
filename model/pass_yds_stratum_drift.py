"""Why pass_yds fails its gate: frozen strata meet a shrinking league.

THE PUZZLE. The player engine ships rush_yds and rec_yds and withholds
pass_yds, which over-claims at every synthetic line on held-out
2024-25 (claimed 31.19% vs actual 26.49% at 1.0x projection). Two
revisions have not fixed it, and the cause was never established --
"the biggest market sitting out" has been an unexplained failure.

THE MECHANISM. prob_over() reads an ECDF of actual/proj ratios
stratified by projected opportunities, and the stratum cuts are the
TRAIN terciles, frozen. League passing volume has fallen: median
projected attempts ran 36-37 in 2017-2020 and 32-33 in 2024-25. Frozen
cuts do not move with it, so 46.8% of held-out pass_yds rows land in
stratum 0 instead of the 33.3% the terciles were built to hold -- and
stratum 0 is the low-volume shape, where train P(ratio > 1) is 47%
against 23% and 5% in the other two.

Mixing those weights reproduces the failure exactly:
  0.467*0.4705 + 0.364*0.2288 + 0.169*0.0529 = 0.3119
which is the claimed number in the results artifact, to four decimals.

IT EXPLAINS ALL THREE MARKETS, which is the reason to believe it:
  market     volume drift   stratum-0 share   claim inflation   status
  pass_yds      -3.03           46.7%            +6.11pp        withheld
  rush_yds      +1.11           27.4%            -1.29pp        ships (conservative)
  rec_yds       -0.08           32.8%            +0.24pp        ships (calibrated)
One frozen-threshold bug, three outcomes, each the sign the mechanism
predicts. This is the cold-start family again: a constant fitted in
one era applied to another.

A SIMPSON'S PARADOX SITS ON TOP OF IT, and it is why the failure
looked like era drift in the outcomes. Pooled, held-out actual (26.5%)
is HIGHER than train (25.1%). Within every stratum it is 4-5pp LOWER.
The composition shift is doing the work, not the outcome distribution.

WHAT WAS TRIED AND REJECTED. If frozen absolute cuts are the problem,
stratify on volume normalized by the league's current median instead.
Graded on a train-internal split (fit 2017-2020, grade 2021-2023, so
the 2024-25 holdout is never spent):

                       frozen (shipped)    causal-normalized
  pass_yds   mean|err|     0.0152              0.0200
             worst over    +0.0269             -0.0026

It removes the over-claim and makes the mean error worse: an
over-claim becomes a larger under-claim. Directionally safer, not
calibrated. NOT SHIPPED.

A caveat that matters more than the result: an earlier version of this
test normalized by the FULL season's median, which is not knowable at
prediction time. That lookahead version scored 0.0105 and looked like
a fix. The causal version is the honest one, and it is the one above.

WHERE THIS LEAVES pass_yds. Still withheld, now for a known reason
rather than an unexplained one. The train-internal split also
under-powers the test: drift across it is -1.6 attempts against -3.0
into the holdout, so the very effect being corrected is weakest
exactly where it is legal to look repeatedly.

AN OPERATIONAL CONSEQUENCE FOR THE SHIPPING MARKETS. rush_yds and
rec_yds are calibrated today because their volume has not drifted.
Nothing protects them if it starts. The stratum-0 share is the thing
to watch, and this script prints it for all three.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pass_yds_stratum_drift_results.json")
TRAIN = range(2017, 2024)        # 2016 is burn-in, excluded from shapes
HOLDOUT = [2024, 2025]
MULTS = (0.7, 0.85, 1.0, 1.15, 1.3)


def load_preds(path=None):
    """Walk-forward predictions. Regenerated from the engine unless a
    cached parquet is passed -- the engine run pulls ten seasons."""
    if path and os.path.exists(path):
        return pd.read_parquet(path)
    from model.player_projection import load_rz, load_seasons, merge_rz, walk_forward
    data = load_seasons(2016, 2025)
    try:
        data = merge_rz(data, load_rz(2016, 2025))
    except Exception as err:                              # noqa: BLE001
        print(f"  red-zone merge unavailable ({err}); continuing on v2 lambda")
    preds, _ = walk_forward(2016, 2025, data=data)
    return preds


def diagnose(pr):
    pr = pr[pr["proj"] > 0]
    rows = {}
    for mkt in ("pass_yds", "rush_yds", "rec_yds"):
        d = pr[pr["mkt"] == mkt]
        tr, te = d[d["season"].isin(TRAIN)], d[d["season"].isin(HOLDOUT)]
        cuts = tr["opp_proj"].quantile([1 / 3, 2 / 3]).values

        def st(o, c=cuts):
            return 0 if o <= c[0] else (1 if o <= c[1] else 2)

        tr_s, te_s = tr.assign(st=tr["opp_proj"].map(st)), te.assign(st=te["opp_proj"].map(st))
        per = []
        for s in (0, 1, 2):
            a, b = tr_s[tr_s.st == s], te_s[te_s.st == s]
            per.append({
                "stratum": s, "n_train": int(len(a)),
                "train_p_ratio_gt_1": round(float(((a["actual"] / a["proj"]) > 1).mean()), 4),
                "holdout_share": round(float(len(b) / len(te_s)), 4),
                "holdout_p_ratio_gt_1": (round(float(((b["actual"] / b["proj"]) > 1).mean()), 4)
                                         if len(b) else None),
            })
        claim = sum(p["holdout_share"] * p["train_p_ratio_gt_1"] for p in per)
        pooled = float(((tr_s["actual"] / tr_s["proj"]) > 1).mean())
        rows[mkt] = {
            "train_median_opp": round(float(tr["opp_proj"].median()), 2),
            "holdout_median_opp": round(float(te["opp_proj"].median()), 2),
            "volume_drift": round(float(te["opp_proj"].median() - tr["opp_proj"].median()), 2),
            "stratum_0_share_holdout": per[0]["holdout_share"],
            "stratum_0_share_expected": 0.3333,
            "claim_inflation_vs_pooled": round(claim - pooled, 4),
            "reconstructed_claim_at_1x": round(claim, 4),
            "actual_at_1x": round(float((te["actual"] > te["proj"]).mean()), 4),
            "strata": per,
        }
    return rows


def main():
    cache = os.environ.get("PRED_CACHE")
    pr = load_preds(cache)
    rows = diagnose(pr)

    print("Frozen strata vs a shrinking league\n")
    print(f"{'market':>9} {'drift':>7} {'st0 share':>10} {'inflation':>10} "
          f"{'claim@1x':>9} {'actual@1x':>10}")
    for mkt, r in rows.items():
        print(f"{mkt:>9} {r['volume_drift']:>7.2f} {r['stratum_0_share_holdout']*100:>9.1f}% "
              f"{r['claim_inflation_vs_pooled']:>+10.4f} {r['reconstructed_claim_at_1x']:>9.4f} "
              f"{r['actual_at_1x']:>10.4f}")
    print("\nWatch the stratum-0 share. It is 33.3% by construction on train; the")
    print("distance from that is how far the frozen cuts have drifted out of date.")

    out = {
        "_provenance": {
            "script": "model/pass_yds_stratum_drift.py",
            "generated": __import__("datetime").date.today().isoformat(),
            "train": [min(TRAIN), max(TRAIN)], "holdout": HOLDOUT,
            "what_this_is": ("A diagnosis of a known gate failure, not a new gate. The "
                             "holdout numbers here re-describe a decision already made on "
                             "these seasons (pass_yds was withheld on them twice); no "
                             "parameter is selected from them."),
        },
        "finding": ("Stratum cuts are frozen train terciles. League passing volume fell, so "
                    "46.8% of held-out pass_yds rows land in the low-volume stratum instead "
                    "of 33.3%, and that stratum claims 47%. Mixing the shifted weights "
                    "reproduces the published 0.3119 claim exactly."),
        "candidate_fix_rejected": {
            "name": "stratify on volume normalized by the league's trailing median",
            "graded_on": "train-internal split (fit 2017-2020, grade 2021-2023)",
            "pass_yds_mean_abs_err": {"frozen": 0.0152, "causal_normalized": 0.0200},
            "pass_yds_worst_over_claim": {"frozen": 0.0269, "causal_normalized": -0.0026},
            "verdict": ("Removes the over-claim, worsens mean error -- trades an over-claim "
                        "for a larger under-claim. Not shipped."),
            "lookahead_warning": ("Normalizing by the FULL season median scores 0.0105 and "
                                  "looks like a fix. That number is not available at "
                                  "prediction time. The causal figures above are the real "
                                  "ones."),
        },
        "markets": rows,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
