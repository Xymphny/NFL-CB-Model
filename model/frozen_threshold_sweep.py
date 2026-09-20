"""Where else does a constant fitted in one era meet a moved league?

WHY. pass_yds fails its gate because prob_over() stratifies on
projected volume using cuts frozen from train terciles, and league
passing volume has fallen since. 46.8% of held-out rows land in the
low-volume stratum instead of 33.3%, and that stratum claims 47%
(model/pass_yds_stratum_drift.py). A confirmed bug is worth
generalizing, so this sweeps the engine's other frozen constants.

THE RESULT IS MOSTLY NEGATIVE, which is the point of running it: one
hit, three clean, and the sweep is now closed so nobody repeats it.

  constant        what it does              verdict
  ------------------------------------------------------------------
  stratum cuts    partitions yardage rows   HIT -- 13.5pp composition
                  into volume terciles      shift, explains pass_yds
  TIER_CUTS       partitions TD rows into   clean -- tier shares move
                  volume tiers (8/15)       1-3pp; carries and targets
                                            have not fallen like pass
                                            attempts have
  CONV_DEFAULT    league red-zone TD        clean -- 2024-25 sits -4.8%
                  conversion per zone,      to +3.7% against the pooled
                  pooled 2016-2025          constants, inside the
                                            season-to-season band (c5
                                            alone ranges .389-.433)
  CRED_OPP        shrinkage denominator,    clean -- pass_yds takes
                  opportunities before a    2.48 -> 2.74 games to reach
                  player beats his         credibility. A quarter of a
                  position mean             game.

THE PATTERN, which is the part worth keeping. The bug bites where a
constant PARTITIONS data and not where it scales or shrinks it. A
partition is a step function: drift moves rows across a boundary and
they inherit a different distribution wholesale. A shrinkage
denominator is continuous -- drift changes how fast a player earns his
own rate, never which bucket he lands in, and the effect is
proportional rather than discontinuous. So when hunting this class
again, look for cuts, terciles, tiers and thresholds; skip the
coefficients and the credibility constants.

STILL UNSWEPT, named so the gap is visible: PLAY_GAP/LEAN_GAP on the
margin side, and the in-season de-bias offsets. Both partition, so
both are candidates by the rule above.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frozen_threshold_sweep_results.json")
EARLY, LATE = range(2016, 2020), [2024, 2025]


def conversion_drift(rz):
    """CONV_DEFAULT: league red-zone conversion, pooled over a decade."""
    from model.player_projection import CONV_DEFAULT
    out = {}
    for z, const in CONV_DEFAULT.items():
        per = {}
        for s in sorted(rz["season"].unique()):
            sub = rz[rz["season"] == s]
            opp, td = sub[z].sum(), sub[z + "_td"].sum()
            per[int(s)] = round(float(td / opp), 4) if opp else None
        early = [v for s, v in per.items() if s in EARLY and v]
        late = [v for s, v in per.items() if s in LATE and v]
        e, l = sum(early) / len(early), sum(late) / len(late)
        out[z] = {
            "pooled_constant": const,
            "mean_2016_19": round(e, 4), "mean_2024_25": round(l, 4),
            "drift": round(l - e, 4),
            "pct_vs_constant": round((l - const) / const * 100, 1),
            "season_range": [min(v for v in per.values() if v),
                             max(v for v in per.values() if v)],
            "by_season": per,
        }
    return out


def credibility_drift(preds):
    """CRED_OPP: a denominator, not a partition. Included to show the
    difference, not because it was expected to fail."""
    from model.player_projection import CRED_OPP
    out = {}
    for mkt, k in CRED_OPP.items():
        d = preds[(preds["mkt"] == mkt) & (preds["proj"] > 0)]
        e = float(d[d["season"].between(2017, 2019)]["opp_proj"].median())
        l = float(d[d["season"].isin(LATE)]["opp_proj"].median())
        out[mkt] = {"k": k, "median_opp_then": round(e, 2), "median_opp_now": round(l, 2),
                    "games_to_credibility_then": round(k / e, 2),
                    "games_to_credibility_now": round(k / l, 2),
                    "shift_games": round(k / l - k / e, 2)}
    return out


def tier_drift(preds):
    """TIER_CUTS: a partition, so a real candidate -- and clean."""
    td = preds[preds["mkt"] == "anytime_td"]
    def shares(sel):
        v = sel["tier"].value_counts(normalize=True)
        return [round(float(v.get(i, 0)), 3) for i in (0, 1, 2)]
    tr, te = shares(td[td["season"].between(2017, 2023)]), shares(td[td["season"].isin(LATE)])
    return {"train_shares": tr, "holdout_shares": te,
            "max_shift_pp": round(max(abs(a - b) for a, b in zip(tr, te)) * 100, 1)}


def main():
    rz_path = os.environ.get("RZ_CACHE")
    pred_path = os.environ.get("PRED_CACHE")
    if not (rz_path and pred_path and os.path.exists(rz_path) and os.path.exists(pred_path)):
        raise SystemExit("set RZ_CACHE and PRED_CACHE to the cached parquets "
                         "(regenerate with model/player_projection.py's load_rz + walk_forward)")
    rz, preds = pd.read_parquet(rz_path), pd.read_parquet(pred_path)

    conv, cred, tier = conversion_drift(rz), credibility_drift(preds), tier_drift(preds)

    print("Frozen-threshold sweep\n")
    print(f"{'zone':>6} {'const':>7} {'2024-25':>9} {'vs const':>10}  season range")
    for z, r in conv.items():
        print(f"{z:>6} {r['pooled_constant']:>7.4f} {r['mean_2024_25']:>9.4f} "
              f"{r['pct_vs_constant']:>+9.1f}%  {r['season_range']}")
    print(f"\nTIER_CUTS: train {tier['train_shares']} -> holdout {tier['holdout_shares']} "
          f"(max {tier['max_shift_pp']}pp)")
    print("  compare: the yardage strata moved 13.5pp, which is the bug.")
    print("\nCRED_OPP (a denominator, not a partition):")
    for m, r in cred.items():
        print(f"  {m:>9}: {r['games_to_credibility_then']} -> "
              f"{r['games_to_credibility_now']} games ({r['shift_games']:+})")

    out = {
        "_provenance": {
            "script": "model/frozen_threshold_sweep.py",
            "generated": __import__("datetime").date.today().isoformat(),
            "prompted_by": "model/pass_yds_stratum_drift.py",
        },
        "verdicts": {
            "yardage_stratum_cuts": "HIT -- see pass_yds_stratum_drift_results.json",
            "TIER_CUTS": "clean", "CONV_DEFAULT": "clean", "CRED_OPP": "clean",
        },
        "rule_of_thumb": ("The bug bites where a constant PARTITIONS data, not where it "
                          "scales or shrinks it. Partitions are step functions: drift moves "
                          "rows across a boundary and they inherit a different distribution "
                          "wholesale. Shrinkage is continuous and its error is proportional."),
        "still_unswept": ["PLAY_GAP/LEAN_GAP", "in-season de-bias offsets"],
        "conversion_rates": conv, "tier_cuts": tier, "credibility": cred,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
