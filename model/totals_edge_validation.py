"""Has the totals model ever beaten the market? Measured, finally.

WHY THIS EXISTS. The board flags totals at PLAY_GAP+1 and LEAN_GAP+1
and has done since the beginning. The spread side publishes its own
indictment -- performance.json carries model_mae against market_mae,
and the model is currently losing 14.74 to 13.69. Totals get no such
line, because generate_performance computes those two numbers from
`spreads` only. A market has been on the board, taking real stakes,
with no accuracy measurement of any kind behind it.

THE MODEL. predict_total() is two features: combined offensive VOA and
wind. That is all.

WHAT 1,039 WALK-FORWARD GAMES SAY (2021-2025, weeks 4-17, ratings
rebuilt from only the weeks before each game):

                    MAE      RMSE     prediction sd
  model total      10.583   13.353       2.505
  market total     10.230   13.099       4.269
  actual                                13.678

It loses to the market on both error measures, and the third column is
the more damning one: the model's totals barely move. Against actual
totals with a standard deviation of 13.7, it varies by 2.5 where the
market varies by 4.3. A prediction that is nearly a constant cannot
carry an edge, which is the same signature the NGS coefficient bug
left on margins.

AND THE BETS DO NOT WIN. Grading each disagreement as the board would
bet it (model above market = over, below = under), pushes dropped:

  threshold        n     hit rate    95% CI          clears 52.4%
  |gap| >= 3      423     48.9%   [.442, .537]           no
  |gap| >= 4      290     50.0%   [.442, .558]           no
  |gap| >= 5      186     52.7%   [.455, .599]           no
  |gap| >= 6       99     56.6%   [.468, .663]           no
  all            1029     49.6%   [.465, .526]     z = -1.82

WHAT IS HONESTLY MIXED ABOUT IT. The hit rate rises monotonically with
the threshold, which is what a real signal looks like, and the thin
top slice is the only thing here that is not negative. But no interval
clears breakeven, the pooled result sits BELOW it at z = -1.82, and
the model loses on MAE. Raising the threshold to 6 because that is
where this sample looks best is precisely the test-set selection that
put half_life=100 into production, so it is not done.

VERDICT: no demonstrated edge. Totals are withheld from the board the
way pass yards are withheld from the player engine -- in code, with
the number that put them there on the card. The machinery stays so the
claim can be re-tested; this is a verdict on the evidence, not a
permanent one.

HISTORY IS NOT REWRITTEN. Totals already published were real claims and
stay graded in the record, wins and losses alike. The withhold applies
from its effective date forward, which is why the grader checks a date
rather than a flag.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "totals_validation.json")
BREAKEVEN = 0.524
EFFECTIVE_FROM = "2026-09-20"


def walk_forward(first=2021, last=2025):
    from ingest.nfl_pbp import load_season
    from ingest.nfl_schedules import load_schedules
    from model.prediction import predict_total
    from model.ratings import (add_home_field_and_rest, add_recency_weights,
                               add_situation_buckets, compute_baselines,
                               compute_raw_voa, filter_garbage_time,
                               opponent_adjust, score_all_plays, team_ratings)
    rows = []
    for s in range(first, last + 1):
        sched, full = load_schedules(seasons=[s]), load_season(s)
        for wk in range(4, 18):
            df = full[full["week"] < wk].copy()
            if df.empty:
                continue
            df = add_situation_buckets(df)
            df = score_all_plays(df, use_turnover_luck_adjustment=True)
            df = filter_garbage_time(df)
            df = add_home_field_and_rest(df, sched)
            df = compute_raw_voa(df, compute_baselines(df))
            df = opponent_adjust(df, iterations=3, regression=0.5)
            df = add_recency_weights(df)
            r = team_ratings(df, use_recency_weights=True)
            g = sched[sched["week"] == wk].dropna(subset=["home_score", "away_score"])
            for _, gm in g.iterrows():
                h, a = gm["home_team"], gm["away_team"]
                if h not in r.index or a not in r.index:
                    continue
                rows.append({
                    "season": s, "week": wk,
                    "model_total": predict_total(
                        r.loc[h, "offense_voa"] + r.loc[a, "offense_voa"], gm.get("wind", 0.0)),
                    "actual_total": gm["home_score"] + gm["away_score"],
                    "market_total": gm.get("total_line", np.nan)})
        print(f"  walked {s}", flush=True)
    return pd.DataFrame(rows)


def main():
    cache = os.environ.get("TOTALS_CACHE")
    d = (pd.read_parquet(cache) if cache and os.path.exists(cache) else walk_forward())
    d = d.dropna(subset=["market_total"])

    acc = {}
    for name, col in (("model", "model_total"), ("market", "market_total")):
        e = d["actual_total"] - d[col]
        acc[name] = {"mae": round(float(e.abs().mean()), 3),
                     "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
                     "prediction_sd": round(float(d[col].std()), 3)}
    acc["actual_sd"] = round(float(d["actual_total"].std()), 3)

    g = d[d["actual_total"] != d["market_total"]].copy()      # drop pushes
    g["gap"] = g["model_total"] - g["market_total"]
    g["hit"] = np.where(g["gap"] > 0, g["actual_total"] > g["market_total"],
                        g["actual_total"] < g["market_total"]).astype(float)
    buckets = []
    for thr in (3.0, 4.0, 5.0, 6.0):
        s = g[g["gap"].abs() >= thr]
        if len(s) < 30:
            continue
        p = float(s["hit"].mean())
        se = float(np.sqrt(p * (1 - p) / len(s)))
        buckets.append({"min_abs_gap": thr, "n": int(len(s)), "hit_rate": round(p, 4),
                        "se": round(se, 4),
                        "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)],
                        "clears_breakeven": bool(p - 1.96 * se > BREAKEVEN)})
    p = float(g["hit"].mean())
    se = float(np.sqrt(p * (1 - p) / len(g)))
    pooled = {"n": int(len(g)), "hit_rate": round(p, 4), "se": round(se, 4),
              "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)],
              "z_vs_breakeven": round((p - BREAKEVEN) / se, 2)}

    supported = bool(any(b["clears_breakeven"] for b in buckets)
                     and acc["model"]["mae"] < acc["market"]["mae"])

    print(f"\n{'':>14} {'MAE':>8} {'RMSE':>8} {'pred sd':>9}")
    for k in ("model", "market"):
        a = acc[k]
        print(f"{k + ' total':>14} {a['mae']:>8.3f} {a['rmse']:>8.3f} {a['prediction_sd']:>9.3f}")
    print(f"{'actual sd':>14} {'':>8} {'':>8} {acc['actual_sd']:>9.3f}")
    print(f"\n{'threshold':>12} {'n':>6} {'hit':>8} {'95% CI':>18} {'clears':>8}")
    for b in buckets:
        print(f"  |gap|>={b['min_abs_gap']:<3.0f} {b['n']:>6} {b['hit_rate']:>8.4f} "
              f"[{b['ci95'][0]:>6.3f},{b['ci95'][1]:>6.3f}] "
              f"{'YES' if b['clears_breakeven'] else 'no':>8}")
    print(f"\n  pooled n={pooled['n']} hit {pooled['hit_rate']:.4f} "
          f"z vs breakeven {pooled['z_vs_breakeven']:+.2f}")
    print(f"\nsupported: {supported}")

    out = {
        "_provenance": {"script": "model/totals_edge_validation.py",
                        "generated": __import__("datetime").date.today().isoformat(),
                        "games": int(len(d)), "seasons": [2021, 2025],
                        "method": "walk-forward: ratings rebuilt from only the weeks before each game"},
        "supported": supported,
        "effective_from": EFFECTIVE_FROM,
        "accuracy": acc,
        "by_threshold": buckets,
        "pooled": pooled,
        "why": ("The totals model loses to the market on MAE and RMSE, its predictions vary "
                "by 2.5 points where the market varies by 4.3 against an actual spread of "
                "13.7, and no betting threshold clears the 52.4% breakeven -- pooled it sits "
                "below it at z=-1.82. The hit rate does rise with the threshold, which is "
                "what a real signal looks like, but raising the cut to where this sample "
                "looks best is the test-set selection that put half_life=100 into "
                "production. Withheld until it can be shown, not selected."),
        "history_not_rewritten": ("Totals published before effective_from were real claims "
                                  "and stay graded in the record, wins and losses alike."),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
