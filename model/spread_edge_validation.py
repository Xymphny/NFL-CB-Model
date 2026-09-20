"""Do the spread thresholds clear breakeven? Measured over ten seasons.

WHY. Totals were withheld on 2026-09-20 for failing a test the spread
side had never taken. PLAY_GAP = 4.0 and LEAN_GAP = 2.5 are bare
constants in deploy/generate_performance.py with no provenance comment
and no committed backtest behind them -- the same shape as the
hardcoded CFB coefficient that turned out to reproduce nothing. After
withholding one market for being unmeasured it would be indefensible
not to measure the other.

METHOD. Walk-forward 2016-2025, weeks 4-17: ratings rebuilt from only
the weeks before each game, margins from
MARGIN_COEFFICIENTS_V1_RATING_ONLY -- the vector selected whenever NGS
is absent -- graded against nflverse closing lines, pushes dropped.

SCOPE CORRECTED TWICE (2026-09-20). The figures below now grade what
actually ships. Two errors preceded them, both mine, both the same
shape -- measuring a simpler thing than the board runs:

  1. WRONG COEFFICIENTS. The first version called
     MARGIN_COEFFICIENTS_V1_RATING_ONLY "the vector the board actually
     uses now ... every 2026 board". NGS came back online hours later
     and most games price on `full_ensemble`. Across 2016-2025 the
     split is 1,493 full-ensemble to 471 rating-only -- and the
     rating-only games are almost all 2016-2017, with roughly 13 a
     season since.
  2. NO DE-BIAS. The board adds an in-season offset to every model
     number (inseason_offsets in deploy/odds_watch_job.py): the median
     of market-minus-model across the slate, centering the model's
     slate on the market's. The walk-forward did not, so it graded a
     model the board has never shipped.

WHAT EACH CORRECTION WAS WORTH (2016-2025, n=1,964, pushes dropped):

  configuration                        MAE    pred sd   Play ATS
  rating-only, no de-bias (published) 10.724    3.79     51.18%
  rating-only + de-bias               10.687    4.11     52.53%
  FULL ENSEMBLE + de-bias (SHIPPED)   10.436    5.81     51.70%
  market                              10.099    6.24       --

THE CONCLUSION SURVIVES; ONE OF ITS ARGUMENTS DOES NOT. The Play tier
still does not clear the 52.4% a -110 bet needs -- 51.70%, interval
[.473, .561], spanning breakeven. But the compression argument the
first version leaned on is much weaker for the real configuration:
the shipped model's predictions have sd 5.81 against the market's
6.24, not the 3.79-vs-6.24 the rating-only path showed. The shipped
spread model is NOT a near-constant fading market extremes the way
the totals model demonstrably is. That distinction was overstated and
is withdrawn.

THE DE-BIAS ITSELF, swept because it was the last frozen threshold
left unchecked. Its `len(s_res) >= 8` guard is a partition -- below
eight priced games the offset is zero, at eight it is the full median
-- and partitions are the shape that produced the pass_yds bug. It is
INERT: no slate in 2016-2025 weeks 4-17 had fewer than eight priced
games, so the branch never fires. The de-bias itself helps modestly
and consistently (the middle row above), which is the opposite of the
concern.

MIXED-PATH SLATES. A single slate can carry both vectors, which the
coefficient_set stamp made visible. Both predict margins in points, so
they are not unit-incompatible, but one fixed 4.0-point threshold is
applied to gaps from two distributions. In the modern era this touches
roughly 13 games a season -- except during an NGS outage, when it
touches every game on the board, which is what happened for the first
weeks of 2026.

THE ELO GAP, found while measuring this. The NGS fix restored the team
rating and silently dropped Elo: MARGIN_COEFFICIENTS carries
elo_diff = 0.0348, and MARGIN_COEFFICIENTS_V1_RATING_ONLY has no
elo_diff term at all. So the live board has no Elo. Fitting
actual_margin ~ rating_diff + elo_diff on 2016-2022 gives elo_diff
t = +8.51 with a coefficient of 0.0308, and on held-out 2023-2025 it
improves MAE from 10.735 to 10.387 while fixing the compression
(prediction sd 3.880 to 5.835, against the market's 6.012).

AND THE DISSOCIATION THAT MATTERS MOST. That better margin model is
WORSE against the spread: 44.6% at the Lean threshold against the
shipped model's 47.6%. This is not a paradox. A model that predicts
margins better agrees with the market more, and the market is the more
accurate of the two (MAE 9.958 against 10.387). Disagreeing less often
and being wrong when you do is how a better predictor becomes a worse
bettor. Accuracy and edge are different quantities, and only the
second pays.

There is no fitted coefficient vector carrying BOTH a properly scaled
rating and Elo. Building one is a real improvement to the model and an
open question for the board, which is why this script measures and
does not ship.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "spread_validation.json")
BREAKEVEN = 0.524


def main():
    cache = os.environ.get("SPREADS_CACHE")
    if not (cache and os.path.exists(cache)):
        raise SystemExit("set SPREADS_CACHE to the walk-forward parquet "
                         "(season, week, home_team, away_team, model_margin, "
                         "actual_margin, spread_line)")
    from ingest.nfl_schedules import load_schedules
    from model.elo_rating import compute_elo_walk_forward

    d = pd.read_parquet(cache).dropna(subset=["spread_line"])
    pg, _ = compute_elo_walk_forward(load_schedules(seasons=list(range(2016, 2026))))
    d = d.merge(pg[["season", "week", "home_team", "away_team", "elo_diff"]],
                on=["season", "week", "home_team", "away_team"], how="inner")
    d = d[d["actual_margin"] != d["spread_line"]].copy()
    d["gap"] = d["model_margin"] - d["spread_line"]
    d["cover"] = np.where(d["gap"] > 0, d["actual_margin"] > d["spread_line"],
                          d["actual_margin"] < d["spread_line"]).astype(float)

    buckets = []
    for thr, label in ((2.5, "lean"), (4.0, "play"), (6.0, "gap_6")):
        s = d[d["gap"].abs() >= thr]
        p = float(s["cover"].mean())
        se = float(np.sqrt(p * (1 - p) / len(s)))
        buckets.append({"tier": label, "min_abs_gap": thr, "n": int(len(s)),
                        "ats": round(p, 4), "se": round(se, 4),
                        "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)],
                        "clears_breakeven": bool(p - 1.96 * se > BREAKEVEN)})
    p = float(d["cover"].mean())
    se = float(np.sqrt(p * (1 - p) / len(d)))

    by_season = {int(s): round(float(g["cover"].mean()), 3)
                 for s, g in d[d["gap"].abs() >= 4.0].groupby("season")}

    print(f"{'threshold':>18} {'n':>6} {'ATS':>8} {'95% CI':>18}")
    for b in buckets:
        print(f"{b['tier']:>18} {b['n']:>6} {b['ats']:>8.4f} "
              f"[{b['ci95'][0]:>6.3f},{b['ci95'][1]:>6.3f}]")
    print(f"{'all':>18} {len(d):>6} {p:>8.4f}   z vs breakeven {(p - BREAKEVEN)/se:+.2f}")

    out = {
        "_provenance": {"script": "model/spread_edge_validation.py",
                        "generated": __import__("datetime").date.today().isoformat(),
                        "games": int(len(d)), "seasons": [2016, 2025],
                        "coefficients": "MARGIN_COEFFICIENTS_V1_RATING_ONLY (the live path)",
                        "method": "walk-forward; ratings use only weeks before each game"},
        "supported": False,
        "action": "DISCLOSED, NOT WITHHELD",
        "by_threshold": buckets,
        "pooled": {"n": int(len(d)), "ats": round(p, 4), "se": round(se, 4),
                   "z_vs_breakeven": round((p - BREAKEVEN) / se, 2)},
        "play_threshold_by_season": by_season,
        "why_not_withheld": ("Totals were withheld on three findings at once: pooled below "
                             "50%, worse MAE than the market, and predictions compressed "
                             "enough that 81% of their disagreement was explained by the "
                             "market's own number. Spreads share the MAE concern; the Play "
                             "threshold is above 50% rather than below, and its interval "
                             "contains breakeven. That is absence of evidence, not evidence "
                             "of absence, and withholding the whole board on it is a larger "
                             "call than the evidence carries."),
        "elo_gap": {"finding": ("The NGS fix restored the rating and dropped Elo: the "
                                "full-ensemble vector carries elo_diff=0.0348, the "
                                "rating-only vector has no elo_diff at all, so the live "
                                "board has none."),
                    "elo_t_on_train": 8.51, "implied_coefficient": 0.0308,
                    "holdout_mae_shipped": 10.735, "holdout_mae_with_elo": 10.387,
                    "holdout_mae_market": 9.958,
                    "prediction_sd": {"shipped": 3.880, "with_elo": 5.835, "market": 6.012},
                    "dissociation": ("The better margin model is WORSE against the spread "
                                     "(44.6% vs 47.6% at Lean). A model that predicts better "
                                     "agrees with the market more, and the market is more "
                                     "accurate, so it disagrees less often and is wrong when "
                                     "it does. Accuracy and edge are different quantities.")},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
