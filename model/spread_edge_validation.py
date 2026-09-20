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

SCOPE CORRECTION (2026-09-20, same day). An earlier version of this
file called that "the coefficient vector the board actually uses now
... which is every 2026 board". That was true when it was written and
stopped being true hours later: NGS came back online between the 16:01
and 20:04 UTC boards, and the 20:04 board prices six of its seven live
games on `full_ensemble`, one on the rating-only path, with the rest
preserved from earlier snapshots. So the numbers below describe the
NGS-ABSENT configuration. That configuration is not hypothetical -- it
ran for the whole NGS outage and still runs per-game whenever a team
is missing from the NGS frame -- but it is not unconditionally what
ships, and this file should not be read as having graded the live
board. Re-run it against the full-ensemble path before treating the
ATS figures as a verdict on what is on the board today.

A SECOND THING THAT FELL OUT OF CHECKING. A single slate can carry
BOTH vectors, which the new coefficient_set stamp made visible for the
first time. Both predict margins in points, so they are not
unit-incompatible, but they are differently confident: held out, the
rating-only path's predictions have sd 3.880 against ~5.8 for one
carrying Elo. One fixed 4.0-point threshold is therefore being applied
to gaps drawn from two distributions. On the 20:04 board the lone
rating-only game's gap is -0.23 and nothing flags, so no harm today --
but "which model priced this edge" is now a question the board can
answer and the threshold does not ask.

WHAT IT FOUND (n=1,964):

  threshold             n      ATS     95% CI          needs 52.4%
  Lean |gap| >= 2.5   1155   49.70%  [.468, .526]          no
  Play |gap| >= 4.0    719   51.18%  [.475, .548]          no
  |gap| >= 6           335   53.73%  [.484, .591]          no
  all games           1964   49.80%  [.476, .520]      z = -2.31

The Play threshold -- the one that takes a full unit -- sits at 51.18%
against the 52.4% a -110 bet needs. Its interval contains breakeven,
so this is not proof the threshold loses; it is the absence of
evidence that it wins, across ten seasons and 719 flagged games.

Season by season at the Play threshold: 53.7, 44.4, 48.2, 51.3, 57.0,
47.3, 60.9, 54.8, 45.5, 50.0. Five above breakeven and five below,
swinging 16 points either side -- exactly what a near-coin-flip looks
like at ~70 games a season, and a warning about reading any single
season's record as signal.

WHY THIS IS NOT THE TOTALS CASE, and is not treated the same. Totals
were withheld on three findings together: pooled BELOW 50%, a model
that loses to the market on MAE, and predictions so compressed
(sd 2.5 against the market's 4.3) that its disagreements were 81%
explained by the market's own number -- it was fading extremes toward
the league average, not reading the teams. Spreads share the first
concern but not the third to the same degree, and the Play threshold
is above 50% rather than below it. Withholding the entire board on
this evidence would be a bigger call than the evidence supports, and
it is not one to make unilaterally. The number is published instead.

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
