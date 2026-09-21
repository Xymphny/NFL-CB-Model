#!/usr/bin/env python3
"""Baseball has the same two-rule problem hockey has, and it lands elsewhere.

WHY THIS EXISTS
model/calibration_pit.json flagged the MLB HOME score as marginally
miscalibrated -- a maximum PIT deviation of 0.0133 against a Kolmogorov scale
of 0.0123 -- while the away side passed. A one-sided defect in a model that
treats the two sides symmetrically is a structural hint, not noise, and
following it lands on two league rules that no smooth distribution expresses.

  1. EXTRA INNINGS RESOLVE EVERY GAME. Across 12,148 games, zero finished
     tied. The shipped model assigns 10.07% of its mass to a tied final
     score, which is the same defect ADR 0007 found in hockey.

  2. THE HOME TEAM STOPS BATTING WHEN IT LEADS. It never gets a ninth
     inning while ahead, and a walk-off ends play the instant it takes the
     lead. Both truncate the home score from above, in exactly the games the
     home team wins.

The fingerprints are unambiguous:

    home runs when home wins      6.024
    away runs when away wins      6.443      the winner's own score, 0.42 lower
    variance of home score        9.63
    variance of away score       10.49       narrower, as truncation predicts
    P(home wins by exactly 1)    0.1725
    P(away wins by exactly 1)    0.1111      a 6.1 point asymmetry

WHERE IT HURTS, WHICH IS NOT WHERE IT LOOKS
The runline sits at 1.5 and is FINE: the model puts 0.6420 below the line
against an actual 0.6409. The fictitious tie mass and the missing one-run
wins are both on the same side of 1.5, so the errors cancel where the market
is priced.

The MONEYLINE is the broken one, and it is the market that looks safest
because recommend.py already conditions the tie out. Conditioning
redistributes that 10% PROPORTIONALLY, and reality gives it overwhelmingly to
the home side. Model P(home | no tie) is 0.5063 against an actual home win
rate of 0.5315 -- a 2.5 point understatement, systematic, in one direction, on
every game.

Writes model/mlb_rules_structure.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coverline.core.distributions import (  # noqa: E402
    NegativeBinomialScoreDistribution as NB,
)
from coverline.execution.recommend import cover_probability  # noqa: E402
from coverline.leagues.mlb.model import R_AWAY, R_HOME  # noqa: E402
from model.mlb_model import run_walk_forward  # noqa: E402

OUT = HERE / "mlb_rules_structure.json"


def main() -> int:
    sch = pd.read_csv(HERE / "mlb_schedule_cache.csv")
    pitch = pd.read_csv(HERE / "mlb_pitching_cache.csv")
    w = run_walk_forward(sch, pitch).dropna(
        subset=["home_score", "away_score", "exp_home", "exp_away"])
    w = w[(w.home_score >= 0) & (w.away_score >= 0)]
    h = w.home_score.to_numpy(int)
    a = w.away_score.to_numpy(int)
    m = h - a

    model_margin = np.zeros(9)
    p_home, rl_home = [], []
    for r in w.itertuples():
        d = NB(float(r.exp_home), float(r.exp_away), R_HOME, R_AWAY)
        for i, k in enumerate(range(-4, 5)):
            model_margin[i] += d.margin_pmf(k)
        p_home.append(cover_probability(d, 0.0)[0])
        rl_home.append(1.0 - d.margin_cdf(1.5))
    model_margin /= len(w)

    report = {
        "_provenance": {
            "script": "model/measure_mlb_rules.py",
            "n_games": int(len(w)),
            "seasons": sorted(int(s) for s in w.season.unique()),
            "descriptive": True,
            "graded": False,
            "note": ("properties of the data and of the SHIPPED model against "
                     "it; nothing is fitted here, so no season is spent"),
        },
        "extra_innings_resolve_every_game": {
            "ties_in_final_scores": int((m == 0).sum()),
            "model_mass_on_a_tie": round(float(model_margin[4]), 4),
        },
        "home_truncation": {
            "mean_home_runs_when_home_wins": round(float(h[m > 0].mean()), 4),
            "mean_away_runs_when_away_wins": round(float(a[m < 0].mean()), 4),
            "variance_home": round(float(h.var(ddof=1)), 4),
            "variance_away": round(float(a.var(ddof=1)), 4),
            "p_home_wins_by_one": round(float((m == 1).mean()), 4),
            "p_away_wins_by_one": round(float((m == -1).mean()), 4),
            "why": ("the home team never bats in the ninth while leading and "
                    "a walk-off ends play the instant it takes the lead, so "
                    "its score is truncated from above in exactly the games "
                    "it wins"),
        },
        "margin_distribution": {
            str(k): {"model": round(float(model_margin[i]), 4),
                     "actual": round(float((m == k).mean()), 4)}
            for i, k in enumerate(range(-4, 5))
        },
        "market_impact": {
            "moneyline": {
                "model_p_home_conditioned": round(float(np.mean(p_home)), 4),
                "actual_home_win_rate": round(float((m > 0).mean()), 4),
                "error": round(float((m > 0).mean() - np.mean(p_home)), 4),
                "why": ("conditioning the tie out redistributes that mass "
                        "PROPORTIONALLY; the walk-off rule gives it "
                        "overwhelmingly to the home side"),
            },
            "runline": {
                "model_p_home_minus_1_5": round(float(np.mean(rl_home)), 4),
                "actual": round(float((m >= 2).mean()), 4),
                "error": round(float((m >= 2).mean() - np.mean(rl_home)), 4),
                "why": ("the fictitious tie mass and the missing one-run wins "
                        "are on the SAME side of 1.5, so the errors cancel "
                        "where this market is priced. Right for the wrong "
                        "reason, and it is the reason that would change."),
            },
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    mk = report["market_impact"]
    print(f"ties: model {report['extra_innings_resolve_every_game']['model_mass_on_a_tie']}"
          f" actual {report['extra_innings_resolve_every_game']['ties_in_final_scores']}")
    print(f"moneyline error {mk['moneyline']['error']:+.4f}   "
          f"runline error {mk['runline']['error']:+.4f}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
