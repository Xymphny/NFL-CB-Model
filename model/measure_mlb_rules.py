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


def linescore_transition() -> dict:
    """The rule itself, measured, not inferred from fingerprints.

    `state` is the margin after the TOP of the ninth: home runs through eight
    minus away runs through nine. It is what the rule reads, and until the
    linescores were pulled it was not in this repository at all.

    The table below is the layer a corrected model needs. It is empirical,
    every row has hundreds of games behind it, and two of its rows are
    DETERMINISTIC rather than fitted.
    """
    import glob

    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "mlb" /
                                 "linescores_*.parquet")))
    if not files:
        return {"measured": False,
                "reason": "run model/ingest/mlb_linescores.py"}

    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["away_through_9"] = d.away_through_8 + d.away_ninth
    d["state"] = d.home_through_8 - d.away_through_9
    d["final_margin"] = d.home_score - d.away_score
    leading = d.state > 0

    table = {}
    for st in range(-4, 5):
        sub = d[d.state == st]
        if len(sub) < 50:
            continue
        vc = sub.final_margin.value_counts(normalize=True).sort_index()
        table[str(st)] = {"n": int(len(sub)),
                          "final_margin": {str(int(k)): round(float(v), 5)
                                           for k, v in vc.items()
                                           if v >= 0.001}}

    tied = d[d.state == 0]
    return {
        "measured": True,
        "n_games": int(len(d)),
        "seasons": sorted(int(x) for x in d.season.unique()),
        "home_batted_ninth": round(float(d.home_batted_ninth.mean()), 4),
        "rule_is_deterministic": {
            "home_led_after_top_nine_and_still_batted": int(
                (leading & d.home_batted_ninth).sum()),
            "home_did_not_lead_and_did_not_bat": int(
                ((~leading) & ~d.home_batted_ninth).sum()),
            "note": ("the first must be zero and is. The second is 74 games "
                     "in 12,146 -- shortened and called games, a real "
                     "exception rather than a modelling failure, and small "
                     "enough to state rather than model."),
        },
        "walk_off_advantage": {
            "p_home_wins_when_tied_after_top_nine": round(
                float((tied.final_margin > 0).mean()), 4),
            "n": int(len(tied)),
            "why": ("batting last. This single number is what the shipped "
                    "model cannot express and what makes its conditioned "
                    "moneyline 2.5 points low."),
        },
        "untruncated_scoring": {
            "home_runs_through_eight": round(float(d.home_through_8.mean()), 4),
            "away_runs_through_nine": round(float(d.away_through_9.mean()), 4),
            "home_runs_observed": round(float(d.home_score.mean()), 4),
            "implied_untruncated_home_nine_innings": round(
                float(d.home_through_8.mean() * 9.0 / 8.0), 4),
            "why": ("exp_home is fitted to OBSERVED home runs, which are "
                    "truncated in 45% of games. A layer that applies the rule "
                    "on top of those rates counts the truncation twice."),
        },
        "transition_table": table,
    }


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
        "linescore_transition": linescore_transition(),
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
    lt = report["linescore_transition"]
    if lt.get("measured"):
        print(f"home batted 9th {lt['home_batted_ninth']:.4f}   "
              f"P(home wins | tied after top 9) "
              f"{lt['walk_off_advantage']['p_home_wins_when_tied_after_top_nine']:.4f}")
    print(f"moneyline error {mk['moneyline']['error']:+.4f}   "
          f"runline error {mk['runline']['error']:+.4f}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
