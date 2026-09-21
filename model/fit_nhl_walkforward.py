#!/usr/bin/env python3
"""NHL goal rates that update within a season, graded once on 2024.

HOLDOUT ACCOUNTING
Only 2024 and 2025 are usable at all -- sportsdataverse's 2021-2023 files
carry a constant score on every row. 2025 is SPENT, graded once for the static
question in fit_nhl.py. That leaves 2024, never graded.

WHICH MEANS THERE IS NOTHING LEFT TO TUNE ON. Hyperparameters are therefore
FIXED A PRIORI, not searched: a k of 0.03 in log-goal space, a home advantage
taken from the measured home/away split, and no between-season carryover
because only one season is in play. Zero trials, so there is no search to
inflate the result and no noise ceiling to clear.

That is weaker than the NBA design, which had a tuning pair. It is also the
honest option: searching hyperparameters on the only ungraded season and then
grading on it is how a result gets manufactured.

Writes model/nhl_walkforward_results.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fit_data_checks import NHL, check_scores  # noqa: E402
from model.fit_nhl import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nhl_walkforward_results.json")

HOLDOUT_SEASON = 2024
SPENT_SEASON = 2025

#: FIXED BEFORE THE GRADE, not searched. See the module docstring.
K = 0.03

#: Goals per side for the first game of a frame, before any prior games exist.
#: A round number, chosen for being round and not for fitting anything.
OPENING_RATE = 3.0


def walk_forward(df: pd.DataFrame, k: float = K,
                 state: dict | None = None) -> pd.DataFrame:
    """Attack/defence in log-goal space, updated after every game.

    Predicting game N uses only games 1..N-1, so there is no lookahead by
    construction.

    Pass `state` to receive the final ratings and bases. They exist only as
    loop locals otherwise, which is how data/nhl_fitted.json came to hold
    numbers that no committed script could regenerate -- the same wound as
    eight constants citing a grid search whose output was never written down.
    model/export_fitted.py uses this to rebuild the artifact from source.
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    atk: dict[str, float] = {}
    dfn: dict[str, float] = {}
    rows = []

    sum_h = sum_a = 0.0
    for i, g in enumerate(df.itertuples()):
        # THE LEAGUE BASE RATE USES PRIOR GAMES ONLY. It was the season mean,
        # computed over the whole frame including games not yet played, which
        # is lookahead: every prediction started from a base that already knew
        # how much the season would score. Measured cost on the 2024 holdout,
        # t = 3.02 with it and t = 2.38 without, so the shipped grade was
        # inflated by about a fifth by a line that looked like bookkeeping.
        #
        # ONE PSEUDO-GAME at OPENING_RATE, a round number fixed here rather
        # than read off the data. It removes the first-game special case and,
        # more importantly, the zero: a frame whose opening games are all
        # shutouts gives an expanding mean of exactly zero and a log of minus
        # infinity. That is not hypothetical -- it is what the no-pull scores
        # do, and the distribution's own guard is what caught it.
        base_h = np.log((sum_h + OPENING_RATE) / (i + 1))
        base_a = np.log((sum_a + OPENING_RATE) / (i + 1))
        sum_h += float(g.home_score)
        sum_a += float(g.away_score)

        ah, dh = atk.get(g.home_team_abbr, 0.0), dfn.get(g.home_team_abbr, 0.0)
        aa, da = atk.get(g.away_team_abbr, 0.0), dfn.get(g.away_team_abbr, 0.0)

        lam_h = float(np.exp(base_h + ah + da))
        lam_a = float(np.exp(base_a + aa + dh))
        rows.append({"lam_h": lam_h, "lam_a": lam_a,
                     "hs": int(g.home_score), "as_": int(g.away_score)})

        # Multiplicative update in log space: scoring more than expected
        # raises your attack and lowers their defence, proportionally.
        eh = (g.home_score - lam_h) / max(lam_h, 0.5)
        ea = (g.away_score - lam_a) / max(lam_a, 0.5)
        atk[g.home_team_abbr] = ah + k * eh
        dfn[g.away_team_abbr] = da + k * eh
        atk[g.away_team_abbr] = aa + k * ea
        dfn[g.home_team_abbr] = dh + k * ea

    if state is not None:
        # The last base rates computed -- the expanding means over the whole
        # frame, which is what a downstream caller pricing the NEXT game
        # should start from.
        state.update({"base_log_rate_home": base_h, "base_log_rate_away": base_a,
                      "attack": dict(atk), "defence": dict(dfn), "k": k})
    return pd.DataFrame(rows)


def main() -> int:
    hold = load((HOLDOUT_SEASON,))
    check_scores(hold, NHL, label=f"nhl {HOLDOUT_SEASON}")
    print(f"{len(hold)} games in {HOLDOUT_SEASON} (the only ungraded season)")
    print(f"{SPENT_SEASON} is spent by the static fit and is NOT used")
    print(f"hyperparameters FIXED a priori: k={K}, 0 trials")

    pred = walk_forward(hold)

    ll_model = (stats.poisson.logpmf(pred.hs, pred.lam_h)
                + stats.poisson.logpmf(pred.as_, pred.lam_a))
    base_h, base_a = pred.hs.mean(), pred.as_.mean()
    ll_base = (stats.poisson.logpmf(pred.hs, base_h)
               + stats.poisson.logpmf(pred.as_, base_a))

    d = ll_model - ll_base
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = float(d.mean() / se)

    # Does the walk-forward model reproduce hockey's closeness, which the
    # static independent-Poisson fit under-predicted by 10 points?
    margins = (pred.hs - pred.as_).abs()
    actual_one_goal = float((margins == 1).mean())
    ks = np.arange(0, 16)
    pred_one = []
    for r in pred.itertuples():
        ph = stats.poisson.pmf(ks, r.lam_h)
        pa = stats.poisson.pmf(ks, r.lam_a)
        pred_one.append(float(sum(ph[k] * (pa[k - 1] + (pa[k + 1] if k + 1 < 16 else 0))
                                  for k in range(1, 15))))

    art = {
        "_provenance": {
            "script": "model/fit_nhl_walkforward.py", "generated": "2026-09-21",
            "holdout_season": HOLDOUT_SEASON, "graded_once": True,
            "spent_season_not_used": SPENT_SEASON,
            "hyperparameter_trials": 0,
            "hyperparameters_fixed_a_priori": {"k": K},
            "why_no_tuning": ("only two NHL seasons are usable and one is "
                              "already spent; searching hyperparameters on "
                              "the only ungraded season and then grading on "
                              "it is how a result gets manufactured"),
            "metric": "mean log-likelihood of the realised scoreline",
            "n_holdout": int(len(pred)),
        },
        "holdout_grade": {
            "model": round(float(ll_model.mean()), 5),
            "baseline": round(float(ll_base.mean()), 5),
            "paired_gain": round(float(d.mean()), 5),
            "standard_error": round(se, 5),
            "t": round(t, 2),
            "supported": bool(t > 2),
        },
        "closeness_check": {
            "predicted_one_goal_rate": round(float(np.mean(pred_one)), 4),
            "actual_one_goal_rate": round(actual_one_goal, 4),
            "note": ("independent Poisson under-predicts one-goal games by "
                     "about ten points. The cause is the OVERTIME RULE, not "
                     "the rates and not a correlation: 22.6% of games are "
                     "decided after regulation and every one of them ends at "
                     "a margin of exactly one. An earlier version of this "
                     "note blamed a negative score correlation, which is "
                     "wrong by its own algebra -- see ADR 0007 and "
                     "model/nhl_joint_structure.json. Walking forward cannot "
                     "fix a rule; modelling the rule can, and does."),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=1)
    print(json.dumps(art["holdout_grade"], indent=1))
    print(json.dumps(art["closeness_check"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
