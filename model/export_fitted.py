#!/usr/bin/env python3
"""Rebuild data/{nhl,nba}_fitted.json from source. Nothing here is hand-written.

WHY THIS EXISTS
Both fitted artifacts shipped this morning with no generator. They were
correct and they were graded, and a reader who wanted to know where ANA's
attack rating of -0.23424 came from had nowhere to look: the ratings lived as
loop locals inside walk_forward and were never returned.

That is the repository's oldest wound repeated -- eight production constants
citing a grid search whose output exists in no committed file. The number
being right does not fix it. A number nobody can regenerate is a claim.

WHAT IS REPRODUCED, AND WHAT IS NOT
The ratings, bases, hyperparameters and the holdout grade all come from
re-running the graded walk-forward on the retained inputs, so this script
reproduces the shipped artifact rather than asserting it. A test compares a
fresh export against the committed file and fails on any drift.

The prose fields -- why a season was not used, what is still unmodelled -- are
written here, because they are judgements and the place for a judgement is
one file, next to the numbers it qualifies.

SPENDS NOTHING. Re-running a grade that was already taken on the same season
with the same a-priori hyperparameters is not a second look at the holdout;
it is the same look, recomputed. No search happens here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import fit_nba_walkforward as nbawf  # noqa: E402
from model import fit_nhl_walkforward as nhlwf  # noqa: E402
from model.fit_nba import load as load_nba  # noqa: E402
from model.fit_nhl import load as load_nhl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

#: Digits kept in the artifacts. Matches what shipped; stated so a diff
#: against the committed file is a real comparison and not a rounding fight.
ND = 5

JOINT_WARNING = (
    "The RATES are graded and supported. The JOINT distribution is not, and "
    "the reason recorded when this first shipped was WRONG. It said the two "
    "scores are negatively correlated at -0.14 so real games stay closer than "
    "independent rates allow. Negative covariance makes margins MORE "
    "dispersed, not less -- Var(H-A) = Var(H) + Var(A) - 2Cov -- so that "
    "mechanism predicts the opposite of what is observed, and the measured "
    "dispersion ratio is 1.06 over 2016-2023, above one, not below. "
    "model/nhl_joint_structure.json has the measurement. What is actually "
    "happening is two league RULES, neither of them a correlation: the "
    "overtime rule, which is deterministic -- no NHL game ends tied and all "
    "2,166 games decided after regulation end at a margin of exactly one, so "
    "about 22% of the distribution's tie mass is moved onto plus and minus "
    "one by fiat -- and empty-net goals, which fire conditional on a late "
    "deficit and drain two-goal games into three-goal games across exactly "
    "the 1.5 line the puck line is priced on. The puck line therefore stays "
    "out of primary_markets, but the repair is a rules layer, not a fitted "
    "correlation, and a fitted correlation would also be non-stationary: it "
    "moves from -0.06 in 2017 to -0.14 in 2023."
)

CLOSENESS_NOTE = (
    "Independent Poisson under-predicts one-goal games by about ten points. "
    "The cause is the overtime rule, not the rates and not a correlation: "
    "22.6% of games are decided after regulation and every one of them ends "
    "at a margin of exactly one. Walking forward cannot fix a rule. See "
    "joint_distribution_warning."
)


def _round(obj, nd: int = ND):
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, nd)
    return obj


def export_nhl() -> dict:
    hold = load_nhl((nhlwf.HOLDOUT_SEASON,))
    state: dict = {}
    pred = nhlwf.walk_forward(hold, state=state)

    ll_model = (stats.poisson.logpmf(pred.hs, pred.lam_h)
                + stats.poisson.logpmf(pred.as_, pred.lam_a))
    ll_base = (stats.poisson.logpmf(pred.hs, pred.hs.mean())
               + stats.poisson.logpmf(pred.as_, pred.as_.mean()))
    d = ll_model - ll_base
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = float(d.mean() / se)

    margins = (pred.hs - pred.as_).abs()
    ks = np.arange(0, 16)
    pred_one = [
        float(sum(stats.poisson.pmf(ks, r.lam_h)[k]
                  * (stats.poisson.pmf(ks, r.lam_a)[k - 1]
                     + (stats.poisson.pmf(ks, r.lam_a)[k + 1] if k + 1 < 16 else 0))
                  for k in range(1, 15)))
        for r in pred.itertuples()
    ]

    return {
        "_provenance": {
            "script": "model/export_fitted.py",
            "fit": "model/fit_nhl_walkforward.py",
            "holdout_season": nhlwf.HOLDOUT_SEASON,
            "graded_once": True,
            "spent_season_not_used": nhlwf.SPENT_SEASON,
            "hyperparameter_trials": 0,
            "hyperparameters_fixed_a_priori": {"k": nhlwf.K},
            "why_no_tuning": ("only two NHL seasons are usable and one is "
                              "already spent; searching hyperparameters on the "
                              "only ungraded season and then grading on it is "
                              "how a result gets manufactured"),
            "metric": "mean log-likelihood of the realised scoreline",
            "n_holdout": int(len(pred)),
        },
        "inputs": [
            f"data/raw/sportsdataverse/nhl_{y}.parquet"
            for y in (nhlwf.HOLDOUT_SEASON, nhlwf.SPENT_SEASON)
        ],
        "base_log_rate_home": round(float(state["base_log_rate_home"]), ND),
        "base_log_rate_away": round(float(state["base_log_rate_away"]), ND),
        "k": state["k"],
        "holdout_grade": {
            "model": round(float(ll_model.mean()), ND),
            "baseline": round(float(ll_base.mean()), ND),
            "paired_gain": round(float(d.mean()), ND),
            "standard_error": round(se, ND),
            "t": round(t, 2),
            "supported": bool(t > 2),
        },
        "closeness_check": {
            "predicted_one_goal_rate": round(float(np.mean(pred_one)), 4),
            "actual_one_goal_rate": round(float((margins == 1).mean()), 4),
            "note": CLOSENESS_NOTE,
        },
        "joint_distribution_warning": JOINT_WARNING,
        "attack": _round(state["attack"]),
        "defence": _round(state["defence"]),
    }


def export_nba() -> dict:
    art = json.loads((ROOT / "model" / "nba_walkforward_results.json").read_text())
    hp = art["chosen_hyperparameters"]
    seasons = tuple(art["_provenance"]["tune_seasons"]) + (
        art["_provenance"]["holdout_season"],
    )
    df = load_nba(seasons)
    state: dict = {}
    pred = nbawf.walk_forward(df, hp["k"], hp["home_adv"], hp["carryover"],
                              state=state)

    # SIGMA COMES FROM THE TUNE SEASONS, NOT THE HOLDOUT, and the first
    # version of this line took it from the holdout. It reproduced every
    # rating exactly and quietly changed sigma from 14.1666 to 12.9903, an 8%
    # move in the number every NBA spread price divides by -- because a sigma
    # estimated on the graded season is the graded season being used to fit.
    # The comparison against the committed artifact is what caught it, which
    # is the entire argument for having a generator at all.
    _, sd, _ = nbawf.score(pred, art["_provenance"]["tune_seasons"])

    out = {
        "_provenance": {
            "script": "model/export_fitted.py",
            "fit": "model/fit_nba_walkforward.py",
            **{k: v for k, v in art["_provenance"].items() if k != "script"},
        },
        "inputs": [f"data/raw/sportsdataverse/nba_{y}.parquet" for y in seasons],
        "hyperparameters": hp,
        "sigma_constant": round(float(sd), 4),
        "sigma_varies_with_spread": False,
        "sigma_note": (
            "a sigma varying with predicted spread was tested and REJECTED on "
            "a separate holdout (t = -6.48). ADR 0005 predicted the opposite; "
            "see that record for why the test does not refute it -- these "
            "ratings separate games across 0.9 to 8.6 points where market "
            "spreads reach the high teens, so the range that would show a "
            "spread-dependent sigma is not in the data yet."
        ),
        "holdout_grade": art["holdout_grade"],
        "multiple_testing": art["multiple_testing"],
        "ratings": _round(state["ratings"], 4),
    }
    return out


def export_nhl_rules() -> dict:
    """The goalie-pull and overtime layers, graded once on 2022-2023.

    The pull table is re-measured from the tune seasons rather than copied out
    of the results file, so this reproduces it rather than transcribing it.
    """
    from model.fit_nhl_rules import (
        HOLDOUT, MAX_LEAD, TUNE, load, measure_overtime, measure_pull_table,
    )

    res = json.loads((ROOT / "model" / "nhl_rules_results.json").read_text())
    tune = load(TUNE)
    table = measure_pull_table(tune, max_lead=MAX_LEAD)
    ot_home = measure_overtime(tune)

    # CURRENT NO-PULL RATINGS, so the layer is usable and not merely graded.
    # These are NOT a new claim. The rating method -- walk forward within
    # season at k = 0.03 -- was graded separately on its own holdout; this
    # applies that graded method to the most recent season with goal-level
    # data to get ratings a live slate can be priced from. Producing ratings
    # is not grading, and no season is spent here.
    #
    # They are also a DIFFERENT OBJECT from the ratings in nhl_fitted.json,
    # which are fitted to final scores and therefore already contain the
    # empty-net goals the pull layer adds. Composing with those would count
    # them twice, which is why NHLModel refuses to.
    latest = max(
        int(f.stem.split("_")[1]) for f in (ROOT / "data" / "raw" / "nhl").glob(
            "goals_20*.parquet")
    )
    current = load((latest,))
    st: dict = {}
    nhlwf.walk_forward(
        current.rename(columns={"home_score": "_fh", "away_score": "_fa",
                                "nopull_h": "home_score",
                                "nopull_a": "away_score"}),
        state=st,
    )

    return {
        "_provenance": {
            "script": "model/export_fitted.py",
            "fit": "model/fit_nhl_rules.py",
            **{k: v for k, v in res["_provenance"].items() if k != "script"},
        },
        "current_ratings": {
            "season": latest,
            "basis": "no-pull regulation scores -- sixty minutes with both "
                     "goalies on the ice",
            "method": "walk forward within season, k = 0.03, as graded in "
                      "model/fit_nhl_walkforward.py",
            "not_a_new_grade": "the METHOD was graded; these are that method "
                               "applied to the latest season to produce "
                               "ratings, which spends no holdout",
            "base_log_rate_home": round(float(st["base_log_rate_home"]), ND),
            "base_log_rate_away": round(float(st["base_log_rate_away"]), ND),
            "attack": _round(st["attack"]),
            "defence": _round(st["defence"]),
        },
        "inputs": (
            [f"data/raw/nhl/nhl_{y}.parquet" for y in TUNE + HOLDOUT]
            + [f"data/raw/nhl/goals_{y}.parquet" for y in TUNE + HOLDOUT]
        ),
        "holdout_grade": res["holdout_grade"],
        "decomposition": res["decomposition"],
        "margin_shape": res["margin_shape"],
        "overtime_home_win_prob": round(float(ot_home), 4),
        "max_lead": MAX_LEAD,
        "pull_table": {
            str(lead): {f"{i},{j}": round(float(p), 6)
                        for (i, j), p in sorted(row.items())}
            for lead, row in sorted(table.items())
        },
        "what_this_does_not_model": (
            "TIMING. The pull layer is a per-game table, not a hazard over "
            "the closing minutes, so it cannot tell a pull with ninety "
            "seconds left from one with three minutes left. It also "
            "conditions on the NO-PULL margin, which is not what a coach "
            "sees -- they see the score as played, which differs whenever an "
            "earlier empty-net goal has already landed. Both are stated "
            "because they are the first things to revisit, not because they "
            "are believed harmless."
        ),
    }


def main() -> int:
    for name, build in (("nhl_fitted.json", export_nhl),
                        ("nba_fitted.json", export_nba),
                        ("nhl_rules.json", export_nhl_rules)):
        art = build()
        dest = DATA / name
        before = dest.read_text() if dest.exists() else None
        text = json.dumps(art, indent=2) + "\n"
        dest.write_text(text)
        print(f"{name}: {'unchanged' if text == before else 'rewritten'} "
              f"(t = {art['holdout_grade']['t']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
