#!/usr/bin/env python3
"""Does modelling the two NHL rules beat modelling the final score directly?

THE QUESTION, STATED BEFORE IT IS ANSWERED
The shipped NHL model fits Poisson rates to FINAL scores and prices from them.
ADR 0007 says that is the wrong object, because a final score is a no-pull
sixty minutes, plus a goalie pull that fires conditional on the score, plus a
tie-break that awards exactly one goal. This script tests whether composing
those layers predicts the realised final scoreline better than fitting the
final score directly.

Both models predict THE SAME THING -- the final home and away scores -- so the
comparison is a paired log-likelihood difference and a t statistic on it.

WHAT IS HELD FIXED SO THE TEST IS ABOUT THE LAYER
The rating scheme. Both sides walk forward within season with the same k =
0.03 that the shipped fit uses, fixed a priori. The only difference is what
gets rated (final scores against no-pull scores) and what the rates are then
pushed through. If the layer wins, it is the layer.

HOLDOUT ACCOUNTING, FIXED BEFORE RUNNING
  tune     2016-2021   6,952 games   pull table and overtime split measured here
  holdout  2022-2023   2,624 games   graded ONCE

These are the seasons pulled from the league's own API in ADR 0007, and
nothing has ever been graded on them. The sportsdataverse 2024 and 2025
seasons stay spent and stay out. The tune and holdout seasons were chosen as
the earliest six and latest two BEFORE any model was run, which matters
because the later seasons are the ones with the most pull activity and
choosing them afterwards would be choosing the result.

2024 AND 2025 EXIST AND ARE DELIBERATELY NOT IN THIS GRADE. They were pulled
from the same API afterwards, to produce current ratings. Folding them into
the holdout after seeing the result would be enlarging a test set because the
answer was liked, and the answer here was liked very much. They are available
for the NEXT question instead.

ZERO HYPERPARAMETER TRIALS. k is inherited, the pull table is an empirical
frequency table with no smoothing parameter, and the overtime split is a
single measured proportion. There is no search here, so there is no noise
ceiling to clear.

Writes model/nhl_rules_results.json. The shippable artifact, data/nhl_rules.json,
comes from model/export_fitted.py, which is the one generator for everything
under data/ -- see the note in that file about artifacts that shipped with no
producer.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coverline.leagues.nhl.rules import (  # noqa: E402
    GoaliePullLayer, NHLFinalScoreDistribution, OvertimeLayer,
)
from model.fit_nhl_walkforward import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhl"
OUT = ROOT / "model" / "nhl_rules_results.json"

TUNE = (2016, 2017, 2018, 2019, 2020, 2021)
HOLDOUT = (2022, 2023)

#: Inherited from the shipped fit, not searched here.
K = 0.03

#: Leads beyond this share the tail row. Pull goals at four or more appear in
#: 7% of games; splitting the tail finer estimates noise.
MAX_LEAD = 4


def load(seasons) -> pd.DataFrame:
    """One row per game: final score, regulation score, no-pull regulation score.

    game_date is made unique per game so that every sort in this script and in
    walk_forward is a total order. Two frames of the same games must line up
    row for row, and pandas' default sort is not stable.
    """
    fin = pd.concat([pd.read_parquet(RAW / f"nhl_{y}.parquet") for y in seasons],
                    ignore_index=True)
    gl = pd.concat([pd.read_parquet(RAW / f"goals_{y}.parquet") for y in seasons],
                   ignore_index=True)

    gl = gl.sort_values(["game_id", "period", "time_in_period"])
    scorer_home = gl.groupby("game_id").home_score_after.diff().fillna(
        gl.home_score_after
    ) > 0
    code = gl.situation_code.fillna("")
    net_empty = (code.str[0] == "0") | (code.str[-1] == "0")
    gl = gl.assign(scorer_home=scorer_home, net_empty=net_empty)

    reg = gl[gl.period_type == "REG"]
    reg = reg.assign(h=reg.scorer_home.astype(int),
                     a=(~reg.scorer_home).astype(int))
    full = reg.groupby("game_id")[["h", "a"]].sum().rename(
        columns={"h": "reg_h", "a": "reg_a"})
    pulled = reg[reg.net_empty].groupby("game_id")[["h", "a"]].sum().rename(
        columns={"h": "pull_h", "a": "pull_a"})

    df = fin.set_index("game_id").join(full).join(pulled)
    df[["reg_h", "reg_a", "pull_h", "pull_a"]] = (
        df[["reg_h", "reg_a", "pull_h", "pull_a"]].fillna(0).astype(int)
    )
    df["nopull_h"] = df.reg_h - df.pull_h
    df["nopull_a"] = df.reg_a - df.pull_a
    df = df.reset_index()
    df["game_date"] = df.game_date.astype(str) + "#" + df.game_id.astype(str)
    return df.sort_values("game_date").reset_index(drop=True)


def measure_pull_table(df: pd.DataFrame, max_lead: int = MAX_LEAD) -> dict:
    """Empirical P(leader gain, trailer gain | absolute no-pull margin).

    No smoothing. Every row here has at least several hundred games behind it,
    and a smoothing constant would be a hyperparameter this design does not
    have room for.
    """
    m = df.nopull_h - df.nopull_a
    lead_home = m > 0
    gain_leader = np.where(lead_home, df.pull_h, df.pull_a)
    gain_trailer = np.where(lead_home, df.pull_a, df.pull_h)
    key = np.minimum(np.abs(m), max_lead)

    counts: dict[int, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int))
    for L, gl_, gt in zip(key, gain_leader, gain_trailer):
        counts[int(L)][(int(gl_), int(gt))] += 1

    table: dict[int, dict[tuple[int, int], float]] = {}
    for L, row in counts.items():
        n = sum(row.values())
        table[L] = {k: v / n for k, v in row.items()}
    return table


def measure_overtime(df: pd.DataFrame) -> float:
    ot = df[df.last_period_type != "REG"]
    return float((ot.home_score > ot.away_score).mean())


def rate(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    """Walk forward within each season on the chosen score columns."""
    out = []
    for _, season in df.groupby("season", sort=True):
        # Select BEFORE renaming. Renaming nopull_h to home_score while
        # home_score is still present leaves two columns of that name, and
        # walk_forward then multiplies Series instead of floats.
        s = season[["game_date", "home_team_abbr", "away_team_abbr",
                    home_col, away_col]].rename(
            columns={home_col: "home_score", away_col: "away_score"})
        out.append(walk_forward(s, k=K).assign(season=season.season.iloc[0]))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    tune, hold = load(TUNE), load(HOLDOUT)
    print(f"tune {len(tune)} games {TUNE}   holdout {len(hold)} games {HOLDOUT}")

    table = measure_pull_table(tune)
    ot_home = measure_overtime(tune)
    print(f"overtime home win probability (tune) {ot_home:.4f}")
    for L in sorted(table):
        row = sorted(table[L].items(), key=lambda kv: -kv[1])[:3]
        print(f"  lead {L}: " + "  ".join(f"{k}={v:.3f}" for k, v in row))

    pull = GoaliePullLayer(table=table, max_lead=MAX_LEAD)
    overtime = OvertimeLayer(home_win_prob=ot_home)

    # Baseline: rate final scores, score final scores. What ships today.
    base = rate(hold, "home_score", "away_score")
    ll_base = np.asarray(stats.poisson.logpmf(base.hs, base.lam_h)
                         + stats.poisson.logpmf(base.as_, base.lam_a))

    # Rules: rate NO-PULL scores, push through both layers, score final scores.
    rules = rate(hold, "nopull_h", "nopull_a")
    finals = hold.sort_values(["season", "game_date"]).reset_index(drop=True)
    assert len(rules) == len(finals), "rated rows and final scores misaligned"

    ll_rules = np.empty(len(finals))
    for i, (r, f) in enumerate(zip(rules.itertuples(), finals.itertuples())):
        d = NHLFinalScoreDistribution(r.lam_h, r.lam_a, pull, overtime)
        p = d.joint()[int(f.home_score), int(f.away_score)]
        ll_rules[i] = np.log(max(p, 1e-300))

    # rate() walks seasons in sorted order and walk_forward sorts on a
    # game_date made unique per game, so base, rules and finals are the same
    # games in the same order. Asserted rather than trusted, because a silent
    # misalignment here would compare each game to a different game and still
    # produce a plausible t.
    assert len(base) == len(ll_rules)
    assert (base.hs.to_numpy() == finals.home_score.to_numpy()).all(), (
        "baseline rows and final scores are misaligned"
    )
    diff = ll_rules - ll_base
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    t = float(diff.mean() / se)

    # DECOMPOSITION, because a t of 40 is a reason for suspicion and not for
    # celebration. Most of the gain is not modelling -- it is knowing the
    # rules of hockey. The baseline puts about a sixth of its probability on a
    # tied final score, which the league does not permit. Applying ONLY the
    # overtime rule to the baseline's own rates isolates how much of the win
    # is that, and how much is the goalie-pull layer earning its place.
    n_grid = 22
    ks = np.arange(n_grid)
    ll_ot_only = np.empty(len(finals))
    for i, (lh, la) in enumerate(zip(base.lam_h, base.lam_a)):
        j = np.outer(stats.poisson.pmf(ks, lh), stats.poisson.pmf(ks, la))
        tie = np.diag(j).copy()
        np.fill_diagonal(j, 0.0)
        for k in range(n_grid - 1):
            j[k + 1, k] += tie[k] * ot_home
            j[k, k + 1] += tie[k] * (1.0 - ot_home)
        j /= j.sum()
        ll_ot_only[i] = np.log(max(
            j[min(int(finals.home_score.iloc[i]), n_grid - 1),
              min(int(finals.away_score.iloc[i]), n_grid - 1)], 1e-300))

    def _paired(d: np.ndarray) -> dict:
        s = float(d.std(ddof=1) / np.sqrt(len(d)))
        return {"gain": round(float(d.mean()), 5), "standard_error": round(s, 5),
                "t": round(float(d.mean() / s), 2)}

    decomposition = {
        "overtime_rule_alone": _paired(ll_ot_only - ll_base),
        "both_layers": _paired(ll_rules - ll_base),
        "goalie_pull_layer_over_and_above": _paired(ll_rules - ll_ot_only),
        "share_of_gain_that_is_the_overtime_rule": round(
            float((ll_ot_only - ll_base).mean() / (ll_rules - ll_base).mean()), 4
        ),
        "note": ("the overtime rule is most of it, and it is a rule anyone can "
                 "look up rather than a modelling result -- the shipped model "
                 "was assigning real probability to an outcome the league "
                 "forbids. The goalie-pull layer is the smaller, genuinely "
                 "modelled part, and it clears its own gate separately."),
    }

    # Shape check: does the composed model reproduce the non-monotonicity?
    shape_model = np.zeros(6)
    for r in rules.itertuples():
        d = NHLFinalScoreDistribution(r.lam_h, r.lam_a, pull, overtime)
        for j in range(6):
            shape_model[j] += (d.margin_pmf(j) if j == 0
                               else d.margin_pmf(j) + d.margin_pmf(-j))
    shape_model /= len(rules)
    am = (finals.home_score - finals.away_score).abs()
    shape_actual = np.array([float((am == j).mean()) for j in range(6)])

    art = {
        "_provenance": {
            "script": "model/fit_nhl_rules.py",
            "tune_seasons": list(TUNE),
            "holdout_seasons": list(HOLDOUT),
            "graded_once": True,
            "hyperparameter_trials": 0,
            "k_inherited_from": "model/fit_nhl_walkforward.py",
            "k": K,
            "metric": "mean log-likelihood of the realised final scoreline",
            "baseline": "Poisson rates fitted to FINAL scores -- what ships today",
            "n_holdout": int(len(diff)),
            "source": "api-web.nhle.com via model/ingest/",
        },
        "holdout_grade": {
            "rules_model": round(float(ll_rules.mean()), 5),
            "baseline": round(float(ll_base.mean()), 5),
            "paired_gain": round(float(diff.mean()), 5),
            "standard_error": round(se, 5),
            "t": round(t, 2),
            "supported": bool(t > 2),
        },
        "decomposition": decomposition,
        "margin_shape": {
            str(j): {"model": round(float(shape_model[j]), 4),
                     "actual": round(float(shape_actual[j]), 4)}
            for j in range(6)
        },
        "overtime_home_win_prob": round(ot_home, 4),
        "pull_table": {str(L): {f"{i},{j}": round(p, 6)
                                for (i, j), p in sorted(row.items())}
                       for L, row in sorted(table.items())},
        "max_lead": MAX_LEAD,
    }
    OUT.write_text(json.dumps(art, indent=2) + "\n")
    print(json.dumps(art["holdout_grade"], indent=1))
    print("decomposition:")
    for k in ("overtime_rule_alone", "goalie_pull_layer_over_and_above"):
        print(f"  {k:36s} {decomposition[k]}")
    print("  overtime rule is %.1f%% of the gain"
          % (100 * decomposition["share_of_gain_that_is_the_overtime_rule"]))
    print("margin shape (model vs actual):")
    for j in range(6):
        print(f"  |m|={j}  {shape_model[j]:.4f}  {shape_actual[j]:.4f}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
