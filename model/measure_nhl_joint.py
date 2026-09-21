#!/usr/bin/env python3
"""What an NHL final score actually is, measured before anything is fitted.

WHY THIS EXISTS
The shipped NHL package says the joint distribution is unmodelled because the
two scores are negatively correlated at -0.14 and "real games stay closer than
independent rates allow." Half of that is right and the mechanism is wrong,
which matters because the wrong mechanism points at the wrong repair: a
correlation parameter, fitted once, on a quantity that is drifting.

Negative covariance makes margins MORE dispersed, not less:
Var(H - A) = Var(H) + Var(A) - 2Cov(H, A). So the stated mechanism predicts
FEWER one-goal games, and the observed problem is that there are MORE of them.
The sign of the correlation is real; the story attached to it is backwards.

This script measures the three things that are actually happening, none of
which is a correlation parameter:

  1. THE OVERTIME RULE, which is deterministic. No NHL game ends tied and
     every game decided after regulation ends at a margin of exactly one. So
     the entire tie mass of the regulation distribution -- about 22% -- is
     moved onto plus and minus one by a rule, not by a scoring process.

  2. EMPTY-NET GOALS, which are conditional on the score. A team down two
     pulls its goalie and rarely comes back, so two-goal games drain into
     three. A team down one pulls and often ties, vanishing into overtime and
     back out at one. The result is a NON-MONOTONE margin distribution with
     more three-goal games than two-goal games, across exactly the 1.5 line
     the puck line is priced on.

  3. DRIFT. Whatever the number is, it is not stationary: the margin variance
     ratio and the score correlation both grow across 2016-2025, which is what
     a league pulling goalies earlier every year looks like. A fitted
     correlation is therefore a season-specific constant with no reason to
     persist, while a goalie-pull layer is a mechanism that can be tracked.

DESCRIPTIVE, NOT GRADED. Nothing here is fitted and nothing is held out,
because nothing here is a prediction -- these are properties of the data. The
holdout seasons stay unspent for whatever model this motivates.

Writes model/nhl_joint_structure.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "model" / "nhl_joint_structure.json"


def load_finals() -> pd.DataFrame:
    """Both sources, tagged. Agreement between them is the point."""
    frames = []
    for f in sorted((RAW / "nhl").glob("nhl_20*.parquet")):
        d = pd.read_parquet(f)
        d["source"] = "nhle-api"
        frames.append(d)
    for y in (2024, 2025):
        f = RAW / "sportsdataverse" / f"nhl_{y}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d = d[d.game_type == "R"] if "game_type" in d.columns else d
        d = d.dropna(subset=["home_score", "away_score"])
        d = d[["season", "home_team_abbr", "away_team_abbr",
               "home_score", "away_score"]].copy()
        d["home_score"] = d.home_score.astype(int)
        d["away_score"] = d.away_score.astype(int)
        d["last_period_type"] = None  # sportsdataverse does not carry it
        d["source"] = "sportsdataverse"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["margin"] = df.home_score - df.away_score
    return df


def load_goals() -> pd.DataFrame | None:
    files = sorted((RAW / "nhl").glob("goals_20*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def dispersion(d: pd.DataFrame) -> dict:
    """Margin variance against what independent Poisson at the same means
    would give. A ratio above one is over-dispersion relative to independence,
    which is what negative covariance produces."""
    h = d.home_score.to_numpy(float)
    a = d.away_score.to_numpy(float)
    m = h - a
    indep = h.mean() + a.mean()
    return {
        "n": int(len(d)),
        "corr": round(float(np.corrcoef(h, a)[0, 1]), 4),
        "var_margin": round(float(m.var()), 4),
        "var_margin_if_independent": round(float(indep), 4),
        "dispersion_ratio": round(float(m.var() / indep), 4),
    }


def margin_shape(d: pd.DataFrame) -> dict:
    """Empirical |margin| against independent Poisson at the pooled means."""
    h = d.home_score.to_numpy(float)
    a = d.away_score.to_numpy(float)
    emp = d.margin.abs().value_counts(normalize=True)
    sk = {k: float(stats.skellam.pmf(k, h.mean(), a.mean())) for k in range(-9, 10)}
    out = {}
    for j in range(6):
        pred = sk[0] if j == 0 else sk[j] + sk[-j]
        out[str(j)] = {
            "empirical": round(float(emp.get(j, 0.0)), 4),
            "independent_poisson": round(pred, 4),
        }
    return out


def overtime_rule(d: pd.DataFrame) -> dict:
    """Is the tie-break deterministic, or only usually one goal?"""
    known = d[d.last_period_type.notna()]
    ot = known[known.last_period_type != "REG"]
    return {
        "games_with_period_type": int(len(known)),
        "ot_or_so_share": round(float((known.last_period_type != "REG").mean()), 4),
        "ties_in_final_scores": int((known.margin == 0).sum()),
        "ot_games_not_decided_by_exactly_one": int((ot.margin.abs() != 1).sum()),
        "deterministic": bool(
            (known.margin == 0).sum() == 0 and (ot.margin.abs() != 1).sum() == 0
        ),
    }


def empty_net(goals: pd.DataFrame, finals: pd.DataFrame) -> dict:
    """The layer that cannot be recovered from final scores.

    Strips every empty-net goal off the final score and re-measures the shape.
    If the non-monotonicity is an empty-net artefact it should disappear here,
    and if it does not, the story is wrong again and should be said so.
    """
    goals = goals.copy()
    # ANY modifier containing empty-net, not equality with it. The league also
    # files awarded-empty-net and own-goal-empty-net, 16 goals here, and an
    # equality test drops them.
    goals["is_en"] = goals.goal_modifier.fillna("").str.contains("empty-net")

    g = goals.sort_values(["game_id", "period", "time_in_period"])
    # Diff WITHIN the game. A flat diff across the frame charges each game's
    # first goal against the previous game's last one, which silently
    # mislabels roughly one goal in six as the wrong team's.
    scorer_is_home = g.groupby("game_id").home_score_after.diff().fillna(
        g.home_score_after
    ) > 0

    # situation_code digits are goalie-present flags, away first and home last.
    # A second route to the same fact, so the league's flag can be checked
    # rather than trusted.
    #
    # The test is whether the net the SCORER SHOT AT was empty, not whether
    # either net was. Testing "either" disagrees with the flag on 846 of
    # 21,027 goals, and 830 of those are the PULLED team scoring into a
    # guarded net -- a goal scored while your own net is empty is not an
    # empty-net goal. With both corrections the two fields agree on every
    # goal, 1.0000, which is why the flag can now be used without hedging.
    codes = g.situation_code.fillna("")
    four = codes.str.len() == 4
    away_net_empty = four & (codes.str[0] == "0")
    home_net_empty = four & (codes.str[-1] == "0")
    shot_at_empty = np.where(scorer_is_home, away_net_empty, home_net_empty)
    mismatch = (g.is_en.to_numpy() != shot_at_empty) & four.to_numpy()
    agree = float(1.0 - mismatch.sum() / four.sum())
    # WHICH FIELD IS WRONG, on the handful that still disagree. Twelve of the
    # thirteen are the league's own flag saying empty-net while the situation
    # code says both goalies are on the ice, at 15:04 to 19:53 of the third
    # period -- which is when empty-net goals happen and when goalies are not
    # on the ice. The cross-check validated the flag and found errors in the
    # OTHER field, which is the opposite of what it was built to test and is
    # the reason the flag is the one used downstream.
    mism = g[mismatch]
    late = mism[(mism.period >= 3) & (mism.time_in_period >= "15:00")]
    # Score state BEFORE the goal, from the running score after it.
    g = g.assign(
        pre_home=g.home_score_after - scorer_is_home.astype(int),
        pre_away=g.away_score_after - (~scorer_is_home).astype(int),
        scorer_is_home=scorer_is_home,
    )
    en = g[g.is_en]
    lead = np.where(en.scorer_is_home, en.pre_home - en.pre_away,
                    en.pre_away - en.pre_home)
    by_lead = pd.Series(lead).value_counts(normalize=True).sort_index()

    # Strip EN goals off the final scores.
    strip = (
        g[g.is_en]
        .assign(h=lambda x: x.scorer_is_home.astype(int),
                a=lambda x: (~x.scorer_is_home).astype(int))
        .groupby("game_id")[["h", "a"]].sum()
    )
    fin = finals[finals.game_id.notna()].set_index("game_id") if "game_id" in finals else None
    stripped = None
    if fin is not None:
        # LEFT, not inner. An inner join keeps only games that had an
        # empty-net goal, which measures the shape of empty-net games rather
        # than the shape of the league. The first version did that and
        # reported 42% of games at a two-goal margin.
        j = fin.join(strip, how="left")
        if len(j):
            d = j.assign(
                home_score=j.home_score - j.h.fillna(0).astype(int),
                away_score=j.away_score - j.a.fillna(0).astype(int),
            )
            d["margin"] = d.home_score - d.away_score
            stripped = margin_shape(d)

    # Per season, because the layer is not stationary and a pooled number
    # would hide that. Time is minutes into the third period, where almost
    # all of these happen.
    per_season = {}
    if "season" in finals.columns:
        gm = finals.set_index("game_id").season.to_dict()
        g2 = g.assign(season=g.game_id.map(gm))
        mins = g2.time_in_period.str.split(":", expand=True).astype(float)
        g2 = g2.assign(minute=mins[0] + mins[1] / 60.0)
        games_per = finals.groupby("season").size()
        for season, grp in g2[g2.is_en].groupby("season"):
            n_games = int(games_per.get(season, 0))
            if not n_games:
                continue
            third = grp[grp.period == 3]
            per_season[str(int(season))] = {
                "empty_net_per_game": round(len(grp) / n_games, 4),
                "mean_minute_in_third": round(float(third.minute.mean()), 2),
            }

    return {
        "goals": int(len(goals)),
        "by_season": per_season,
        "empty_net_share_of_goals": round(float(goals.is_en.mean()), 4),
        "empty_net_goals_per_game": round(
            float(goals.is_en.sum() / goals.game_id.nunique()), 4
        ),
        "flag_agrees_with_situation_code": round(agree, 5),
        "flag_code_disagreements": {
            "n": int(mismatch.sum()),
            "of": int(four.sum()),
            "flag_says_empty_code_says_not": int((mism.is_en).sum()),
            "in_the_last_five_minutes_of_the_third": int(len(late)),
            "verdict": "the situation code is the field in error, not the "
                       "flag: a goal filed as empty-net at 19:18 of the third "
                       "with a code reading both goalies on the ice is a bad "
                       "code, not a bad flag",
        },
        "lead_of_scoring_team_before_empty_net_goal": {
            str(int(k)): round(float(v), 4) for k, v in by_lead.items()
        },
        "margin_shape_with_empty_net_goals_removed": stripped,
    }


def main() -> int:
    finals = load_finals()
    api = finals[finals.source == "nhle-api"]

    report: dict = {
        "_provenance": {
            "script": "model/measure_nhl_joint.py",
            "descriptive": True,
            "graded": False,
            "note": "properties of the data, not a prediction -- nothing here "
                    "spends a holdout season",
        },
        "overtime_rule": overtime_rule(finals),
        "by_season": {},
        "pooled": {},
        "margin_shape": {},
    }

    for (src, season), d in finals.groupby(["source", "season"]):
        report["by_season"][f"{src}:{int(season)}"] = dispersion(d)

    for src, d in finals.groupby("source"):
        report["pooled"][src] = dispersion(d)
        report["margin_shape"][src] = margin_shape(d)

    # Regulation, with the overtime rule undone.
    ot = api.last_period_type != "REG"
    reg = api.copy()
    lo = api[["home_score", "away_score"]].min(axis=1)
    reg.loc[ot, "home_score"] = lo[ot]
    reg.loc[ot, "away_score"] = lo[ot]
    reg["margin"] = reg.home_score - reg.away_score
    report["regulation_only"] = dispersion(reg) | {"margin_shape": margin_shape(reg)}

    goals = load_goals()
    if goals is not None:
        seasons_done = sorted(
            int(f.stem.split("_")[1]) for f in (RAW / "nhl").glob("goals_20*.parquet")
        )
        report["empty_net"] = empty_net(
            goals, api[api.season.isin(seasons_done)]
        ) | {"seasons": seasons_done}
    else:
        report["empty_net"] = {"measured": False,
                               "reason": "no goal-level files; run model/ingest/nhl_goals.py"}

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overtime_rule"], indent=1))
    print("pooled:", json.dumps(report["pooled"], indent=1))
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
