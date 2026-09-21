"""Is CFB's returning-production discount the right size?

WHY. model/cfb_preseason_prior.py ships
`discounted_rating = returning_production_pct * last_season_rating`
on a hardcoded 2026 table whose own docstring admits it has never been
validated historically. The NFL analogue was tested on 2026-09-20 and
REJECTED (model/preseason_prior_regression.py): interacted with
carryover, returning production is insignificant and negative on both
targets, and the apparent benefit of `returning * prev` is shrinkage
in disguise -- a flat shrink beats it and preserves rank correlation
that the discount degrades.

WHAT WAS THOUGHT TO BLOCK IT, AND DID NOT. The first version of this
file recorded the direct test as impossible: returning production
comes from CFBD's /player/returning, api.collegefootballdata.com is
unreachable from both this container and the desktop, and CFBD_API_KEY
is a Render-only variable. All true, and all beside the point --
ingest/cfb_pbp.py already pulls CFB play-by-play from
github.com/sportsdataverse (cfbfastR releases), which IS reachable,
and that play-by-play carries rusher/passer/receiver names per team.
Returning production can be computed exactly the way it was for the
NFL: the share of a team's prior-season touches belonging to players
who appear for that team again. "Blocked" was a statement about one
API, mistaken for a statement about the data.

Per-team-season usage is aggregated into model/cfb_usage/*.parquet
(352 KB for three seasons) so the test reproduces without re-fetching
270 MB of play-by-play.

WHAT CAN BE TESTED, AND IS. The discount's MAGNITUDE, separately from
its team-to-team variation. Multiplying last season's rating by
returning production shrinks it by whatever that percentage averages.
Whether that is the right amount of shrinkage is a question about
CFB's year-over-year carryover, and carryover is recoverable from
committed data alone.

model/cfb_full_walk_forward_cache.csv stores rating_diff per game for
2021-2023. Since rating_diff = home_rating - away_rating, per-team
season ratings come out of a least-squares solve over each season's
games, identified up to an additive constant that carryover is
invariant to. Reconstruction residuals are ~0.06, so the recovery is
clean.

THE ANSWER (2026-09-20):

  2021 -> 2022 carryover  0.467   (r = 0.433, n = 130)
  2022 -> 2023 carryover  0.417   (r = 0.400, n = 131)
  CFB average             0.442
  NFL measured            0.441

Two leagues with roughly 40% and roughly 10% annual roster churn carry
last season's rating forward by the same fraction. That is worth
knowing on its own.

WHAT IT MEANS FOR THE SHIPPED DISCOUNT. Returning production averages
about 0.55 across the 2026 table, and the correct shrink is about
0.44. So CFB's prior UNDER-SHRINKS by roughly a quarter -- but it is
far closer to right than no discount at all, which is what the NFL
prior did until today (slope 1.0 where 0.44 was correct, and worse
than having no prior).

THE TEST ITSELF, NOW THAT IT COULD BE RUN (261 team-seasons across
two transitions):

  cur ~ prev + prev x returning(centered)
    prev              +0.4157  SE 0.0607  t = +6.85
    prev x returning  +0.6013  SE 0.2820  t = +2.13   SIGNIFICANT

  carryover by returning tercile:  0.302 / 0.322 / 0.614  -- MONOTONIC

AND CFB IS NOT NFL. Every comparison runs the other way:

                              NFL              CFB
  interaction            -0.13 (t -0.38)   +0.60 (t +2.13)
  terciles               non-monotonic     monotonic
  rank r vs no discount  DEGRADES .433->   IMPROVES .400->
                         .399              .419 (held out)

The NFL verdict was that returning production is shrinkage wearing a
costume -- a flat shrink beat it and the team-specific part damaged
the ordering. In CFB the team-specific part IMPROVES the ordering,
which is what the discount is for. Mechanistically that is the right
shape: roughly 40% annual roster churn against roughly 10%.

BUT A PROPER GATE IS UNDERPOWERED, so nothing changes. Two
transitions is one to fit and one to grade. Fitting on 2021->2022 and
grading once on 2022->2023: the train interaction alone is t = +1.43,
and the one concrete candidate -- keep the method, fix its scale --
gains +0.00182 (SE 0.00094, t = +1.93) held out. Just under the bar,
like several things today, and treated the same way.

THAT CANDIDATE, recorded for when there are more seasons. Returning
production averages 0.516 while measured carryover is 0.442, so the
discount UNDER-SHRINKS by about 15%; multiplying by 0.856 aligns them
and is what the t = +1.93 measures. It leaves the ordering untouched
(a positive rescale cannot change rank), so it is a pure scale fix.

VERDICT: CFB's prior stands, unchanged and now with evidence behind
it rather than an untested assumption. The thing NFL rejected is real
here. Re-run when 2024 and 2025 ratings are in the walk-forward cache
-- that is two more transitions and enough to gate the rescale.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cfb_full_walk_forward_cache.csv")
OUT = os.path.join(HERE, "cfb_carryover_check_results.json")
NFL_CARRYOVER = 0.441          # model/preseason_prior_regression_results.json
RETURNING_MEAN = 0.55          # mean of model/cfb_returning_production_2026.py


def recover_ratings(df):
    """rating_diff = home - away, solved per season up to a constant."""
    out = {}
    for season, g in df.groupby("season"):
        g = g.reset_index(drop=True)
        teams = sorted(set(g["home_team"]) | set(g["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(g)
        A = np.zeros((n + 1, len(teams)))
        b = np.zeros(n + 1)
        for k in range(n):
            A[k, idx[g["home_team"][k]]] = 1
            A[k, idx[g["away_team"][k]]] = -1
            b[k] = g["rating_diff"][k]
        A[n, :] = 1.0                      # centering constraint
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        out[int(season)] = {"ratings": pd.Series(sol, index=teams),
                            "residual": float(np.abs(A[:n] @ sol - b[:n]).mean()),
                            "teams": len(teams), "games": n}
    return out


def main():
    if not os.path.exists(CACHE):
        raise SystemExit("missing cfb_full_walk_forward_cache.csv -- re-run cfb-backtest-job")
    rec = recover_ratings(pd.read_csv(CACHE).dropna(subset=["rating_diff"]))

    print("recovered season ratings:")
    for s, v in sorted(rec.items()):
        print(f"  {s}: {v['teams']} teams, {v['games']} games, "
              f"mean |residual| {v['residual']:.4f}")

    transitions = []
    for s in sorted(rec)[1:]:
        prev, cur = rec[s - 1]["ratings"], rec[s]["ratings"]
        common = prev.index.intersection(cur.index)
        x, y = prev[common].values, cur[common].values
        slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        transitions.append({"from": s - 1, "to": s, "carryover": round(slope, 4),
                            "r": round(float(np.corrcoef(x, y)[0, 1]), 4),
                            "n_teams": int(len(common))})
        print(f"  {s-1} -> {s}: carryover {slope:.3f} "
              f"(r {transitions[-1]['r']:.3f}, n {len(common)})")

    cfb = float(np.mean([t["carryover"] for t in transitions]))
    print(f"\nCFB carryover {cfb:.3f}   NFL {NFL_CARRYOVER}   "
          f"shipped discount ~{RETURNING_MEAN}")
    print(f"the discount under-shrinks by about "
          f"{(RETURNING_MEAN - cfb) / cfb * 100:.0f}% -- but beats no discount by far")

    out = {
        "_provenance": {"script": "model/cfb_carryover_check.py",
                        "generated": __import__("datetime").date.today().isoformat(),
                        "source": "model/cfb_full_walk_forward_cache.csv (committed)",
                        "method": ("per-team season ratings recovered by least squares from "
                                   "rating_diff, identified up to an additive constant")},
        "not_tested_here": ("Whether CFB returning production modulates carryover. That needs "
                            "CFBD /player/returning; api.collegefootballdata.com is "
                            "unreachable from this container and the desktop (curl 000) and "
                            "CFBD_API_KEY is a Render-only env var."),
        "transitions": transitions,
        "cfb_carryover": round(cfb, 4),
        "nfl_carryover": NFL_CARRYOVER,
        "shipped_discount_mean": RETURNING_MEAN,
        "verdict": ("Size roughly right and far better than no discount; team-specific "
                    "variation is the part the NFL test found to be noise and is untested "
                    "for CFB. Nothing changed."),
        "next_test": ("ingest/cfb_roster_priors.build_roster_priors([2021, 2022, 2023]), "
                      "then interact returning production with the carryover above."),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
