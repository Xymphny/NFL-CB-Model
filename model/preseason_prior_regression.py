"""The preseason prior uses last season raw. It should not.

WHAT STARTED THIS. Stage 3's H1a: teams that return more of their
roster should carry last season's rating further, so the prior's
weight ought to scale with continuity. That is the hypothesis CFB's
prior already acts on -- model/cfb_preseason_prior.py ships
`discounted_rating = returning_production_pct * last_season_rating`,
on a hardcoded 2026 table whose own docstring admits it has never been
validated historically.

H1A IS REJECTED. Returning production was computed for every NFL
team-season 2017-2025 from player-week touches (carries + targets +
attempts): the share of a team's prior-season production belonging to
players who appear for that team the next year. Median 73%, range
21-99%, and the extremes are face-valid (2021 DET at 21% after the
Stafford teardown; 2017 SF at 21% in Shanahan's first year).

Interacted with carryover on the training seasons it does nothing:

  target                  prev x returning      SE      t
  point differential           -0.3026        0.3395   -0.89
  the actual VOA rating        -0.1313        0.3493   -0.38

Not significant on either, and NEGATIVE on both -- the opposite sign
to the hypothesis. Bucketed by continuity tercile the carryover slopes
run 0.432 / 0.193 / 0.464, which is non-monotonic, the shape of noise.

CFB'S FORMULA IS SHRINKAGE WEARING A COSTUME. On NFL history,
`returning * prev` does beat raw `prev` on MAE (4.74 vs 5.14) -- but
only because multiplying by ~0.7 shrinks toward the mean, which any
constant would do. A FLAT shrink does better still (4.58), and the
returning-discount degrades rank correlation (0.399 vs 0.433) because
its team-specific part is noise. The gain is the shrinkage; the
continuity is the costume. This is NFL, not CFB -- college rosters
turn over far harder and the mechanism could be real there -- but CFB
ships this untested by its own admission, and that now looks worth
testing rather than assuming.

WHAT THE INVESTIGATION FOUND INSTEAD, which is the shippable part.
Year-over-year carryover of the rating is 0.441 (SE 0.063), and
model/preseason_prior.py uses the prior RAW: `effective_prior =
prior_rating`. A slope of 1.0 where 0.44 is correct more than doubles
last season's weight.

It is worse than that. When the year-over-year correlation is below
0.5, a slope of 1.0 is worse than a slope of ZERO, and it is: on
held-out seasons the raw prior scores MAE 0.0913 against 0.0846 for
simply predicting the league mean. The shipped prior is worse than no
prior at all.

GATED, NOT ASSUMED. Coefficients fit on 2017-2022 and graded once on
2023-2025, which the fit never saw: MAE 0.0787 vs 0.0913, a 13.9%
improvement, paired gain +0.0127 (SE 0.0055, t = +2.30), better in all
three held-out seasons individually. Offense and defense are regressed
separately (0.432 and 0.348 -- defense carries over less, which is a
real and well-known fact about football) and total stays exactly
offense - defense, as it already did.

ONE THING THIS DOES NOT DO. k=2 was calibrated against the RAW prior
(model/calibrate_credibility_k.py). A better prior deserves more
weight, so k=2 is now conservative -- but re-tuning it on these same
seasons is the leak model/holdout_discipline.py exists to prevent. It
is left alone, and flagged here instead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "preseason_prior_regression_results.json")
TRAIN, HOLDOUT = range(2017, 2023), range(2023, 2026)


def season_ratings(cache=None):
    """Final-season ratings for 2016-2025. Regenerated from the pipeline
    unless a cached parquet is supplied."""
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    from ingest.nfl_pbp import load_season
    from ingest.nfl_schedules import load_schedules
    from model.ratings import (add_home_field_and_rest, add_recency_weights,
                               add_situation_buckets, compute_baselines,
                               compute_raw_voa, filter_garbage_time,
                               opponent_adjust, score_all_plays, team_ratings)
    rows = []
    for s in range(2016, 2026):
        sched, df = load_schedules(seasons=[s]), load_season(s)
        df = add_situation_buckets(df)
        df = score_all_plays(df, use_turnover_luck_adjustment=True)
        df = filter_garbage_time(df)
        df = add_home_field_and_rest(df, sched)
        df = compute_raw_voa(df, compute_baselines(df))
        df = opponent_adjust(df, iterations=3, regression=0.5)
        df = add_recency_weights(df)
        r = team_ratings(df, use_recency_weights=True)
        rows += [{"season": s, "team": t, "total_rating": float(r.loc[t, "total_rating"]),
                  "offense_voa": float(r.loc[t, "offense_voa"]),
                  "defense_voa": float(r.loc[t, "defense_voa"])} for t in r.index]
        print(f"  rated {s}", flush=True)
    return pd.DataFrame(rows)


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    se = np.sqrt(resid.var(ddof=2) / (np.var(x, ddof=1) * (len(x) - 1)))
    return float(b), float(a), float(se)


def main():
    r = season_ratings(os.environ.get("RATINGS_CACHE"))
    cols = ("total_rating", "offense_voa", "defense_voa")
    prev = r.assign(season=r["season"] + 1).rename(columns={c: "prev_" + c for c in cols})
    d = r.merge(prev, on=["season", "team"])
    tr, te = d[d["season"].isin(TRAIN)], d[d["season"].isin(HOLDOUT)]

    fits = {c: _fit(tr["prev_" + c], tr[c]) for c in cols}
    print(f"carryover fitted on {min(TRAIN)}-{max(TRAIN)} (n={len(tr)}):")
    for c, (b, a, se) in fits.items():
        print(f"  {c:>13}: {b:.3f} (SE {se:.3f})")

    bo, ao, _ = fits["offense_voa"]
    bd, ad, _ = fits["defense_voa"]
    regressed = (bo * te["prev_offense_voa"] + ao) - (bd * te["prev_defense_voa"] + ad)
    graded = {}
    for name, pred in (("raw_prev_shipped", te["prev_total_rating"]),
                       ("regressed_per_component", regressed),
                       ("league_mean_no_prior", pd.Series(
                           np.full(len(te), tr["total_rating"].mean()), index=te.index))):
        e = te["total_rating"] - pred
        graded[name] = {"mae": round(float(e.abs().mean()), 4),
                        "rmse": round(float(np.sqrt((e ** 2).mean())), 4)}

    gain = ((te["total_rating"] - te["prev_total_rating"]).abs()
            - (te["total_rating"] - regressed).abs())
    se_gain = float(gain.std(ddof=1) / np.sqrt(len(gain)))

    print(f"\ngraded once on {min(HOLDOUT)}-{max(HOLDOUT)} (n={len(te)}):")
    for k, v in graded.items():
        print(f"  {k:>24}  MAE {v['mae']:.4f}  RMSE {v['rmse']:.4f}")
    print(f"\n  paired MAE gain {gain.mean():+.4f} +/- {se_gain:.4f} "
          f"(t={gain.mean() / se_gain:+.2f})")
    print("  NOTE: the raw prior is worse than predicting the league mean. Below a")
    print("  year-over-year correlation of 0.5, a slope of 1.0 loses to a slope of 0.")

    out = {
        "_provenance": {
            "script": "model/preseason_prior_regression.py",
            "generated": __import__("datetime").date.today().isoformat(),
            "fit_seasons": [min(TRAIN), max(TRAIN)],
            "graded_seasons": [min(HOLDOUT), max(HOLDOUT)],
            "graded_once": True,
        },
        "h1a_continuity_priors": {
            "verdict": "REJECTED",
            "interaction_on_point_differential": {"estimate": -0.3026, "se": 0.3395, "t": -0.89},
            "interaction_on_voa_rating": {"estimate": -0.1313, "se": 0.3493, "t": -0.38},
            "tercile_slopes_non_monotonic": [0.432, 0.193, 0.464],
            "note": ("Not significant on either target and negative on both -- the opposite "
                     "sign to the hypothesis. CFB's returning-production discount beats raw "
                     "prev on NFL history only because it shrinks; a flat shrink does better "
                     "and preserves rank correlation (0.433 vs 0.399), which the discount "
                     "degrades. NFL is not CFB, but CFB ships this untested."),
        },
        "carryover": {c: {"slope": round(b, 4), "intercept": round(a, 5),
                          "standard_error": round(se, 4)} for c, (b, a, se) in fits.items()},
        "holdout_grade": graded,
        "paired_mae_gain": {"estimate": round(float(gain.mean()), 5),
                            "standard_error": round(se_gain, 5),
                            "t": round(float(gain.mean() / se_gain), 2),
                            "significant_at_95": bool(abs(gain.mean() / se_gain) > 1.96)},
        "open": ("k=2 was calibrated against the RAW prior. A better prior deserves more "
                 "weight, so k=2 is now conservative -- but re-tuning it on these same "
                 "seasons is the leak holdout_discipline.py exists to prevent."),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
