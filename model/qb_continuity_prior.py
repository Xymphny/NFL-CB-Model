"""H1b: does QB continuity decide how far last season carries?

THE HYPOTHESIS, AND WHY IT WAS WORTH TESTING. H1a asked whether roster
continuity scales the preseason prior and was rejected outright
(model/preseason_prior_regression.py): continuity as a single scalar
is noise, negative on both targets. But a scalar averaged over every
touch is a blunt instrument. The quarterback is the one position whose
replacement genuinely changes a team's distribution, so the sharper
question is whether HIS continuity does what the roster-wide measure
could not.

Primary QB = the passer with the most attempts for that team-season.
About 61% of teams keep him, steady across 2017-2025.

WHAT THE TRAINING SEASONS SAID -- and they said it loudly (n=192):

  target          prev x QB continuity     SE       t
  offense_voa            +0.355          0.135   +2.63
  total_rating           +0.323          0.136   +2.36
  defense_voa            -0.048          0.137   -0.35

Split by whether the passer returned, the offensive carryover is
0.088 against 0.543 -- six times more of last year's offense survives
when the quarterback does.

AND IT PASSED ITS PLACEBO, which is why this was not dismissed as a
confound. Good teams both keep their quarterbacks and stay good, so a
team-quality proxy would lift DEFENSIVE carryover too. It does not:
0.380 against 0.329, and an interaction indistinguishable from zero.
The effect lands only where the mechanism says it should.

IT FAILED ITS GATE ANYWAY. Coefficients fit on 2017-2022 and graded
once on 2023-2025, against the unconditional regression already
shipped:

  prior                                 MAE      RMSE
  regressed, unconditional (shipped)   0.0785   0.0955
  regressed, QB-conditional            0.0789   0.0959

Paired gain -0.00042 (SE 0.00245, t = -0.17). Better in 2023 and 2024,
worse in 2025. Null, and if anything slightly negative.

NOT SHIPPED. A significant training interaction plus a clean placebo
plus an obvious mechanism is still not held-out evidence, and this
project's rule is that nothing ships without it. Two of three seasons
improving is not a result; it is two of three.

WHY IT PROBABLY DID NOT SURVIVE. t = 2.63 at n=192 is p ~ 0.009 before
accounting for having looked at three targets, and splitting one slope
into two buys variance that a 96-row holdout has to pay for. A real
but modest effect would look exactly like this.

WHAT WOULD CHANGE THE ANSWER: more seasons, not more searching. The
temptation after a failed gate is to retreat to a narrower version --
apply it only when the quarterback CHANGED, where the training contrast
is widest -- and that is fishing on the same data. Deliberately not
done. Re-run this when 2026 and 2027 have closed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "qb_continuity_prior_results.json")
TRAIN, HOLDOUT = range(2017, 2023), range(2023, 2026)


def qb_continuity(pw):
    """Did the team's primary passer return? Known at roster time, so
    it is causally available when the prior is built."""
    qb = (pw[pw["attempts"] > 0]
          .groupby(["season", "team", "player_id"], as_index=False)["attempts"].sum()
          .sort_values("attempts", ascending=False)
          .drop_duplicates(["season", "team"]))
    rows = []
    for s in sorted(pw["season"].unique())[1:]:
        prev, cur = (qb[qb.season == s - 1].set_index("team"),
                     qb[qb.season == s].set_index("team"))
        for t in prev.index:
            if t in cur.index:
                rows.append({"season": int(s), "team": t,
                             "qb_same": int(prev.loc[t, "player_id"] == cur.loc[t, "player_id"])})
    return pd.DataFrame(rows)


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return float(b), float(y.mean() - b * x.mean())


def _ols(cols, y):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    cov = (res @ res) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov))


def main():
    rc, pc = os.environ.get("RATINGS_CACHE"), os.environ.get("PW_CACHE")
    if not (rc and pc and os.path.exists(rc) and os.path.exists(pc)):
        raise SystemExit("set RATINGS_CACHE (season ratings, see "
                         "model/preseason_prior_regression.py) and PW_CACHE "
                         "(nflverse stats_player_week 2016-2025)")
    r, pw = pd.read_parquet(rc), pd.read_parquet(pc)
    q = qb_continuity(pw)

    cols = ("total_rating", "offense_voa", "defense_voa")
    prev = r.assign(season=r["season"] + 1).rename(columns={c: "prev_" + c for c in cols})
    d = r.merge(prev, on=["season", "team"]).merge(q, on=["season", "team"])
    tr, te = d[d["season"].isin(TRAIN)].copy(), d[d["season"].isin(HOLDOUT)]
    tr["qb_c"] = tr["qb_same"] - tr["qb_same"].mean()

    inter, splits = {}, {}
    for c in cols:
        b, se = _ols([tr["prev_" + c].values, (tr["prev_" + c] * tr["qb_c"]).values], tr[c].values)
        inter[c] = {"estimate": round(float(b[2]), 4), "se": round(float(se[2]), 4),
                    "t": round(float(b[2] / se[2]), 2)}
        splits[c] = {str(k): round(_fit(tr[tr.qb_same == k]["prev_" + c],
                                        tr[tr.qb_same == k][c])[0], 4) for k in (0, 1)}

    uo, ud = (_fit(tr.prev_offense_voa, tr.offense_voa), _fit(tr.prev_defense_voa, tr.defense_voa))
    qo = {k: _fit(tr[tr.qb_same == k].prev_offense_voa, tr[tr.qb_same == k].offense_voa)
          for k in (0, 1)}

    def predict(sel, conditional):
        off = ((sel.qb_same.map({k: qo[k][0] for k in (0, 1)}) * sel.prev_offense_voa
                + sel.qb_same.map({k: qo[k][1] for k in (0, 1)})) if conditional
               else uo[0] * sel.prev_offense_voa + uo[1])
        return off - (ud[0] * sel.prev_defense_voa + ud[1])

    graded = {}
    for name, p in (("regressed_unconditional_shipped", predict(te, False)),
                    ("regressed_qb_conditional", predict(te, True))):
        e = te.total_rating - p
        graded[name] = {"mae": round(float(e.abs().mean()), 4),
                        "rmse": round(float(np.sqrt((e ** 2).mean())), 4)}
    gain = ((te.total_rating - predict(te, False)).abs()
            - (te.total_rating - predict(te, True)).abs())
    se_g = float(gain.std(ddof=1) / np.sqrt(len(gain)))

    print(f"H1b -- QB continuity and the preseason prior   (train n={len(tr)})\n")
    print(f"{'target':>14} {'prev x QB':>11} {'SE':>8} {'t':>7}   carryover 0 / 1")
    for c in cols:
        i = inter[c]
        print(f"{c:>14} {i['estimate']:>+11.3f} {i['se']:>8.3f} {i['t']:>+7.2f}   "
              f"{splits[c]['0']:.3f} / {splits[c]['1']:.3f}")
    print(f"\ngraded once on {min(HOLDOUT)}-{max(HOLDOUT)} (n={len(te)}):")
    for k, v in graded.items():
        print(f"  {k:>34}  MAE {v['mae']:.4f}  RMSE {v['rmse']:.4f}")
    print(f"\n  paired gain {gain.mean():+.5f} +/- {se_g:.5f} (t={gain.mean()/se_g:+.2f})")
    print("  VERDICT: NOT SHIPPED. Significant on train, clean placebo, null on the gate.")

    out = {
        "_provenance": {"script": "model/qb_continuity_prior.py",
                        "generated": __import__("datetime").date.today().isoformat(),
                        "fit_seasons": [min(TRAIN), max(TRAIN)],
                        "graded_seasons": [min(HOLDOUT), max(HOLDOUT)],
                        "related": "model/preseason_prior_regression.py (H1a + the shipped fix)"},
        "verdict": "REJECTED -- passed on train, failed its held-out gate",
        "train_interactions": inter,
        "carryover_split_by_qb_continuity": splits,
        "placebo": ("QB continuity must move offense and not defense, or it is a team-quality "
                    "proxy. Offense t=+2.63, defense t=-0.35. The placebo PASSED -- and the "
                    "hypothesis still failed its gate, which is the point worth keeping."),
        "holdout_grade": graded,
        "paired_gain": {"estimate": round(float(gain.mean()), 5),
                        "standard_error": round(se_g, 5),
                        "t": round(float(gain.mean() / se_g), 2)},
        "not_done_deliberately": ("Retreating to a narrower version after a failed gate -- "
                                  "applying it only when the QB changed, where the training "
                                  "contrast is widest -- is fishing on the same data."),
        "revisit": "when 2026 and 2027 have closed; more seasons, not more searching",
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
