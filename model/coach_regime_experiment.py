"""
The coach-regime question, tested on history: do the model's
EARLY-SEASON flags (weeks 1-4, where the prior dominates the blend)
grade differently in games involving a first-year external head
coach than in stable-regime games?

Three hypotheses, all live before this run:
  H-worse:  regime flags are fade-the-market-repricing errors -> cap them.
  H-better: markets OVERSHOOT new-coach repricing (hype premium) ->
            the ghost prior is a contrarian asset; capping would be wrong.
  H-wash:   no detectable difference -> chip stays advisory, no teeth.

Design notes (honesty items):
- This is a conditional SPLIT of the existing validated model's flags,
  not a fitted feature -- no new parameters, so no train/test split is
  required for the split itself; the underlying predictions are
  walk-forward as always.
- Internal promotions (tier 0) count as STABLE, per the live 2026
  measurement (Buffalo's internal promo behaved exactly like a stable
  team: zero rank movement week 1).
- 2016-2023 pool: 48 external regimes. Early-season regime flags will
  number ~100 -- SE around 5 points of ATS. Only coarse verdicts are
  claimable, and the ship rule treats "inconclusive" as "no teeth."
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.test_full_ensemble import build_combined_dataset
from model.prediction import predict_margin

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
EARLY_WEEKS = (1, 2, 3, 4)
SEASONS = list(range(2016, 2024))


def load_regimes():
    c = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "data", "coach_changes.json")))["changes"]
    ext = {(int(season), team) for season, teams in c.items()
           for team, (tier, _) in teams.items() if tier >= 1}
    return ext


def main():
    ext = load_regimes()
    print(f"external regimes loaded: {len(ext)} team-seasons")
    games = pd.read_csv(os.environ.get("GAMES_CSV", GAMES_URL))
    games = games[(games["game_type"] == "REG") & games["season"].isin(SEASONS)]

    print("building walk-forward ensemble dataset (this loads pbp+NGS; several minutes)...")
    comb = build_combined_dataset()
    comb = comb.merge(games[["season", "week", "home_team", "away_team", "spread_line"]].dropna(subset=["spread_line"]),
                      on=["season", "week", "home_team", "away_team"], how="inner")
    comb["model_margin"] = comb.apply(lambda r: predict_margin(
        r["rating_diff"], False, r["rest_diff"], cpoe_diff=r["cpoe_diff"],
        separation_diff=r["separation_diff"], yac_oe_diff=r["yac_oe_diff"],
        ryoe_diff=r["ryoe_diff"], elo_diff=r["elo_diff"]), axis=1)
    comb["edge"] = comb["model_margin"] - comb["spread_line"]
    early = comb[comb["week"].isin(EARLY_WEEKS)].copy()
    early["regime_game"] = early.apply(
        lambda r: (r["season"], r["home_team"]) in ext or (r["season"], r["away_team"]) in ext, axis=1)
    print(f"early-season games with lines: {len(early)} | regime games: {early['regime_game'].sum()}")

    # EVIDENCE ARTIFACT (2026-09-20). A 0/9 result was quoted verbatim
    # to users on the live board and underwrote a staking rule, but
    # existed only as stdout from a run nobody could reproduce without
    # several minutes of pbp+NGS downloads. Every graded cell is now
    # captured and written to model/coach_regime_results.json,
    # following the precedent set by model/cfb_backtest_2023_results.json.
    #
    # AND CAPTURING IT IMMEDIATELY CONTRADICTED THE QUOTED NUMBER.
    # There is no 0/9 cell. Backed-regime early flags grade 0/6 at the
    # Lean threshold and 0/3 at Play -- and Play is a SUBSET of Lean,
    # so 6 + 3 = 9 double-counts three games. Whether the original was
    # that double-count or a stale run cannot be recovered from stdout
    # that no longer exists, which is the whole argument for writing
    # the artifact. Every quoted site was corrected to 0/6 on
    # 2026-09-20; the rule itself stands, because 0/6 still points the
    # same way and the rule only caps stakes rather than blocking.
    captured = []

    def grade(df, label, min_edge):
        d = df[df["edge"].abs() >= min_edge]
        pushes = d["actual_margin"] == d["spread_line"]
        win = ((d["edge"] > 0) == (d["actual_margin"] > d["spread_line"]))[~pushes]
        captured.append({
            "label": label.strip(), "min_edge": min_edge,
            "n_flags": int(len(d)), "n_graded": int(len(win)),
            "wins": int(win.sum()) if len(win) else 0,
            "pushes": int(pushes.sum()),
            "ats_pct": round(float(win.mean() * 100), 2) if len(win) else None,
            "se_pct": round(float(100 * (0.25 / len(win)) ** 0.5), 2) if len(win) else None,
        })
        if len(win) == 0:
            return
        pct = win.mean() * 100
        se = 100 * np.sqrt(0.25 / len(win))
        print(f"  {label:34} |edge|>={min_edge}: {win.sum():4d}/{len(win):4d} = {pct:5.2f}%  (SE +/-{se:.1f})")
        return win

    for t in (2.5, 4.0):
        print(f"\n=== threshold {t} ===")
        grade(early[~early["regime_game"]], "STABLE-regime early flags", t)
        w = grade(early[early["regime_game"]], "REGIME-game early flags", t)
        # Direction inside regime games: did the model's pick side back
        # the regime team, or fade it?
        rg = early[early["regime_game"] & (early["edge"].abs() >= t)].copy()
        def picked_regime(r):
            picked_home = r["edge"] > 0
            team = r["home_team"] if picked_home else r["away_team"]
            return (r["season"], team) in ext
        rg["picked_regime"] = rg.apply(picked_regime, axis=1)
        for side, lbl in ((True, "model BACKED the regime team"), (False, "model FADED the regime team")):
            grade(rg[rg["picked_regime"] == side], f"  {lbl}", t)

    # Reference: all-season flags at same thresholds (the known baseline).
    print("\n=== reference: weeks 5+ (prior faded) ===")
    late = comb[~comb["week"].isin(EARLY_WEEKS)]
    for t in (2.5, 4.0):
        grade(late, f"weeks 5+ all flags", t)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "coach_regime_results.json")
    with open(out_path, "w") as fh:
        json.dump({
            "_provenance": {
                "script": "model/coach_regime_experiment.py",
                "generated": __import__("datetime").date.today().isoformat(),
                "seasons": list(SEASONS),
                "early_weeks": list(EARLY_WEEKS),
                "regime_source": "data/coach_changes.json (tier >= 1 = external hire)",
                "n_external_team_seasons": len(ext),
                "caveat": ("Small samples. The headline backed-regime cell is single-digit; "
                           "read the SE column before treating any cell as settled. The seasons "
                           "here overlap the 2016-2021 window MARGIN_COEFFICIENTS was fit on, so "
                           "the ensemble margins are partly in-sample for those years."),
            },
            "grades": captured,
        }, fh, indent=2)
    print(f"\nwrote {out_path}: {len(captured)} graded cells")


if __name__ == "__main__":
    main()
