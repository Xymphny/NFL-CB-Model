#!/usr/bin/env python3
"""Measure what fixing the NGS team-code mismatch is actually worth.

ADR 0004 records the bug: the Rams are `LA` in the ratings and schedule and
`LAR` in the nflverse NGS release, so every Rams game falls back to the
rating-only coefficient vector -- 17 a season since 2022, losing Elo too.

The fix is one line of normalisation. This grades it before shipping, because
the reason to be careful is not doubt about which code means the Rams: the
backtest that produced the validated NGS improvement ran against the same
mismatch, so the full ensemble's validated support never included these games.
Moving 26 held-out games onto it is a model change.

DESIGN
No fitting happens here. Both coefficient vectors are fixed and were fit on
2016-2021, so grading on 2022-2023 is honest held-out. The comparison is
paired per game: same game, same ratings, two vectors, one actual margin.

Writes model/ngs_team_code_fix_results.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coverline.leagues.nfl.model import (  # noqa: E402
    MARGIN_COEFFICIENTS, MARGIN_COEFFICIENTS_V1_RATING_ONLY, GameFeatures,
    predict_margin,
)
from coverline.leagues.nfl.ngs import team_features  # noqa: E402
from ingest.nfl_schedules import load_schedules  # noqa: E402
from model.elo_rating import compute_elo_walk_forward  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "expanded_walk_forward_cache.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "ngs_team_code_fix_results.json")

#: Coefficients were fit on 2016-2021. These two are clean for this question.
HOLDOUT_SEASONS = (2022, 2023)
TEAM = "LA"


def _ngs_cache(seasons, weeks):
    """NGS features per (season, week), point-in-time: weeks < W."""
    out = {}
    for s in seasons:
        for w in sorted(weeks):
            try:
                out[(s, w)] = team_features(s, through_week=int(w), min_teams=28)
            except Exception as exc:
                out[(s, w)] = None
                print(f"  [ngs] {s} w{w}: unavailable ({type(exc).__name__})")
    return out


def main() -> int:
    games = pd.read_csv(CACHE)
    games = games[games.season.isin(HOLDOUT_SEASONS)]
    rams = games[(games.home_team == TEAM) | (games.away_team == TEAM)].copy()
    print(f"{len(rams)} {TEAM} games in {HOLDOUT_SEASONS[0]}-{HOLDOUT_SEASONS[1]}")

    ngs = _ngs_cache(sorted(rams.season.unique()), sorted(rams.week.unique()))

    # Elo, walk-forward over real results. Pre-game ratings only, so a game's
    # elo_diff uses nothing from that game. Built across ALL seasons up to the
    # holdout so ratings carry their real history rather than restarting.
    sched = load_schedules(seasons=list(range(2014, max(HOLDOUT_SEASONS) + 1)))
    per_game, _ = compute_elo_walk_forward(sched)
    elo_lookup = {
        (int(r.season), int(r.week), str(r.home_team), str(r.away_team)):
            float(r.home_elo_pre) - float(r.away_elo_pre)
        for r in per_game.itertuples()
        if not pd.isna(getattr(r, "home_elo_pre", None))
    }
    print(f"  [elo] {len(elo_lookup)} games with pre-game ratings")

    rows, skipped = [], []
    for g in rams.itertuples():
        f = ngs.get((g.season, g.week))
        if f is None or g.home_team not in f.index or g.away_team not in f.index:
            skipped.append(f"{g.season}w{g.week} {g.home_team}/{g.away_team}")
            continue

        base = dict(rating_diff=float(g.rating_diff),
                    rest_diff=float(g.rest_diff),
                    is_neutral_site=not bool(g.home_field))

        elo_diff = elo_lookup.get(
            (int(g.season), int(g.week), g.home_team, g.away_team))
        if elo_diff is None:
            skipped.append(
                f"{g.season}w{g.week} {g.home_team}/{g.away_team} (no elo)")
            continue

        # The rating-only vector has NO elo term, so passing elo_diff to it
        # changes nothing -- which is exactly the shipped behaviour being
        # compared against.
        rating_only = predict_margin(GameFeatures(**base, ngs_present=False,
                                                  elo_diff=elo_diff))
        full = predict_margin(GameFeatures(
            **base, ngs_present=True,
            cpoe_diff=float(f.loc[g.home_team, "team_cpoe"]
                            - f.loc[g.away_team, "team_cpoe"]),
            separation_diff=float(f.loc[g.home_team, "team_avg_separation"]
                                  - f.loc[g.away_team, "team_avg_separation"]),
            yac_oe_diff=float(f.loc[g.home_team, "team_yac_over_expected"]
                              - f.loc[g.away_team, "team_yac_over_expected"]),
            ryoe_diff=float(f.loc[g.home_team, "team_ryoe"]
                            - f.loc[g.away_team, "team_ryoe"]),
            elo_diff=elo_diff,
        ))
        rows.append({"season": int(g.season), "week": int(g.week),
                     "game": f"{g.home_team}/{g.away_team}",
                     "actual": float(g.actual_margin),
                     "elo_diff": round(elo_diff, 2),
                     "rating_only": rating_only, "full_ensemble": full})

    if not rows:
        print("no gradeable games; refusing to write a result")
        return 1

    df = pd.DataFrame(rows)
    err_ro = (df.rating_only - df.actual).abs()
    err_fe = (df.full_ensemble - df.actual).abs()
    gain = err_ro - err_fe            # positive = the fix helps
    n = len(df)
    se = float(gain.std(ddof=1) / np.sqrt(n))
    t = float(gain.mean() / se) if se > 0 else float("nan")

    art = {
        "_provenance": {
            "script": "model/ngs_team_code_fix.py",
            "generated": "2026-09-21",
            "question": ("what the ADR 0004 team-code fix is worth: Rams games "
                         "priced with the full ensemble instead of the "
                         "rating-only fallback"),
            "holdout_seasons": list(HOLDOUT_SEASONS),
            "why_holdout": ("both coefficient vectors were fit on 2016-2021, "
                            "so these seasons are clean for this comparison"),
            "graded_once": True,
            "paired": "same game, same ratings, two vectors, one actual margin",
            "elo_included": True,
            "elo_note": ("elo_diff is the real walk-forward pre-game "
                         "difference, passed to BOTH vectors. The rating-only "
                         "vector has no elo term, so it ignores it -- which is "
                         "the shipped behaviour. The full ensemble uses it. So "
                         "this measures the fix AS IT WOULD SHIP: NGS block "
                         "plus Elo, against the fallback. An earlier version "
                         "held elo at 0.0 on both sides, which compared "
                         "rating-only against a full ensemble missing one of "
                         "its own features."),
        },
        "n_games": n,
        "skipped": skipped,
        "mae": {"rating_only_shipped": round(float(err_ro.mean()), 4),
                "full_ensemble_fixed": round(float(err_fe.mean()), 4)},
        "paired_gain": {"estimate": round(float(gain.mean()), 5),
                        "standard_error": round(se, 5),
                        "t": round(t, 2),
                        "supported": bool(t > 2)},
        "games": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=1)

    print(f"\ngraded {n} games ({len(skipped)} skipped)")
    print(f"  MAE rating-only (shipped): {art['mae']['rating_only_shipped']}")
    print(f"  MAE full ensemble (fixed): {art['mae']['full_ensemble_fixed']}")
    print(f"  paired gain {gain.mean():+.4f}  SE {se:.4f}  t = {t:+.2f}")
    print(f"  supported: {art['paired_gain']['supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
