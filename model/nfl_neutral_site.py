#!/usr/bin/env python3
"""How the LIVE NFL vector handles neutral-site games, measured.

Writes data/nfl_neutral_site.json. A finding, not a fit: nothing here is
read by a pricing path.

WHY THIS EXISTS
The model docstring and the migration ledger carried "net home edge for
equal teams is -1.13 against a market near +2.5" as a live defect. It is a
property of the FULL-ENSEMBLE vector, and live NFL prices never use that
vector: RatingsSnapshotSource sets ngs_present=False on every game, so every
live price comes from MARGIN_COEFFICIENTS_V1_RATING_ONLY, whose equal-team
home edge is +1.65. What the live vector does at a neutral site -- drop its
2.83-point home term -- had never been checked. This checks it.

WHAT IS MEASURED
The rating-only vector over model/expanded_walk_forward_cache.csv
(2014-2023), joined to nflverse's `location` and closing `spread_line`,
split by site. Two residuals per group: actual margin minus model (noisy),
and model minus the closing line (much less noisy, the market as referee).

WHAT IT FOUND (2026-09-25)
At home sites the model sits 0.27 below the close (SE 0.10). At neutral
sites it sits 1.86 below (SE 0.61, n = 34): dropping the whole home term
removes about 1.6 points more than the market does. Outcomes lean the same
way (+1.29, SE 2.28) and are too few to say more. NOT CORRECTED: 34 games,
no holdout. Revisit when a held-out season of neutral games with ratings
exists (2024 onward are not in the cache).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
CACHE = ROOT / "model" / "expanded_walk_forward_cache.csv"
OUT = ROOT / "data" / "nfl_neutral_site.json"
SEASONS = (2014, 2023)


def load(schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    from coverline.leagues.nfl import model as M
    c = pd.read_csv(CACHE)
    if schedule is None:
        from coverline.leagues.nfl.live import load_schedule
        schedule = load_schedule(list(range(SEASONS[0], SEASONS[1] + 1)))
    s = schedule[schedule.game_type == "REG"][
        ["season", "week", "home_team", "away_team", "location", "spread_line"]]
    m = c.merge(s, on=["season", "week", "home_team", "away_team"], how="inner")
    k = M.MARGIN_COEFFICIENTS_V1_RATING_ONLY
    m["pred"] = (k["intercept"] + k["rating_diff"] * m.rating_diff
                 + k["rest_diff"] * m.rest_diff
                 + k["home_field"] * (m.location != "Neutral"))
    return m


def summarise(m: pd.DataFrame) -> dict:
    out = {}
    for site in ("Home", "Neutral"):
        d = m[m.location == site]
        r = d.actual_margin - d.pred
        g = (d.pred - d.spread_line).dropna()
        out[site.lower()] = {
            "n": int(len(d)),
            "outcome_residual": round(float(r.mean()), 3),
            "outcome_residual_se": round(float(r.std(ddof=1) / np.sqrt(len(d))), 3),
            "model_minus_close": round(float(g.mean()), 3),
            "model_minus_close_se": round(float(g.std(ddof=1) / np.sqrt(len(g))), 3),
        }
    return out


def build(generated: str) -> dict:
    from coverline.leagues.nfl import model as M
    k = M.MARGIN_COEFFICIENTS_V1_RATING_ONLY
    by_site = summarise(load())
    gap = by_site["neutral"]["model_minus_close"] - by_site["home"]["model_minus_close"]
    se = float(np.hypot(by_site["neutral"]["model_minus_close_se"],
                        by_site["home"]["model_minus_close_se"]))
    return {
        "_provenance": {
            "script": "model/nfl_neutral_site.py",
            "generated": generated,
            "source": "model/expanded_walk_forward_cache.csv x nflverse games.csv location/spread_line",
            "seasons": list(SEASONS),
            "vector": "MARGIN_COEFFICIENTS_V1_RATING_ONLY -- the one every live NFL price uses",
            "a_finding_not_a_fit": True,
        },
        "home_edge_for_equal_teams_live": round(k["home_field"] + k["intercept"], 4),
        "home_term_dropped_at_neutral": k["home_field"],
        "by_site": by_site,
        "neutral_extra_gap_to_close": round(gap, 3),
        "neutral_extra_gap_se": round(se, 3),
        "corrected": False,
        "why_not": ("34 neutral games and no held-out season; the gap is measured "
                    "against the market, not graded on results"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated", required=True)
    a = ap.parse_args(argv)
    art = build(a.generated)
    OUT.write_text(json.dumps(art, indent=1) + "\n")
    n = art["by_site"]["neutral"]
    print(f"wrote {OUT.relative_to(ROOT)}: neutral n={n['n']}, extra gap to close "
          f"{art['neutral_extra_gap_to_close']:+.2f} (SE {art['neutral_extra_gap_se']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
