#!/usr/bin/env python3
"""What pricing a CFB neutral-site game would need, measured. A finding.

Writes data/cfb_neutral_site.json. Nothing here is read by a pricing path:
leagues/cfb/live.py still REFUSES neutral-site games, and this records why
the refusal stands and what would lift it.

THE PROBLEM
The CFB vector has no home term -- home advantage sits in the intercept -- so
a neutral-site game cannot be priced by dropping one. The question is how
much the market takes out for a neutral site, in the model's own units.

THE MEASUREMENT
The DVOA-only vector over model/cfb_full_walk_forward_cache.csv (2021-2023),
joined to CFBD closing lines and to ESPN's neutral-site flag
(data/raw/cfb/neutral_sites_2021_2023.csv, pulled with
model/ingest/cfb_espn.day_rows). The model's gap to the close at neutral
sites, minus the same gap at home sites, is the home edge the market removes
that the model cannot.

FOUND (2026-09-25): +3.85 points, SE 1.77, n = 21 neutral games. A plausible
college home edge and a t of about 2.2 -- on 21 games with no held-out
season, because the lines cache ends in 2023 and no 2024-2025 weekly ratings
exist. Not applied. To lift the refusal: a second source of closing lines
for 2024+ and the weekly ratings recomputed for those seasons, then fit on
2021-2023 and grade once on the new seasons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
FLAGS = ROOT / "data" / "raw" / "cfb" / "neutral_sites_2021_2023.csv"
OUT = ROOT / "data" / "cfb_neutral_site.json"
KEY = ["season", "week", "home_team", "away_team"]
#: ESPN spells one school differently from the play-by-play cache.
ESPN_TO_CACHE = {"Appalachian State": "App State"}


def load() -> pd.DataFrame:
    from coverline.leagues.cfb import model as M
    flags = pd.read_csv(FLAGS).replace({"home_team": ESPN_TO_CACHE, "away_team": ESPN_TO_CACHE})
    c = pd.read_csv(ROOT / "model" / "cfb_full_walk_forward_cache.csv")
    lines = pd.read_csv(ROOT / "model" / "cfb_lines_cache.csv")[KEY + ["spread_line"]]
    m = c.merge(lines, on=KEY).merge(flags[KEY + ["neutral_site"]], on=KEY)
    k = M.MARGIN_COEFFICIENTS_DVOA_ONLY
    m["pred"] = k["intercept"] + k["rating_diff"] * m.rating_diff
    return m


def build(generated: str) -> dict:
    m = load()
    by = {}
    for site, d in (("home", m[~m.neutral_site]), ("neutral", m[m.neutral_site])):
        g = d.pred - d.spread_line
        r = d.actual_margin - d.pred
        by[site] = {"n": int(len(d)),
                    "model_minus_close": round(float(g.mean()), 3),
                    "model_minus_close_se": round(float(g.std(ddof=1) / np.sqrt(len(d))), 3),
                    "outcome_residual": round(float(r.mean()), 3),
                    "outcome_residual_se": round(float(r.std(ddof=1) / np.sqrt(len(d))), 3)}
    gap = by["neutral"]["model_minus_close"] - by["home"]["model_minus_close"]
    se = float(np.hypot(by["neutral"]["model_minus_close_se"], by["home"]["model_minus_close_se"]))
    return {
        "_provenance": {"script": "model/cfb_neutral_site.py", "generated": generated,
                        "source": ("model/cfb_full_walk_forward_cache.csv x model/cfb_lines_cache.csv "
                                   "x data/raw/cfb/neutral_sites_2021_2023.csv"),
                        "vector": "MARGIN_COEFFICIENTS_DVOA_ONLY -- the one live CFB prices use",
                        "a_finding_not_a_fit": True},
        "joined": int(len(m)),
        "by_site": by,
        "home_edge_the_market_removes": round(gap, 3),
        "home_edge_se": round(se, 3),
        "applied": False,
        "why_not": "21 neutral games, no held-out season (lines end 2023; no 2024+ weekly ratings)",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated", required=True)
    a = ap.parse_args(argv)
    art = build(a.generated)
    OUT.write_text(json.dumps(art, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: neutral n={art['by_site']['neutral']['n']}, "
          f"market removes {art['home_edge_the_market_removes']:+.2f} (SE {art['home_edge_se']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
