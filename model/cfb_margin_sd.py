#!/usr/bin/env python3
"""Measure the CFB residual standard deviation from the committed cache.

Exists because the number belongs in a file that produced it rather than in a
constant somebody typed. src/coverline/leagues/cfb/model.py's MARGIN_SD must
equal what this prints, and a guard test asserts it.

CFB margins are far more dispersed than NFL's -- a playoff team against a
bottom-tier program has no NFL analogue -- so borrowing NFL's 13.30 would make
every CFB probability too confident.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coverline.leagues.cfb.model import GameFeatures, predict_margin  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "cfb_full_walk_forward_cache.csv")


def main() -> int:
    g = pd.read_csv(CACHE)
    # The cache carries no elo, so this measures the DVOA-only path -- which
    # is the one a caller without Elo gets, and the conservative choice for a
    # dispersion estimate.
    pred = np.array([predict_margin(GameFeatures(rating_diff=float(r), elo_present=False))
                     for r in g.rating_diff])
    resid = g.actual_margin.values - pred
    print(f"n = {len(g)}  seasons {g.season.min()}-{g.season.max()}")
    print(f"residual mean {resid.mean():+.4f}   sd {resid.std(ddof=1):.4f}")
    print(f"MARGIN_SD = {resid.std(ddof=1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
