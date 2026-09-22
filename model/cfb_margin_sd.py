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
from model import fit_data_checks as checks  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "cfb_full_walk_forward_cache.csv")


def main() -> int:
    g = pd.read_csv(CACHE)
    # IMPOSSIBLE ROWS ARE EXCLUDED. 76 of 1,731 rows carry a final margin of
    # zero, and college football has not permitted a tie since 1996. Two
    # upstream causes: scores never fetched and stored as 0-0, and rows frozen
    # at a mid-game score -- Auburn 22-22 Alabama in 2021, a game Alabama won
    # 24-22 in four overtimes. Every one is also recorded as a home LOSS.
    #
    # They shrank this constant by 1.93%, in the OVERCONFIDENT direction,
    # which is the one that oversizes a stake.
    before = len(g)
    checks.check_no_impossible_ties(
        g[g.actual_margin != 0], "cfb", label="cfb walk-forward cache",
        margin="actual_margin")
    g = g[g.actual_margin != 0].reset_index(drop=True)
    print(f"excluded {before - len(g)} impossible tied rows of {before}")
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
