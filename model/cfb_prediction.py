"""
CFB's production prediction module -- matching the NFL pattern
(model/prediction.py), giving the validated CFB margin model a real,
documented home rather than leaving it as a one-off test script.

MARGIN_COEFFICIENTS below is from the FULL weekly walk-forward
backtest (model/cfb_full_walk_forward.py, every real week 4-13 rather
than 3 checkpoint weeks): train 2021-2022, test 2023 (586 real
held-out games, a much more precise and robust test than the earlier
121-game checkpoint approximation). Straight-up accuracy: 66.21%
(DVOA alone) -> 70.99% (Elo alone) -> 71.84% (ensemble).

Real Elo ratings for CFB use the EXACT SAME code as NFL
(model/elo_rating.py, completely unmodified) -- confirmed directly
that CFB's derived schedule (ingest/cfb_pbp.py's derive_cfb_schedule)
already matches the schema Elo expects.

Opponent-adjustment uses iterations=3, regression=0.5 -- NOTE:
iterations=1 tested better for DVOA ALONE (model/calibrate_cfb_opponent_adjustment.py),
but iterations=3 gives the better REAL ensemble result, confirmed
directly by testing both in the actual deployed configuration. Always
trust the full-ensemble test over a component's standalone accuracy.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MARGIN_COEFFICIENTS = {
    "rating_diff": 15.6918,
    "elo_diff": 0.0673,
    "intercept": -1.9871,
}

# Reference: DVOA-only, from the earlier 3-checkpoint approximation
# (before the full weekly backtest existed).
MARGIN_COEFFICIENTS_DVOA_ONLY = {
    "rating_diff": 39.1897,
    "intercept": 0.8030,
}

# Reference: the ensemble coefficients from the earlier 3-checkpoint
# approximation (121 held-out games), before the full weekly backtest
# (586 held-out games) gave a more precise, robust fit.
MARGIN_COEFFICIENTS_CHECKPOINT_APPROXIMATION = {
    "rating_diff": 11.9848,
    "elo_diff": 0.0700,
    "intercept": -3.6350,
}


def predict_margin(rating_diff, elo_diff=None):
    """
    elo_diff defaults to None (not 0.0) -- same real reason as NFL's
    predict_margin: the coefficients were co-calibrated WITH Elo
    present, so silently treating "Elo unavailable" the same as
    "Elo available and zero" would systematically understate
    rating_diff's real effect. Falls back to the DVOA-only
    coefficients when Elo isn't supplied.
    """
    if elo_diff is None:
        return (
            MARGIN_COEFFICIENTS_DVOA_ONLY["rating_diff"] * rating_diff
            + MARGIN_COEFFICIENTS_DVOA_ONLY["intercept"]
        )
    return (
        MARGIN_COEFFICIENTS["rating_diff"] * rating_diff
        + MARGIN_COEFFICIENTS["elo_diff"] * elo_diff
        + MARGIN_COEFFICIENTS["intercept"]
    )
