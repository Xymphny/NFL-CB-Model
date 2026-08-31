"""
Play-level value scoring — spec Sections 3.1, 3.2, and the turnover-luck
accuracy improvement from the "what increases accuracy" discussion.

This assigns every play a continuous value score before any baseline
comparison happens (that's play_value in Section 3.4's VOA formula).
"""

import numpy as np
import pandas as pd

# Section 3.1 — down-specific success thresholds (fraction of needed yards)
SUCCESS_THRESHOLDS_NFL = {1: 0.45, 2: 0.60, 3: 1.00, 4: 1.00}

# Section 6 — CFB gets steeper thresholds, reflecting higher offensive variance
SUCCESS_THRESHOLDS_CFB = {1: 0.50, 2: 0.70, 3: 1.00, 4: 1.00}


def get_success_thresholds(league: str) -> dict:
    if league.upper() == "CFB":
        return SUCCESS_THRESHOLDS_CFB
    return SUCCESS_THRESHOLDS_NFL


# Turnover point-value-at-field-position table, coarse version.
# Real point-value tables (see: Romer 2006, nflfastR's `ep` field) vary
# continuously by field position — we approximate with a simple curve here
# and note that nflfastR's own `epa` column is a drop-in replacement for
# this whole function once available, since it already does exactly this.
def _turnover_value(yardline_100: float) -> float:
    """
    Point value lost on a turnover, as a function of distance from the
    end zone the possessing team was trying to score in (yardline_100:
    100 = own goal line, 0 = opponent's goal line).

    Turnovers deep in opponent territory (small yardline_100) cost more
    — you were close to scoring and gave it away — than turnovers deep
    in your own territory.
    """
    # Roughly linear from ~3.0 pts (own end) to ~6.0 pts (opponent red zone),
    # a simplification of published field-position point-value curves.
    return 3.0 + 3.0 * (1 - yardline_100 / 100.0)


def score_play(row: pd.Series, league: str = "NFL") -> float:
    """
    Assign a continuous value to a single play.

    Section 3.2 logic:
    - Turnovers get a large fixed penalty scaled by field position
    - Touchdowns get a large fixed bonus
    - Otherwise, value scales with yards gained relative to yards needed
      for the down, continuously above and below the success threshold,
      with diminishing returns on very large gains

    league : "NFL" or "CFB" — selects the success-threshold table
        (Section 6: CFB thresholds are steeper, 50/70/100 vs 45/60/100).
    """
    if row.get("interception") == 1 or row.get("fumble_lost") == 1:
        return -_turnover_value(row["yardline_100"])

    if row.get("touchdown") == 1:
        return 6.0  # fixed TD bonus; red-zone proximity multiplier folds
        # naturally out of this being reached mostly from short yardline_100

    down = row.get("down")
    ydstogo = row.get("ydstogo")
    yards_gained = row.get("yards_gained")

    if pd.isna(down) or pd.isna(ydstogo) or ydstogo <= 0:
        return 0.0

    thresholds = get_success_thresholds(league)
    threshold = thresholds.get(int(down), 1.00)
    pct_of_needed = yards_gained / ydstogo

    # Value curve: 0 at zero yards gained, crosses a "success" reference
    # value (1.0) exactly at the threshold, then grows with diminishing
    # returns (sqrt) beyond it. Below threshold, scales linearly down to
    # a floor penalty for zero/negative gains.
    if pct_of_needed >= threshold:
        excess = pct_of_needed - threshold
        return 1.0 + np.sqrt(max(excess, 0))
    else:
        # Linear from 0 at pct_of_needed=0 up to 1.0 at the threshold;
        # negative gains extend below zero.
        return pct_of_needed / threshold if threshold > 0 else 0.0


def expected_turnover_value(row: pd.Series, league: str = "NFL") -> float:
    """
    Turnover-luck adjustment (accuracy improvement, not in original spec).

    Fumble recovery is close to a league-wide coin flip regardless of
    which team caused the fumble. For fumbles specifically, credit the
    *expected* value (roughly half the turnover value, since recovery
    rate hovers near 50%) rather than the *actual* outcome, so a team
    that ran hot/cold on recovery luck this season isn't over/under-rated
    going forward.

    Interceptions are NOT adjusted here — unlike fumbles, interception
    rate is meaningfully a function of QB/defense skill, not luck, so
    there's no equivalent "expected value" correction to make.
    """
    if row.get("fumble_lost") == 1:
        # Actual: full turnover value was charged. Expected: only ~50% of
        # fumbles are recovered by the defense, so the offense's *expected*
        # cost, ex-ante, was about half the realized penalty.
        full_penalty = -_turnover_value(row["yardline_100"])
        return full_penalty * 0.5
    return score_play(row, league=league)
