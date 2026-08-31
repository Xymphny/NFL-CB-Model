"""
De-vig, bootstrap-based uncertainty quantification, and market
divergence flagging. Spec Sections 9.3 and 10 (item 3).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability (still vig-inflated)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def devig_two_way(prob_a: float, prob_b: float) -> tuple:
    """
    Proportional de-vig — the simplest standard method (Section 7's
    de-vig requirement). Divides out the vig proportionally to each
    side's share of the raw implied probabilities.

    A more sophisticated method (logarithmic/Shin de-vigging) exists and
    is more accurate for favorites/longshots specifically, but adds
    complexity that isn't justified until proportional de-vig is
    validated to be insufficient — start simple, per the spec's general
    "start coarse, refine later" pattern (Section 3.3's precedent).
    """
    total = prob_a + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def bootstrap_rating_uncertainty(
    scored_plays: pd.DataFrame,
    rating_fn,
    n_bootstrap: int = 200,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Section 10, item 3 — uncertainty quantification via bootstrap.

    Resamples the season's plays with replacement, recomputes the rating
    each time, and the spread of the resulting ratings across iterations
    IS the confidence interval. No new data required — this is repeated
    computation on data already gathered.

    Parameters
    ----------
    scored_plays : DataFrame that already has play_value/voa_adj columns
        (i.e. has been through the full pipeline once already)
    rating_fn : callable, DataFrame -> DataFrame, e.g. model.ratings.team_ratings
        Applied to each bootstrap resample.
    n_bootstrap : how many resamples to run. 200 is a reasonable default
        for a first pass — real-money-precision confidence intervals may
        want more, at proportionally higher compute cost.
    sample_frac : fraction of the season's plays to sample per iteration
        (with replacement). 1.0 = same size as the original season.

    Returns
    -------
    DataFrame indexed by team, with columns: rating_mean, rating_std,
    rating_p05, rating_p95 (5th/95th percentile — a 90% interval).
    """
    n_plays = len(scored_plays)
    sample_size = int(n_plays * sample_frac)

    all_ratings = []
    for i in range(n_bootstrap):
        resample = scored_plays.sample(n=sample_size, replace=True, random_state=i)
        ratings = rating_fn(resample)
        all_ratings.append(ratings["total_rating"])

    stacked = pd.concat(all_ratings, axis=1)
    stacked.columns = range(n_bootstrap)

    summary = pd.DataFrame({
        "rating_mean": stacked.mean(axis=1),
        "rating_std": stacked.std(axis=1),
        "rating_p05": stacked.quantile(0.05, axis=1),
        "rating_p95": stacked.quantile(0.95, axis=1),
    })
    return summary


def flag_divergence(
    model_spread: float,
    model_total: float,
    model_win_prob_home: float,
    market_spread: float,
    market_total: float,
    market_odds_home: float,
    market_odds_away: float,
    divergence_threshold_points: float = 2.0,
    divergence_threshold_prob: float = 0.07,
) -> dict:
    """
    Section 9.3 — compare model output to market, flag meaningful gaps.
    Model and market stay fully separate (Section 9.3's explicit
    decision) — this function only computes and reports the gap, it
    does not feed back into the model's own prediction.

    market_odds_home/away should be de-vigged probabilities already
    (see devig_two_way), not raw American odds.
    """
    spread_gap = model_spread - market_spread
    total_gap = model_total - market_total
    prob_gap = model_win_prob_home - market_odds_home

    return {
        "spread_gap": float(spread_gap),
        "total_gap": float(total_gap),
        "win_prob_gap": float(prob_gap),
        # Explicit bool(...) here matters: these comparisons produce
        # numpy.bool_ when the inputs are numpy floats (as they are once
        # flowing through from pandas upstream), and numpy.bool_ isn't
        # JSON-serializable — confirmed as a real "Object of type bool
        # is not JSON serializable" failure in testing.
        "spread_flagged": bool(abs(spread_gap) >= divergence_threshold_points),
        "total_flagged": bool(abs(total_gap) >= divergence_threshold_points),
        "win_prob_flagged": bool(abs(prob_gap) >= divergence_threshold_prob),
    }


if __name__ == "__main__":
    # De-vig sanity check with realistic vig-inflated numbers.
    home_prob_raw = american_to_implied_prob(-150)
    away_prob_raw = american_to_implied_prob(+130)
    print(f"Raw implied probabilities: home={home_prob_raw:.4f}, away={away_prob_raw:.4f}")
    print(f"  (sum = {home_prob_raw + away_prob_raw:.4f} -- this is the vig, should be > 1.0)")

    home_fair, away_fair = devig_two_way(home_prob_raw, away_prob_raw)
    print(f"De-vigged: home={home_fair:.4f}, away={away_fair:.4f} (sum = {home_fair + away_fair:.4f})")

    # Divergence sanity check.
    result = flag_divergence(
        model_spread=-6.5, model_total=48.0, model_win_prob_home=0.68,
        market_spread=-3.5, market_total=45.5, market_odds_home=0.58, market_odds_away=0.42,
    )
    print(f"\nDivergence check: {result}")
