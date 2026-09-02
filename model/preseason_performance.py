"""
Preseason performance signal — a real, data-driven addition to the
preseason prior (Section 11.1), built after finding that nflverse (the
source this whole project runs on) has ZERO preseason play-by-play
data, for any season, ever — confirmed directly, not assumed. Real
2026 preseason results were gathered manually from ESPN instead (see
preseason_2026_results.py), since ESPN has real scores but not
down-by-down play-by-play, which limits what's computable here to a
point-differential signal, not a full Layer 1 rating.

HONEST CAVEATS, read before trusting this:
1. Starters typically play one series total across all of preseason.
   Preseason point differential mostly reflects backup/roster-bubble
   performance, not the team's actual Week 1 roster.
2. A meaningful fraction of preseason participants get cut before
   Week 1 — this signal partly measures players who won't be on the
   team when it matters.
3. Unlike the k=2 last-season-prior weight (calibrated via a real
   backtest against 2021-2023 data), the weight this signal gets below
   is NOT backtested — there's no efficient way to gather multiple
   historical seasons of preseason scores without repeating this same
   manual ESPN-gathering process for each one. Treat the default
   weight as a conservative, defensible starting point, not a
   validated number.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.preseason_2026_results import PRESEASON_2026_RESULTS

# Deliberately conservative and NOT backtested (see module docstring).
POINT_DIFF_TO_RATING_SCALE = 50.0
YARDAGE_MARGIN_TO_RATING_SCALE = 500.0  # yardage differentials run larger in magnitude than point diffs
TAKEAWAY_MARGIN_TO_RATING_SCALE = 10.0  # each net takeaway roughly worth ~5 points of value, per standard estimates

# How the three signals combine into one preseason rating. Point
# differential gets the most weight since it's the actual outcome
# (and is available across all preseason games, not just PRE WK3);
# yardage/takeaway margin are diagnostic context from PRE WK3 only.
POINT_DIFF_COMPONENT_WEIGHT = 0.50
YARDAGE_COMPONENT_WEIGHT = 0.25
TAKEAWAY_COMPONENT_WEIGHT = 0.25

DEFAULT_PRESEASON_WEIGHT = 0.10


def compute_preseason_point_differentials(results: list = None) -> dict:
    """
    Average point differential per team across their real preseason
    games. Returns {team: avg_point_diff}.
    """
    results = results or PRESEASON_2026_RESULTS
    totals = {}
    counts = {}

    for home, away, home_score, away_score, week in results:
        diff = home_score - away_score
        totals[home] = totals.get(home, 0) + diff
        counts[home] = counts.get(home, 0) + 1
        totals[away] = totals.get(away, 0) - diff
        counts[away] = counts.get(away, 0) + 1

    return {team: totals[team] / counts[team] for team in totals}


def compute_boxscore_signals(boxscores: list = None) -> dict:
    """
    Yardage margin and takeaway margin per team, from real PRE WK3 box
    scores. Returns {team: {"yardage_margin": x, "takeaway_margin": y}}.

    yardage_margin = team's yards - opponent's yards
    takeaway_margin = opponent's giveaways - team's own giveaways
    (positive = forced more turnovers than committed)
    """
    from model.preseason_wk3_boxscores import PRESEASON_WK3_BOXSCORES
    boxscores = boxscores or PRESEASON_WK3_BOXSCORES

    result = {}
    for away, home, away_yds, home_yds, away_giveaways, home_giveaways in boxscores:
        result[away] = {
            "yardage_margin": away_yds - home_yds,
            "takeaway_margin": home_giveaways - away_giveaways,
        }
        result[home] = {
            "yardage_margin": home_yds - away_yds,
            "takeaway_margin": away_giveaways - home_giveaways,
        }
    return result


def preseason_diff_to_rating(avg_point_diff: float) -> float:
    """Converts average preseason point differential to this model's
    rating scale. See module docstring — not backtested."""
    return avg_point_diff / POINT_DIFF_TO_RATING_SCALE


def compute_combined_preseason_rating(team: str, results: list = None, boxscores: list = None) -> float:
    """
    Combines point differential (all preseason games) with yardage and
    takeaway margin (PRE WK3 box scores only) into one preseason rating
    signal. Falls back to point-differential alone if no box score data
    exists for a team (e.g., a future season with no boxscores file).
    """
    diffs = compute_preseason_point_differentials(results)
    point_component = preseason_diff_to_rating(diffs.get(team, 0.0))

    try:
        box_signals = compute_boxscore_signals(boxscores)
    except ImportError:
        box_signals = {}

    if team not in box_signals:
        return point_component  # no box score data — point differential alone

    yardage_component = box_signals[team]["yardage_margin"] / YARDAGE_MARGIN_TO_RATING_SCALE
    takeaway_component = box_signals[team]["takeaway_margin"] / TAKEAWAY_MARGIN_TO_RATING_SCALE

    return (
        POINT_DIFF_COMPONENT_WEIGHT * point_component
        + YARDAGE_COMPONENT_WEIGHT * yardage_component
        + TAKEAWAY_COMPONENT_WEIGHT * takeaway_component
    )


def apply_preseason_adjustment(
    prior_rating: float,
    team: str,
    preseason_weight: float = DEFAULT_PRESEASON_WEIGHT,
    results: list = None,
    boxscores: list = None,
) -> float:
    """
    Nudges a prior rating (e.g., last season's final rating) using the
    team's real preseason performance — point differential blended with
    yardage/takeaway margin where available — at a small, deliberately
    conservative weight.
    """
    diffs = compute_preseason_point_differentials(results)
    if team not in diffs:
        return prior_rating  # no preseason data for this team — leave prior untouched

    preseason_rating = compute_combined_preseason_rating(team, results, boxscores)
    return (1 - preseason_weight) * prior_rating + preseason_weight * preseason_rating


if __name__ == "__main__":
    diffs = compute_preseason_point_differentials()
    print(f"{len(diffs)} teams with real 2026 preseason data\n")

    print("Combined preseason rating (point diff + PRE WK3 yardage/takeaway margin):")
    combined = {team: compute_combined_preseason_rating(team) for team in diffs}

    print("\nBest combined preseason ratings:")
    for team, rating in sorted(combined.items(), key=lambda x: -x[1])[:5]:
        print(f"  {team}: {rating:+.3f}")

    print("\nWorst combined preseason ratings:")
    for team, rating in sorted(combined.items(), key=lambda x: x[1])[:5]:
        print(f"  {team}: {rating:+.3f}")

    print(f"\nExample blend: a team with a 0.0 prior and a great combined preseason showing:")
    blended = apply_preseason_adjustment(prior_rating=0.0, team="BAL", preseason_weight=0.10)
    print(f"  BAL blended rating: {blended:+.4f}")
