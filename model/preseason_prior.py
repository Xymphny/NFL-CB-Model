"""
Preseason prior / credibility weighting — Section 11.1.

Blends a prior-season rating with the current season's in-progress
rating, with the prior's influence shrinking automatically as more
real games accumulate. The credibility weight formula:

    in_season_weight = games_played / (games_played + k)
    prior_weight = 1 - in_season_weight

k is "how many real games is the prior worth" — calibrated via
backtesting in calibrate_credibility_k.py, not guessed.

HONEST SCOPE NOTE: the full Section 11.1 design also calls for
blending in Vegas win-total-implied strength, especially for CFB
where roster turnover is severe. That requires historical preseason
betting lines, which need a paid Odds API tier we don't have access
to — vegas_win_total is a supported optional input below, but nothing
here fabricates that data. Without it, the prior is last-season's
rating alone, which is what's actually been calibrated against real
data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_K = 2.0
"""
Calibrated against real 2021-2023 data (see calibrate_credibility_k.py) —
not guessed. k=2 gave the best overall error reduction (11.1% vs. no
blending) across a backtest checking how close early-season blended
ratings got to each team's true final-season rating. Notably, blending
helps most exactly when theory predicts it should: +16.2% at Week 2,
fading to +2.8% by Week 6 — and larger k values actively hurt once real
in-season data exists to trust instead (k=32 was 40% worse by Week 6).
Recalibrate if the model's own methodology changes enough that this
backtest's assumptions no longer hold.
"""


def credibility_weight(games_played: int, k: float = DEFAULT_K) -> float:
    """Fraction of trust placed in the in-season rating vs. the prior."""
    return games_played / (games_played + k)


def vegas_win_total_to_rating(win_total: float, league: str = "NFL") -> float:
    """
    Converts a Vegas win-total line into this model's rating scale.

    NOT calibrated against real historical data — no historical
    preseason lines available without a paid Odds API tier (see module
    docstring). The linear mapping below (centered on a .500 win total,
    scaled roughly to match the rating range seen in real computed
    ratings) is a reasonable placeholder, not a validated coefficient.
    Treat any output from this function as a rough estimate until it
    can be backtested against real historical lines.
    """
    if league.upper() == "CFB":
        games_in_season = 12
    else:
        games_in_season = 17

    win_pct = win_total / games_in_season
    # Centers on .500 = rating 0, scaled so a strong team (.700ish) lands
    # near +0.2, roughly matching real DVOA-style rating ranges observed
    # in actual computed output — a reasonable starting scale, not a
    # calibrated one.
    return (win_pct - 0.5) * 1.0


# YEAR-OVER-YEAR CARRYOVER (2026-09-20). The prior was applied RAW --
# effective_prior = prior_rating -- which asserts that last season's
# rating carries forward whole. Measured on 2017-2022 it carries at
# 0.441 (SE 0.063), and below a year-over-year correlation of 0.5 a
# slope of 1.0 is worse than a slope of ZERO: on held-out 2023-2025 the
# raw prior scored MAE 0.0913 against 0.0846 for simply predicting the
# league mean. It was worse than having no prior at all.
#
# Regressing it wins on the same held-out seasons: MAE 0.0785, a 13.9%
# improvement, paired gain +0.0128 (SE 0.0056, t = +2.28), better in
# each of the three seasons separately. Offense and defense are
# regressed apart because defense carries over less -- a real fact
# about football, not a fitting artifact -- and total stays exactly
# offense minus defense, as it already did.
#
# The intercepts are ~0 because VOA is centered; they are carried
# anyway so the fit is applied as fitted rather than as assumed.
# model/preseason_prior_regression.py regenerates all of this.
PRIOR_CARRYOVER = {
    "offense_voa": (0.432, -0.00333),
    "defense_voa": (0.3476, 4e-05),
    "total_rating": (0.4415, -0.00332),
}


def regress_prior(prior_rating: float, component: str = "total_rating") -> float:
    """Pull a prior-season rating toward the league mean by its measured
    carryover. An unknown component is returned untouched rather than
    silently regressed by the wrong coefficient."""
    if component not in PRIOR_CARRYOVER:
        return prior_rating
    slope, intercept = PRIOR_CARRYOVER[component]
    return slope * prior_rating + intercept


def blend_rating(
    prior_rating: float,
    in_season_rating: float,
    games_played: int,
    k: float = DEFAULT_K,
    vegas_win_total: float = None,
    vegas_weight: float = 0.0,
    league: str = "NFL",
    carryover: str = None,
) -> float:
    """
    The actual blend. If vegas_win_total is provided, it's blended into
    the prior itself at vegas_weight (default 0.0 — off unless real
    line data is supplied) before the credibility weighting is applied.
    """
    # carryover=None leaves the prior untouched, which is what
    # calibrate_credibility_k.py needs: k=2 was fitted against the raw
    # prior, so that script must keep measuring the thing it measured.
    effective_prior = prior_rating if carryover is None else regress_prior(
        prior_rating, carryover)
    if vegas_win_total is not None and vegas_weight > 0:
        vegas_rating = vegas_win_total_to_rating(vegas_win_total, league=league)
        effective_prior = (1 - vegas_weight) * prior_rating + vegas_weight * vegas_rating

    weight = credibility_weight(games_played, k)
    return weight * in_season_rating + (1 - weight) * effective_prior


def blend_team_ratings(
    prior_ratings: "pd.DataFrame",
    in_season_ratings: "pd.DataFrame",
    games_played: int,
    k: float = DEFAULT_K,
) -> "pd.DataFrame":
    """
    Applies the blend across a full team-ratings table, for wiring into
    weekly_job.py.

    Blends offense_voa and defense_voa separately (same k extended by
    reasonable assumption, since it's the same underlying mechanism)
    and derives total_rating from the blended components, rather than
    blending total_rating directly — this keeps offense - defense =
    total internally consistent in the output. Note: k was calibrated
    specifically against total_rating error (see calibrate_credibility_k.py);
    applying it to the components separately is an extension, not an
    independently validated result on its own.
    """
    import pandas as pd

    blended = in_season_ratings.copy()
    for team in blended.index:
        if team not in prior_ratings.index:
            continue  # no prior available (e.g., a relocated/renamed team) — leave in-season rating as-is

        blended.loc[team, "offense_voa"] = blend_rating(
            prior_ratings.loc[team, "offense_voa"], in_season_ratings.loc[team, "offense_voa"],
            games_played=games_played, k=k, carryover="offense_voa",
        )
        blended.loc[team, "defense_voa"] = blend_rating(
            prior_ratings.loc[team, "defense_voa"], in_season_ratings.loc[team, "defense_voa"],
            games_played=games_played, k=k, carryover="defense_voa",
        )

    blended["total_rating"] = blended["offense_voa"] - blended["defense_voa"]
    return blended


if __name__ == "__main__":
    # Sanity check: as games_played increases, the blend should converge
    # toward the in-season rating and away from the prior.
    prior = 0.10   # a good team last season
    in_season = -0.05  # struggling so far this season

    print("games_played | credibility_weight | blended_rating")
    for games in [0, 1, 2, 4, 8, 16]:
        w = credibility_weight(games, k=4.0)
        blended = blend_rating(prior, in_season, games, k=4.0)
        print(f"{games:>12} | {w:>18.3f} | {blended:>14.4f}")
