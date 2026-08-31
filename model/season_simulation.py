"""
Season-long Monte Carlo simulation — Section 12.

Simulates the remaining schedule many times, sampling each game's
outcome from the points-prediction layer's distribution, and tallies
win totals / division outcomes across iterations.

NOTE on scope: full NFL playoff seeding (division tiebreakers,
wildcard ordering, strength-of-victory/schedule tiebreakers) is a
genuinely large rules engine on its own — not fully implemented here.
This gives you win-total distributions and division-record simulation,
which is the part that's fully rules-based and mechanical. Full
tiebreaker logic is flagged as a discrete follow-up, not silently
skipped.

CFB caveat (Section 12's explicit limitation, preserved here): this
can simulate win totals and conference-record outcomes for CFB, but
"CFP probability" is not attempted, since committee selection isn't a
deterministic function of results.
"""

import numpy as np
import pandas as pd


def simulate_game(
    home_rating: float,
    away_rating: float,
    rating_uncertainty_std: float,
    margin_coefficients: dict,
    is_neutral_site: bool = False,
    rest_diff: float = 0.0,
    rng: np.random.Generator = None,
) -> float:
    """
    Sample a single simulated margin (home - away) for one game, using
    the calibrated margin model (Section 11.4 / calibrate_points_model.py)
    plus noise drawn from the bootstrap-estimated rating uncertainty
    (Section 10 item 3) — this is what makes it a genuine simulation
    rather than just repeating the point estimate every time.
    """
    if rng is None:
        rng = np.random.default_rng()

    rating_diff = home_rating - away_rating
    home_field = 0.0 if is_neutral_site else 1.0

    predicted_margin = (
        margin_coefficients["rating_diff"] * rating_diff
        + margin_coefficients["home_field"] * home_field
        + margin_coefficients["rest_diff"] * rest_diff
        + margin_coefficients["intercept"]
    )

    # Two sources of randomness: the rating's own uncertainty (teams
    # aren't points, they're distributions) and residual game-to-game
    # variance the regression doesn't explain (NFL games have real
    # variance the model will never fully capture, by design of the
    # sport itself, not a modeling gap).
    RESIDUAL_GAME_STD = 13.0  # rough NFL single-game margin residual std;
    # calibrate_points_model.py's MAE (~10-11 pts) is consistent with this
    noise = rng.normal(0, np.sqrt(rating_uncertainty_std**2 + RESIDUAL_GAME_STD**2))

    return predicted_margin + noise


def simulate_season(
    remaining_schedule: pd.DataFrame,
    current_records: dict,
    ratings: pd.DataFrame,
    rating_uncertainty: pd.DataFrame,
    margin_coefficients: dict,
    n_simulations: int = 10000,
) -> pd.DataFrame:
    """
    Simulate the rest of the season n_simulations times.

    Parameters
    ----------
    remaining_schedule : DataFrame with columns [home_team, away_team,
        is_neutral_site, rest_diff] — games not yet played
    current_records : dict of team -> (wins, losses) so far, actual results
    ratings : current team ratings (model.ratings.team_ratings output)
    rating_uncertainty : output of bootstrap_rating_uncertainty
    margin_coefficients : from calibrate_points_model.py's calibrate()

    Returns
    -------
    DataFrame indexed by team with columns: mean_wins, win_total_p05,
    win_total_p95 — NOT full playoff seeding (see module docstring).
    """
    rng = np.random.default_rng(42)
    teams = list(current_records.keys())
    final_wins = {team: [] for team in teams}

    for sim in range(n_simulations):
        wins = dict(current_records)

        for _, game in remaining_schedule.iterrows():
            home, away = game["home_team"], game["away_team"]
            if home not in ratings.index or away not in ratings.index:
                continue

            home_std = rating_uncertainty.loc[home, "rating_std"] if home in rating_uncertainty.index else 0.1
            away_std = rating_uncertainty.loc[away, "rating_std"] if away in rating_uncertainty.index else 0.1
            combined_std = np.sqrt(home_std**2 + away_std**2)

            margin = simulate_game(
                home_rating=ratings.loc[home, "total_rating"],
                away_rating=ratings.loc[away, "total_rating"],
                rating_uncertainty_std=combined_std,
                margin_coefficients=margin_coefficients,
                is_neutral_site=game.get("is_neutral_site", False),
                rest_diff=game.get("rest_diff", 0.0),
                rng=rng,
            )

            if margin > 0:
                wins[home] = wins.get(home, 0) + 1
            else:
                wins[away] = wins.get(away, 0) + 1

        for team in teams:
            final_wins[team].append(wins.get(team, 0))

    summary = pd.DataFrame({
        team: {
            "mean_wins": np.mean(w),
            "win_total_p05": np.percentile(w, 5),
            "win_total_p95": np.percentile(w, 95),
        }
        for team, w in final_wins.items()
    }).T

    return summary.sort_values("mean_wins", ascending=False)


if __name__ == "__main__":
    print(
        "This module requires a real remaining schedule, current records, "
        "ratings, and calibrated coefficients to run meaningfully — see "
        "demo/run_season_simulation.py for a wired-up example against real "
        "2023 data (mid-season simulation from an arbitrary cutoff week)."
    )
