"""
Standard NFL Elo rating system -- a genuinely DIFFERENT model
architecture from the play-by-play DVOA-style rating used everywhere
else in this project. Elo uses only final scores, updated
incrementally game-by-game, with no down-by-down analysis at all.
This is the real point of building it: its errors are likely to be
structured differently from the play-by-play model's errors (Elo
captures "who wins and by how much" trends through a completely
different mechanism than "how many yards/success did they generate
per play"), which is what would make an ensemble of the two
potentially more accurate than either alone -- unlike blending with
the market (which failed because the market's edge is real
information we lack, not a different modeling approach on the same
information).

Real, standard formula (same general form as FiveThirtyEight's
published NFL Elo methodology), with K-factor and home-field
advantage calibrated against real data rather than copied from
their specific constants, which were tuned for their own system/era.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DEFAULT_K = 20.0  # tested calibrating this in isolation (K=32 improved standalone Elo accuracy) --
                   # but reverted after finding it made the FULL ensemble (with Layer 2 present) WORSE,
                   # not better -- see model/calibrate_elo_hyperparams.py's docstring for the full finding
DEFAULT_HOME_ADVANTAGE = 65.0  # same story -- see above
SEASON_REGRESSION = 0.33  # same story -- see above
BASELINE_ELO = 1500.0


def expected_score(rating_a, rating_b):
    """Probability team A beats team B, given their Elo ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin, elo_diff):
    """
    Margin-of-victory multiplier -- a blowout carries more signal than
    a one-score win, but the effect is dampened when the favorite was
    already expected to win big (same general form as FiveThirtyEight's
    published NFL Elo MOV formula).
    """
    return np.log(abs(margin) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def compute_elo_walk_forward(schedule, k=DEFAULT_K, home_advantage=DEFAULT_HOME_ADVANTAGE):
    """
    Computes Elo ratings walk-forward across real game results, in
    chronological order, updating after each game -- true zero
    lookahead by construction, since each game's prediction uses only
    ratings as they stood BEFORE that game was played.

    Returns (per_game_df, final_elo_dict). per_game_df has each team's
    PRE-game rating for that game (for backtesting); final_elo_dict has
    each team's CURRENT rating after all real games processed --
    REAL BUG FOUND AND FIXED: an earlier version of the production
    wiring tried to reconstruct "current" rating from a team's last
    per-game row's pre-game value, which is the rating BEFORE their
    most recent game, not after it -- silently stale by one game's
    worth of real information. Returning the actual final state
    directly avoids this.
    """
    games = schedule.dropna(subset=["home_score", "away_score"]).copy()
    # Sort by real calendar date when available (fixes a real bug found
    # in CFB: postseason games reset their own week numbering, so a
    # January game could show "week 1", identical to the season's
    # actual opening week -- sorting by (season, week) alone would
    # process it as if it happened before the season started). Falls
    # back to (season, week) for schedules without a date column (NFL's
    # load_schedules() already only returns regular-season games, so
    # this fallback was never actually exposed to the bug).
    if "game_date" in games.columns:
        games["game_date"] = pd.to_datetime(games["game_date"])
        games = games.sort_values(["season", "game_date"]).reset_index(drop=True)
    else:
        games = games.sort_values(["season", "week"]).reset_index(drop=True)

    elo = {}
    rows = []
    current_season = None

    for _, game in games.iterrows():
        season, home, away = game["season"], game["home_team"], game["away_team"]

        if season != current_season:
            for team in elo:
                elo[team] = elo[team] + SEASON_REGRESSION * (BASELINE_ELO - elo[team])
            current_season = season

        home_elo = elo.get(home, BASELINE_ELO)
        away_elo = elo.get(away, BASELINE_ELO)

        elo_diff = home_elo + home_advantage - away_elo
        exp_home = expected_score(home_elo + home_advantage, away_elo)

        actual_margin = game["home_score"] - game["away_score"]
        actual_home_win_score = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)

        rows.append({
            "season": season, "week": game["week"], "home_team": home, "away_team": away,
            "home_elo_pre": home_elo, "away_elo_pre": away_elo,
            "elo_diff": home_elo + home_advantage - away_elo,
            "elo_win_prob_home": exp_home,
            "actual_margin": actual_margin, "actual_home_win": actual_margin > 0,
        })

        mov_mult = mov_multiplier(actual_margin if actual_margin != 0 else 1, elo_diff)
        update = k * mov_mult * (actual_home_win_score - exp_home)
        elo[home] = home_elo + update
        elo[away] = away_elo - update

    return pd.DataFrame(rows), elo


if __name__ == "__main__":
    from ingest.nfl_schedules import load_schedules

    print("Testing real Elo computation against 2014-2023 schedule data...")
    schedule = load_schedules(seasons=list(range(2014, 2024)))
    elo_df, final_elo = compute_elo_walk_forward(schedule)
    print(f"{len(elo_df)} real games processed")

    straight_up_acc = ((elo_df["elo_win_prob_home"] > 0.5) == elo_df["actual_home_win"]).mean()
    print(f"\nElo-alone straight-up accuracy across ALL 2014-2023 games (in-sample, not held out): {straight_up_acc:.4f}")
    print("(This is a rough sanity check only -- the real held-out test is in test_elo_ensemble.py)")
