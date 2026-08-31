"""
Player props — Section 13.

UNTESTED beyond the design level. Needs a live Odds API key to verify:
(1) whether player props require a plan tier above the $30/mo 20K tier
used for team markets, and (2) actual CFB player-prop coverage depth.
Do that verification before building further against this.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def fetch_player_props(sport_key: str, event_id: str, markets: list, api_key: str) -> dict:
    """
    Pull player prop odds for a specific event.

    sport_key: "americanfootball_nfl" or "americanfootball_ncaaf"
    markets: e.g. ["player_pass_yds", "player_rush_yds", "player_receptions"]
        — exact market key names need verification against the live API
        docs, these are best-guess based on The Odds API's documented
        naming convention for other sports.

    NOTE: player props are typically requested per-event (not
    league-wide like the team markets), so this is a different call
    pattern from ingest/cfb_pbp.py's odds-equivalent — budget credits
    accordingly (Section 9.1's 3-4 refreshes/day was sized around
    team-level markets, not per-event player props).
    """
    import requests
    url = f"{ODDS_API_BASE}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def compute_usage_rate(snap_counts: pd.DataFrame, targets: pd.DataFrame = None) -> pd.DataFrame:
    """
    Per-player usage rate — snap share as a baseline proxy (Section 13's
    "usage rate" input). Target share would need weekly targets data
    (not yet wired in; nflverse's player_stats release has this — not
    pulled here, flagged as the next piece needed for a real projection
    rather than just this snap-share placeholder).
    """
    usage = snap_counts.groupby(["player", "team", "position"]).agg(
        avg_offense_pct=("offense_pct", "mean"),
        games=("week", "nunique"),
    ).reset_index()
    return usage


def project_player_stat_distribution(
    player_usage_rate: float,
    team_offense_rating: float,
    opponent_defense_rating_vs_position: float,
    league_average_stat_per_snap: float,
    games_sample_size: int,
) -> tuple:
    """
    Placeholder projection — mean and std for a player's expected stat
    in a given matchup. This is a simple linear combination, not a
    properly calibrated model (that calibration is the same kind of
    exercise as calibrate_points_model.py, just at player granularity —
    not done here, since it needs a full season of player-level target/
    carry data this pass didn't pull).

    Returns (projected_mean, projected_std).
    """
    projected_mean = (
        player_usage_rate
        * league_average_stat_per_snap
        * (1 + team_offense_rating)
        * (1 + opponent_defense_rating_vs_position)
    )
    # Smaller sample = more uncertainty — crude placeholder, not a real
    # calibrated variance estimate.
    projected_std = projected_mean * (0.3 + 1.0 / max(games_sample_size, 1))
    return projected_mean, projected_std


def prop_over_probability(line: float, projected_mean: float, projected_std: float) -> float:
    """
    Probability of going over a given prop line, assuming a normal
    distribution around the projection — a simplification most real
    prop models refine per stat type (e.g. receptions are closer to
    Poisson-distributed than normal), not addressed in this pass.
    """
    from scipy.stats import norm
    return 1 - norm.cdf(line, loc=projected_mean, scale=projected_std)


if __name__ == "__main__":
    print(
        "This module requires a live Odds API key to test the fetch "
        "functions, and a full season of player target/carry data (not "
        "yet ingested) to properly calibrate project_player_stat_distribution. "
        "Structural placeholders only — verify coverage and pricing tier "
        "(see module docstring) before investing further build time here."
    )
