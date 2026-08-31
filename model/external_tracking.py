"""
External accuracy tracking against ESPN FPI — Section 11.5.

IMPORTANT: this module has NOT been run in the build sandbox.
espn.com is not reachable from this environment's network allowlist
(only github.com, pypi, npm, and similar package-registry domains are
permitted). Test this directly in your own environment before relying
on it, and note Section 7's ToS consideration: scraping ESPN is
generally against their terms of service even though this is commonly
done for personal projects.
"""

import requests
import pandas as pd

# ESPN's FPI is exposed through an undocumented public JSON endpoint
# that tools like espnscrapeR use — this is the same endpoint pattern,
# not independently verified against a live response in this sandbox.
ESPN_FPI_URL = "https://site.api.espn.com/apis/v2/sports/football/nfl/fpi"


def fetch_espn_fpi() -> pd.DataFrame:
    """
    Pull current ESPN FPI ratings.

    NOTE: untested. ESPN's undocumented endpoints change without notice
    more often than a public API would — expect this to need adjustment
    against a real response before it works. If it breaks, inspecting
    the network tab on ESPN's own FPI page (espn.com/nfl/fpi) while it
    loads is the standard way to find the current endpoint/shape.
    """
    resp = requests.get(ESPN_FPI_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Structure is a guess based on ESPN's typical response shape for
    # similar endpoints — verify against the real response and adjust.
    teams = data.get("teams", [])
    rows = []
    for team in teams:
        rows.append({
            "team": team.get("team", {}).get("abbreviation"),
            "fpi_rating": team.get("stats", {}).get("fpi"),
            "fpi_offense": team.get("stats", {}).get("fpi_offense"),
            "fpi_defense": team.get("stats", {}).get("fpi_defense"),
        })
    return pd.DataFrame(rows)


def score_predictions(predictions: pd.DataFrame) -> dict:
    """
    Section 11.5 — score the model, FPI, and the market against actual
    results once games complete.

    Parameters
    ----------
    predictions : DataFrame with columns:
        model_win_prob, fpi_win_prob, market_win_prob (de-vigged),
        model_spread, fpi_spread, market_spread,
        model_total, market_total,
        actual_home_win (bool), actual_margin, actual_total

    Returns
    -------
    dict of scoring metrics per source: Brier score (win probability
    calibration) and MAE (spread/total error).
    """
    def brier_score(probs, outcomes):
        return ((probs - outcomes.astype(float)) ** 2).mean()

    def mae(predicted, actual):
        return (predicted - actual).abs().mean()

    results = {}
    for source in ["model", "fpi", "market"]:
        win_prob_col = f"{source}_win_prob"
        spread_col = f"{source}_spread"
        if win_prob_col in predictions.columns:
            results[f"{source}_brier"] = brier_score(
                predictions[win_prob_col], predictions["actual_home_win"]
            )
        if spread_col in predictions.columns:
            results[f"{source}_spread_mae"] = mae(
                predictions[spread_col], predictions["actual_margin"]
            )

    total_col_model = "model_total"
    if total_col_model in predictions.columns:
        results["model_total_mae"] = mae(predictions[total_col_model], predictions["actual_total"])
    if "market_total" in predictions.columns:
        results["market_total_mae"] = mae(predictions["market_total"], predictions["actual_total"])

    return results


if __name__ == "__main__":
    print("Attempting to fetch ESPN FPI — UNTESTED, expect this to need fixing:")
    try:
        fpi = fetch_espn_fpi()
        print(fpi.head())
    except Exception as e:
        print(f"Failed as expected in this sandbox (ESPN not reachable): {e}")
