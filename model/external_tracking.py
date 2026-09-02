"""
External accuracy tracking against ESPN FPI — Section 11.5.

VERIFIED against the real, live page (espn.com/nfl/fpi) via Claude in
Chrome — the original version of this module was a complete guess
(a nonexistent site.api.espn.com JSON endpoint) and was wrong on both
counts: there is no separate API call for this data at all, and the
guessed response shape didn't match anything real.

Real structure, confirmed directly: the page is server-rendered with
the full dataset embedded inline as `window['__espnfitt__'] = {...}`
in a <script> tag — no XHR/fetch request happens for it (confirmed via
network-request monitoring while the page loaded: nothing but ad/
analytics tracking pixels fired). The real path to the data is
page.content.table.stats, an object keyed "0" through "31", each
value shaped like:
    {"team": {"abbrev": "LAR", "displayName": "Los Angeles Rams", ...},
     "stats": [{"name": "fpi", "value": "5.9"}, {"name": "epaoffense", "value": "4.1"}, ...]}
(field names use "abbrev", not "abbreviation"; stats is a list of
{name, value} pairs, not a nested dict — both wrong in the original
guess.)
"""

import re
import json
import requests
import pandas as pd

ESPN_FPI_PAGE_URL = "https://www.espn.com/nfl/fpi"


def fetch_espn_fpi() -> pd.DataFrame:
    """
    Pulls current ESPN FPI ratings by extracting the real embedded
    __espnfitt__ data blob from the page's HTML — verified structure,
    see module docstring. Requests the page directly (not an API
    endpoint, since none exists for this data).
    """
    resp = requests.get(ESPN_FPI_PAGE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    html = resp.text

    marker = "window['__espnfitt__']="
    start = html.find(marker)
    if start == -1:
        raise ValueError(
            "Could not find the __espnfitt__ data blob in the page — "
            "ESPN may have changed their page structure since this was verified"
        )
    start += len(marker)

    # The JSON object runs until the matching closing brace; find it by
    # brace-counting rather than assuming a fixed end marker, since the
    # object contains nested braces throughout.
    depth = 0
    end = start
    for i, ch in enumerate(html[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    data = json.loads(html[start:end])
    stats_by_index = data["page"]["content"]["table"]["stats"]

    rows = []
    for entry in stats_by_index.values():
        team_abbrev = entry["team"]["abbrev"]
        stat_lookup = {s["name"]: s["value"] for s in entry["stats"]}
        rows.append({
            "team": team_abbrev,
            "fpi_rating": float(stat_lookup.get("fpi", "nan")),
            "fpi_offense": float(stat_lookup.get("epaoffense", "nan")),
            "fpi_defense": float(stat_lookup.get("epadefense", "nan")),
            "fpi_special_teams": float(stat_lookup.get("epaspecialteams", "nan")),
            "fpi_rank": int(stat_lookup.get("fpirank", "0")),
        })

    return pd.DataFrame(rows).set_index("team")


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
