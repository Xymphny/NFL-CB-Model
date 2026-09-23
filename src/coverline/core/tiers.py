"""Tier: how strongly the model disagrees with the market on one game.

    play       its strongest disagreements for the league (top tenth)
    lean       a real but smaller disagreement
    coin_flip  near agreement; names the side it slightly prefers
    no_edge    it broadly agrees with the market

CONVICTION, NOT EDGE. The bands are quantiles of the league's own history of
|p_model - p_market| (model/derive_tier_thresholds.py, ADR 0025), and in every
league measured the side the model prefers has won about as often in the top
band as in the bottom one. Whether any of it is worth money is the market
grade's call (core/market_weight.py): a tier never implies a stake.

The site renders these; it never computes them (the dashboard brief's one
rule). Thresholds live in data/tier_thresholds.json.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_PATH = _ROOT / "data" / "tier_thresholds.json"

TIERS = ("play", "lean", "coin_flip", "no_edge")


class NoThresholds(KeyError):
    """No bands for this league -- the board says so rather than guessing."""


def thresholds(league: str, path: Path = THRESHOLDS_PATH) -> dict:
    art = json.loads(Path(path).read_text())
    row = art.get("leagues", {}).get(league)
    if row is None:
        raise NoThresholds(f"no tier thresholds for {league}")
    if not 0 <= row["coin_flip"] <= row["lean"] <= row["play"] <= 1:
        raise ValueError(f"{league}: tier thresholds are not ordered: {row}")
    return row


def tier(edge: float, bands: dict) -> str:
    """Tier for the preferred side's edge in probability points (0-1)."""
    a = abs(float(edge))
    if a >= bands["play"]:
        return "play"
    if a >= bands["lean"]:
        return "lean"
    if a >= bands["coin_flip"]:
        return "coin_flip"
    return "no_edge"
