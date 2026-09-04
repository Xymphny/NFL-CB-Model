"""
Derivative pricing from the empirical margin distribution -- the
machinery that turns one model margin into fair prices for EVERY
spread-family market: alt spreads, moneylines, half-point values.

Core idea: actual margin = point_estimate + R, where R follows the
empirical ATS residual distribution (data/margin_dist.json, 4,078
games 2010-2024, real key-number mass at 3 and 7 included). Any
derivative is then a probability statement about that distribution:

  P(home covers alt line L) = P(R > L - point_estimate), pushes split out.

Validated below against reality, not assumed: the module's
spread->moneyline conversion is checked against the actual historical
win rate of teams closing at each spread. A pricing engine whose -3
conversion disagrees with 25 years of -3 favorites is broken no matter
how elegant the math.

Scope note: team totals and first-half lines need the JOINT
margin/total distribution and real half-scoring data respectively --
deliberately not faked here with an independence assumption.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DIST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "margin_dist.json")


def load_residual_pmf(path=DIST_PATH):
    with open(path) as f:
        dist = json.load(f)
    pmf = {float(k): v for k, v in dist["residual_distribution"]["residual_pmf"].items()}
    total = sum(pmf.values())
    return {k: v / total for k, v in pmf.items()}


def cover_push_prob(point_estimate, line, pmf):
    """P(home side covers line), P(push). Home covers when margin > line."""
    need = line - point_estimate
    cover = sum(p for r, p in pmf.items() if r > need)
    push = sum(p for r, p in pmf.items() if r == need)
    return cover, push


def prob_to_american(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def fair_price(point_estimate, line, pmf, side="home"):
    """Fair (no-vig) two-way price at a given line, pushes excluded."""
    cover, push = cover_push_prob(point_estimate, line, pmf)
    p_home = cover / (1 - push) if push < 1 else 0.5
    p = p_home if side == "home" else 1 - p_home
    return {"prob": p, "american": prob_to_american(p), "push_prob": push}


def load_ml_scale(path=DIST_PATH):
    with open(path) as f:
        return json.load(f)["residual_distribution"].get("ml_dispersion_scale", 1.0)


def moneyline_from_margin(point_estimate, pmf, scale=None):
    """Win probability from a point estimate. Uses the dispersion scale
    calibrated against 2010-2024 favorite win rates (see margin_dist
    .json's ml_scale_note) -- the unscaled pooled residuals price
    favorites measurably too flat (-7 came out 72.6% vs 75.8% real)."""
    if scale is None:
        scale = load_ml_scale()
    win = sum(p for r, p in pmf.items() if r * scale > -point_estimate)
    tie = sum(p for r, p in pmf.items() if r * scale == -point_estimate)
    p = win / (1 - tie) if tie < 1 else 0.5
    return {"prob": p, "american": prob_to_american(p), "push_prob": tie}


def alt_line_sheet(point_estimate, market_line, pmf, span=3.0, step=0.5):
    """Fair prices at every half-point from market-span to market+span."""
    rows = []
    for offset in np.arange(-span, span + step / 2, step):
        line = round((market_line + offset) * 2) / 2
        fp = fair_price(point_estimate, line, pmf, side="home")
        rows.append({"line": line, "home_fair_prob": round(fp["prob"], 4),
                     "home_fair_price": fp["american"], "push_prob": round(fp["push_prob"], 4)})
    return rows


def half_point_value(point_estimate, from_line, to_line, pmf):
    """Cover-probability gain moving from one line to another --
    directly answers 'is buying the hook at -120 worth it here'."""
    a = fair_price(point_estimate, from_line, pmf)["prob"]
    b = fair_price(point_estimate, to_line, pmf)["prob"]
    return b - a


def validate_against_history(games_csv):
    """The load-bearing check: module's spread->win-prob vs the real
    historical win rate of teams closing at that spread."""
    pmf = load_residual_pmf()
    g = pd.read_csv(games_csv)
    g = g[(g["game_type"] == "REG") & g["season"].between(2010, 2024)].dropna(subset=["spread_line", "home_score", "away_score"])
    g = g[g["home_score"] != g["away_score"]]
    g["home_win"] = g["home_score"] > g["away_score"]

    print(f"{'close':>6} {'n':>5} {'actual win%':>12} {'module':>8} {'gap':>6}")
    for s in [1, 2.5, 3, 3.5, 6.5, 7, 9.5, 13.5]:
        sub = g[g["spread_line"] == s]
        if len(sub) < 40:
            continue
        actual = sub["home_win"].mean()
        # Module's view: a team closing -s has point_estimate s.
        module = moneyline_from_margin(s, pmf)["prob"]
        print(f"{s:>6} {len(sub):>5} {actual*100:>11.1f}% {module*100:>7.1f}% {abs(actual-module)*100:>5.1f}")


if __name__ == "__main__":
    pmf = load_residual_pmf()
    print("Validation -- module moneyline conversion vs 15 seasons of reality:")
    validate_against_history(os.environ.get("GAMES_CSV", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))

    print("\nDemo: model makes a game home -6.0, market has -4.5:")
    for row in alt_line_sheet(6.0, 4.5, pmf, span=2.0):
        print(f"  home {row['line']:+.1f}: fair {row['home_fair_price']:+d} ({row['home_fair_prob']*100:.1f}%, push {row['push_prob']*100:.1f}%)")
    hp = half_point_value(6.0, 3.0, 2.5, pmf)
    print(f"  buying -3 down to -2.5 here adds {hp*100:.1f}% cover probability"
          f" -- breakeven juice cost {prob_to_american(0.5238 - hp) if hp else 'n/a'}")


def team_total_fair(model_total, model_margin, line, side="home", over=True, path=DIST_PATH):
    """Fair price for a team-total using the directly-measured joint
    residual (see margin_dist.json's team_total_residual note)."""
    with open(path) as f:
        tt = json.load(f)["team_total_residual"]
    pmf = {float(k): v for k, v in tt["pmf"].items()}
    est = (model_total + model_margin) / 2 if side == "home" else (model_total - model_margin) / 2
    need = line - est
    over_p = sum(p for r, p in pmf.items() if r > need)
    push = sum(p for r, p in pmf.items() if r == need)
    p = over_p / (1 - push) if push < 1 else 0.5
    p = p if over else 1 - p
    return {"prob": p, "american": prob_to_american(p), "push_prob": push, "point_estimate": est}
