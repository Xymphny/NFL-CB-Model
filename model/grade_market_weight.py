#!/usr/bin/env python3
"""Grade each league's model AGAINST THE MARKET, and write the staking weight.

THE QUESTION
Not "is the model accurate" -- every grade in this repository already asks
that, and most pass. The question staking actually depends on is narrower:
given the closing price, does the model's probability add anything? The
answer is the blend weight core.market_weight estimates, and the number
staking uses is its one-sided lower bound.

HOW EACH BET IS PRICED
Through the production path, not a restatement of it: the league's model
object prices the game, execution.recommend.cover_probability turns that into
P(home covers | no push) at the closing line, and core.pricing.devig removes
the margin from the closing prices with the method the runner uses. Pushes are
dropped -- a push is not a win or a loss for either side. One row per game,
the home side; the away side is its complement and adds no information.

WHAT A LEAGUE NEEDS TO BE GRADED
Model predictions made before each game, and the closing line WITH its prices
for the same games. Leagues without both get no row, and the runner reads a
missing row as weight 0.

  nfl  walk-forward cache 2014-2023 x nflverse closing spreads and prices
  cfb  walk-forward cache x CFBD closing spreads (no prices; see cfb())
  mlb  none: the free Sportsbook Reviews archive the MLB backtest was written
       against now redirects every season to its homepage
  nhl  none: no free historical line source
  nba  none: no free historical line source
Those three are not skipped for lack of trying: the capture job records
closing lines from Thursday on, and this grade runs on them once a season of
settled bets exists.

IN-SAMPLE CONTAMINATION CAN ONLY FLATTER THE MODEL
Where the model's coefficients may have been fitted on the graded seasons,
its predictions are better there than they would be on unseen games, so w
comes out HIGHER than the truth. The estimate is then an upper bound and says
so -- which cuts one way only: an upper bound near zero is a firm zero.

Writes data/market_weights.json.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core import pricing as P  # noqa: E402
from coverline.core.market_weight import estimate  # noqa: E402
from coverline.execution.recommend import cover_probability  # noqa: E402

OUT = ROOT / "data" / "market_weights.json"
DEVIG = "power"          # the runner's default (recommend.price_candidate)


def _devig_home(home_american, away_american) -> float:
    dec = [P.american_to_decimal(float(home_american)),
           P.american_to_decimal(float(away_american))]
    return float(P.devig(dec, DEVIG)[0])


def _rows(model, games: pd.DataFrame, gid, home_line, p_market, asof) -> pd.DataFrame:
    """One graded row per game: p_model, p_market, outcome, season."""
    out = []
    for g in games.itertuples():
        line = home_line(g)
        margin = g.home_score - g.away_score
        if margin + line == 0:
            continue                                   # push
        dist = model.predict(gid(g), asof(g))     # priced as of its own game day
        p_model, _ = cover_probability(dist, line)
        out.append({"season": int(g.season), "p_model": p_model,
                    "p_market": p_market(g), "y": int(margin + line > 0),
                    "line": line})
    return pd.DataFrame(out)


def nfl() -> tuple[pd.DataFrame, dict]:
    from coverline.leagues.nfl.model import NFLModel
    from coverline.leagues.nfl.sources import CachedWalkForwardSource, game_id
    src = CachedWalkForwardSource.load()
    model = NFLModel(src)
    lines = pd.read_parquet(ROOT / "data" / "raw" / "nfl" / "nflverse_lines.parquet")
    lines = lines[lines.game_type == "REG"]
    cache = src.frame.reset_index()
    g = cache.merge(lines, on=["season", "week", "home_team", "away_team"],
                    suffixes=("", "_l"), validate="one_to_one")
    if len(g) != len(cache) or ((g.home_score != g.home_score_l)
                                | (g.away_score != g.away_score_l)).any():
        raise ValueError("the cache and nflverse disagree on which games or scores")
    g = g.dropna(subset=["home_spread_odds", "away_spread_odds"])
    rows = _rows(model, g,
                 gid=lambda r: game_id(r.season, r.week, r.home_team, r.away_team),
                 home_line=lambda r: -float(r.spread_line),
                 p_market=lambda r: _devig_home(r.home_spread_odds, r.away_spread_odds),
                 asof=lambda r: str(r.gameday))
    meta = {
        "market": "spread (closing, nflverse), home side, pushes dropped",
        "model": "NFLModel over model/expanded_walk_forward_cache.csv "
                 "(rating-only vector; no NGS in the cache)",
        "seasons": [int(rows.season.min()), int(rows.season.max())],
        "contamination": (
            "UPPER BOUND. The rating-only coefficients' fit window is not "
            "recorded and no contiguous season range of this cache "
            "reproduces them, so no graded season is provably unseen."),
    }
    return rows, meta


def cfb() -> tuple[pd.DataFrame, dict]:
    """CFBD closing spreads carry NO PRICES, so the market side of each bet is
    taken as 0.5 -- what a standard -110/-110 line devigs to. That is an
    assumption about the price, not a measurement of it, and it is written
    into the result. Its effect is to treat every line as balanced; a line
    shaded to -115 on one side would move p_market by about a point."""
    from coverline.leagues.cfb.model import CFBModel
    from coverline.leagues.cfb.sources import CachedWalkForwardSource, game_id
    src = CachedWalkForwardSource.load()
    model = CFBModel(src)
    lines = pd.read_csv(ROOT / "model" / "cfb_lines_cache.csv")
    cache = src.frame.reset_index()
    g = cache.merge(lines, on=["season", "week", "home_team", "away_team"],
                    suffixes=("", "_l"), validate="one_to_one")
    if ((g.home_score != g.home_score_l) | (g.away_score != g.away_score_l)).any():
        raise ValueError("the cache and the lines cache disagree on scores")
    g = g.dropna(subset=["spread_line"])
    rows = _rows(model, g,
                 gid=lambda r: game_id(r.season, r.week, r.home_team, r.away_team),
                 home_line=lambda r: -float(r.spread_line),
                 p_market=lambda r: 0.5,
                 asof=lambda r: f"{int(r.season)}-12-31")
    meta = {
        "market": "spread (closing, CFBD via model/cfb_lines_cache.csv), home "
                  "side, pushes dropped",
        "market_price": "ASSUMED -110/-110 (p_market = 0.5); the lines carry "
                        "no prices",
        "model": "CFBModel over model/cfb_full_walk_forward_cache.csv "
                 "(DVOA-only vector; the cache carries no Elo)",
        "joined": f"{len(g)} of {len(cache)} cache games have a closing spread",
        "seasons": [int(rows.season.min()), int(rows.season.max())],
        "contamination": (
            "UPPER BOUND. The DVOA-only vector was fitted on this cache's "
            "seasons, so the model is graded partly on data it was fitted to."),
    }
    return rows, meta


def grade(rows: pd.DataFrame) -> dict:
    r = estimate(rows.p_model, rows.p_market, rows.y)
    by_season = {}
    for s, part in rows.groupby("season"):
        if len(part) >= 30:
            e = estimate(part.p_model, part.p_market, part.y)
            by_season[str(s)] = {"n": e.n, "w_hat": round(e.w_hat, 3),
                                 "se": round(e.se, 3)}
    # The plain version of the same question, for a reader who distrusts the
    # blend: when the model disagrees with the line, does it win more often?
    lean = np.sign(rows.p_model - rows.p_market)
    hit = ((lean > 0) & (rows.y == 1)) | ((lean < 0) & (rows.y == 0))
    return {**r.to_dict(), "by_season": by_season,
            "side_agreement_hit_rate": round(float(hit[lean != 0].mean()), 4),
            "break_even_at_minus_110": 0.5238}


LEAGUES = {"nfl": nfl, "cfb": cfb}


def main() -> int:
    out = {
        "_provenance": {
            "script": "model/grade_market_weight.py",
            "estimator": "src/coverline/core/market_weight.py",
            "generated": date.today().isoformat(),
            "devig": DEVIG,
            "staking_weight": "one-sided 95% lower bound of w_hat, clipped to [0, 1]",
            "missing_league_means": "weight 0 -- no market-facing evidence",
        },
        "leagues": {},
    }
    for name, build in LEAGUES.items():
        rows, meta = build()
        out["leagues"][name] = {**meta, **grade(rows)}
        g = out["leagues"][name]
        print(f"{name}: n={g['n']} w_hat={g['w_hat']:+.3f} se={g['se']:.3f} "
              f"-> staking {g['staking_weight']:.3f}  gain t={g['gain_t']:+.2f}  "
              f"side hit {g['side_agreement_hit_rate']:.4f}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
