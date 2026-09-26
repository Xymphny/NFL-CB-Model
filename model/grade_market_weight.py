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
  nba  ESPN public closing records 2023-24 on (model/ingest/espn_closes.py)
       x the live walk-forward replayed before each tip; spread
  nhl  the same records x NHLLiveSource and the rules layer; moneyline
  mlb  the same records x the walk-forward with actual starters; moneyline
The free Sportsbook Reviews archive the MLB backtest was written against now
redirects every season to its homepage; ESPN's records replaced it
(2026-09-25). One retail book's close per game, named in each grade.

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
        # The two margins, home side, kept for the board's outlier threshold
        # (derive_tier_thresholds.outlier_points). Not used by the grade.
        out.append({"season": int(g.season), "p_model": p_model,
                    "p_market": p_market(g), "y": int(margin + line > 0),
                    "line": line, "model_margin": float(dist.margin_mean()),
                    "market_margin": -line})
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


# ------------------------------------ ESPN closes: nba, nhl, mlb (free) ----

#: Seasons with ESPN closing records (model/ingest/espn_closes.py). Each is
#: graded only where the league's model can price it walk-forward.
ESPN_SEASONS = {"nba": (2024, 2026), "nhl": (2024, 2026), "mlb": (2023, 2026)}


def espn_closes(league: str) -> pd.DataFrame:
    lo, hi = ESPN_SEASONS[league]
    parts = []
    for y in range(lo, hi + 1):
        f = ROOT / "data" / "raw" / league / f"espn_closes_{y}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f).assign(season=y))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    return d[~d.neutral_site.astype(bool)]


def _espn_meta(league: str, closes: pd.DataFrame, rows: pd.DataFrame, market: str,
               model: str, contamination: str) -> dict:
    return {"market": f"{market} (closing, ESPN public odds records), home side",
            "books": {k: int(v) for k, v in closes.provider.value_counts().items()},
            "model": model,
            "joined": f"{len(rows)} graded of {len(closes)} ESPN closes",
            "seasons": [int(rows.season.min()), int(rows.season.max())],
            "contamination": contamination}


def nba() -> tuple[pd.DataFrame, dict] | None:
    """Spread. ESPN event ids ARE the NBA schedule's ids, so games join by id.
    Integer lines are left out: the live system refuses them (ADR 0022), so a
    grade that priced them would grade a market the system never trades."""
    from coverline.leagues.nba import live
    closes = espn_closes("nba").dropna(subset=["home_spread", "home_spread_price",
                                               "away_spread_price"])
    if closes.empty:
        return None
    out = []
    for season, part in closes.groupby("season"):
        src = live.NBALiveSource.load(int(season))
        model = live.build_model(src)
        sched = src.schedule
        margins = {(t, h, a): m for t, h, a, m in zip(src.history.tip, src.history.home,
                                                     src.history.away, src.history.margin)}
        for r in part.itertuples():
            if r.espn_id not in sched.index or float(r.home_spread).is_integer():
                continue
            g = sched.loc[r.espn_id]
            m = margins.get((g.tip, g.home, g.away))
            if not bool(g.completed) or m is None:
                continue
            margin = float(m)
            line = float(r.home_spread)
            if margin + line == 0:
                continue
            asof = (g.tip - pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                dist = model.predict(r.espn_id, asof)
            except Exception:                  # refused live: not graded either
                continue
            p_model, _ = cover_probability(dist, line)
            out.append({"season": int(season), "espn_id": r.espn_id, "asof": asof,
                        "provider": r.provider, "p_model": p_model,
                        "model_margin": float(dist.margin_mean()), "market_margin": -line,
                        "p_market": _devig_home(r.home_spread_price, r.away_spread_price),
                        "y": int(margin + line > 0), "line": line})
    rows = pd.DataFrame(out)
    return rows, _espn_meta(
        "nba", closes, rows, "spread",
        "NBAModel on NBALiveSource -- the live path, replayed as of one minute before tip",
        "CLEAN OF TUNING: hyperparameters chosen on 2021-2022. 2024-2025 were "
        "graded once before for a different question (static vs walk-forward "
        "accuracy); this asks a new one, against the market.")


def _moneyline_rows(closes: pd.DataFrame, price) -> pd.DataFrame:
    out = []
    for r in closes.itertuples():
        got = price(r)
        if got is None:
            continue
        p_model, y = got
        out.append({"season": int(r.season), "espn_id": r.espn_id, "provider": r.provider,
                    "p_model": p_model, "p_market": _devig_home(r.home_ml, r.away_ml),
                    "y": y, "line": 0.0})
    return pd.DataFrame(out)


#: Franchises in the graded seasons that the live team table no longer
#: carries, because they no longer exist under that name.
NHL_GONE = {"Arizona Coyotes": "ARI"}


def nhl() -> tuple[pd.DataFrame, dict] | None:
    """Moneyline, through NHLLiveSource and the rules layer, matched to the
    NHL schedule by Eastern date and team."""
    from coverline.execution.matching import local_date
    from coverline.leagues.nhl import live
    from coverline.leagues.nhl.teams import TABLE
    closes = espn_closes("nhl").dropna(subset=["home_ml", "away_ml"])
    if closes.empty:
        return None
    parts = []
    for season, part in closes.groupby("season"):
        src = live.NHLLiveSource.load(int(season))
        model = live.build_model(src)
        sched = src.schedule
        key = {(local_date(t.strftime("%Y-%m-%dT%H:%M:%SZ")), h, a): gid
               for gid, t, h, a in zip(sched.index, sched.start, sched.home_team_abbr,
                                       sched.away_team_abbr)}
        finals = pd.read_parquet(live.RAW / f"nhl_{int(season)}.parquet")
        score = {str(g): (h, a) for g, h, a in zip(finals.game_id, finals.home_score,
                                                    finals.away_score)}

        def price(r):
            try:
                k = (local_date(r.date if r.date.endswith("Z") else r.date + "Z"),
                     NHL_GONE.get(r.home) or TABLE.to_code(r.home),
                     NHL_GONE.get(r.away) or TABLE.to_code(r.away))
            except Exception:
                return None
            gid = key.get(k)
            if gid is None or gid not in score:
                return None
            h, a = score[gid]
            asof = (sched.loc[gid].start - pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                p, _ = cover_probability(model.predict(gid, asof), 0.0)
            except Exception:
                return None
            return p, int(h > a)
        parts.append(_moneyline_rows(part, price))
    rows = pd.concat(parts, ignore_index=True)
    return rows, _espn_meta(
        "nhl", closes, rows, "moneyline",
        "NHLModel on NHLLiveSource with the goalie-pull and overtime layers -- the live path",
        "CLEAN OF TUNING: k fixed a priori, zero hyperparameter trials. The rules "
        "layer's own constants were fitted before these seasons were graded; "
        "see data/nhl_rules.json.")


def mlb() -> tuple[pd.DataFrame, dict] | None:
    """Moneyline, on the walk-forward with ACTUAL starters (the live model is
    quoted on probable starters; a scratch is the one difference)."""
    from coverline.execution.matching import local_date
    from coverline.leagues.mlb.model import GameFeatures, MLBModel, load_rules
    from coverline.leagues.mlb.teams import TABLE
    from deploy.mlb_daily_update import load_mlb_caches
    from model.mlb_model import run_walk_forward
    closes = espn_closes("mlb").dropna(subset=["home_ml", "away_ml"])
    if closes.empty:
        return None
    sched, pit = load_mlb_caches(str(ROOT / "model"))
    walk = run_walk_forward(sched, pit)
    walk["key"] = list(zip(walk.date.astype(str), walk.home_team, walk.away_team))
    counts = walk.key.value_counts()
    walk = walk[walk.key.map(counts) == 1]                    # doubleheaders out
    by_key = {k: r for k, r in zip(walk.key, walk.itertuples())}
    rules = load_rules()
    alias = {"Oakland Athletics": "OAK"}

    class _Src:
        def __init__(self, row):
            self.row = row

        def features(self, gid, asof):
            return GameFeatures(exp_home=float(self.row.exp_home),
                                exp_away=float(self.row.exp_away))

    def price(r):
        try:
            h = alias.get(r.home) or TABLE.to_code(r.home)
            a = alias.get(r.away) or TABLE.to_code(r.away)
        except Exception:
            return None
        w = by_key.get((local_date(r.date if r.date.endswith("Z") else r.date + "Z"), h, a))
        if w is None or w.home_score == w.away_score:
            return None
        p, _ = cover_probability(MLBModel(_Src(w), rules=rules).predict("g", "x"), 0.0)
        return p, int(w.home_score > w.away_score)
    rows = _moneyline_rows(closes, price)
    return rows, _espn_meta(
        "mlb", closes, rows, "moneyline",
        "MLBModel over model.mlb_model.run_walk_forward with actual starters",
        "UPPER BOUND. The rules layer was tuned on 2021-2023 and its run-rate "
        "constants measured on all five cached seasons, so 2023-2025 are not "
        "pristine; 2026 is. An upper bound near zero is a firm zero.")


LEAGUES = {"nfl": nfl, "cfb": cfb, "nba": nba, "nhl": nhl, "mlb": mlb}

#: The ESPN-graded leagues replay whole seasons through the live models --
#: minutes, not seconds -- so their graded rows are kept beside the weights.
#: This script rebuilds them; the reproduction test re-grades the kept rows
#: and re-prices a sample of them through the live path, so the rows cannot
#: drift from the code that made them.
ROWS_DIR = ROOT / "data" / "market_grade"
ESPN_LEAGUES = ("nba", "nhl", "mlb")


def historical_rows(name: str, rebuild: bool = False) -> tuple[pd.DataFrame, dict] | None:
    if name not in LEAGUES:
        return None
    f, mf = ROWS_DIR / f"{name}_rows.parquet", ROWS_DIR / f"{name}_meta.json"
    if name in ESPN_LEAGUES and not rebuild and f.exists() and mf.exists():
        return pd.read_parquet(f), json.loads(mf.read_text())
    got = LEAGUES[name]()
    if got is not None and name in ESPN_LEAGUES and len(got[0]):
        ROWS_DIR.mkdir(parents=True, exist_ok=True)
        got[0].to_parquet(f, index=False)
        mf.write_text(json.dumps(got[1], indent=1) + "\n")
    return got

# ---------------------------------------------------- the paper-trade ledger --

LEDGER_DIR = ROOT / "data" / "ledger"

#: Settled games before a ledger grade replaces a league's historical one, or
#: stands alone for a league with none. The lower bound already penalises a
#: small sample; this keeps a first fortnight of noise off the report.
LEDGER_MIN_GAMES = 150

#: One market per game, in this order. Two markets on one game are two
#: correlated readings of the same result, and counting both would shrink the
#: SE and start staking early.
MARKET_PRIORITY = ("spreads", "h2h", "totals")


def ledger_rows(ledger_dir: Path, league: str) -> pd.DataFrame:
    """One graded row per settled game from the live system's own records.

    Clean in a way the historical grades are not: every probability was
    written down before its game by the model as it actually ran, against a
    price that was actually available. p_market is the median devigged price
    across books at the signal's time -- the price the model saw -- and the
    home (or over) side stands for the game.
    """
    from coverline.execution.ledger import BetLedger
    led = BetLedger(ledger_dir)
    outcome = {o.signal_id: o.result for o in led.outcomes()}
    rows = []
    for s in led.signals():
        res = outcome.get(s.signal_id)
        if s.league != league or res not in ("win", "loss") or s.side not in ("home", "over"):
            continue
        rows.append({"event_id": s.event_id, "market": s.market, "at": s.at,
                     "p_model": s.p_model, "p_market": s.p_market,
                     "y": int(res == "win"), "season": str(s.at)[:4]})
    if not rows:
        return pd.DataFrame(columns=["season", "p_model", "p_market", "y"])
    d = pd.DataFrame(rows)
    d["rank"] = d.market.map({m: i for i, m in enumerate(MARKET_PRIORITY)}).fillna(99)
    d = d[d["rank"] == d.groupby("event_id")["rank"].transform("min")]
    d = d[d["at"] == d.groupby("event_id")["at"].transform("max")]
    g = d.groupby("event_id").agg(p_model=("p_model", "first"),
                                   p_market=("p_market", "median"),
                                   y=("y", "first"), season=("season", "first"))
    return g.reset_index(drop=True)


def choose(name: str, historical: dict | None, ledger: pd.DataFrame) -> dict | None:
    """The grade that sizes bets: the ledger once it is big enough, since it
    is the only grade that is clean by construction; otherwise the historical
    one; otherwise none."""
    led = None
    if len(ledger) >= LEDGER_MIN_GAMES:
        led = {**grade(ledger), "source": "ledger",
               "market": "the live system's own paper trades, one per game",
               "seasons": [ledger.season.min(), ledger.season.max()],
               "contamination": "CLEAN: every probability recorded before its game"}
    if led is not None:
        if historical is not None:
            led["historical_grade"] = {k: historical[k] for k in ("n", "w_hat", "se")}
        return led
    if historical is not None:
        return {**historical, "source": "historical",
                "ledger_games_settled": int(len(ledger))}
    return None


def main(ledger_dir: Path = LEDGER_DIR) -> int:
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
    for name in ("nfl", "cfb", "mlb", "nhl", "nba"):
        historical = None
        if name in LEAGUES:
            got = historical_rows(name, rebuild=True)
            if got is not None and len(got[0]) >= 100:
                rows, meta = got
                historical = {**meta, **grade(rows)}
        g = choose(name, historical, ledger_rows(ledger_dir, name))
        if g is None:
            print(f"{name}: no grade (no historical lines, ledger below "
                  f"{LEDGER_MIN_GAMES} settled games) -> weight 0")
            continue
        out["leagues"][name] = g
        print(f"{name} [{g['source']}]: n={g['n']} w_hat={g['w_hat']:+.3f} "
              f"se={g['se']:.3f} -> staking {g['staking_weight']:.3f}  "
              f"gain t={g['gain_t']:+.2f}  side hit {g['side_agreement_hit_rate']:.4f}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
