#!/usr/bin/env python3
"""Provisional tier thresholds per league: how far is "far" from the market.

WHAT A TIER IS, AND IS NOT
A tier says how strongly the model disagrees with the market ON THIS GAME,
relative to how much it disagrees on that league's games in general. It is
conviction, not edge. Whether conviction is worth money is the market grade's
question (data/market_weights.json), and today its answer is no for every
league -- so a Play is the model's strongest disagreement and nothing more.

That is not a caveat added for form. The derivation below measured it: in
both football backtests the side the model prefers wins between 49% and 53%
in every band, and no band is distinguishable from break-even -- the top band
is nominally above it (NFL 52.9%, CFB 52.6%) by under a fifth of a standard
error. Bigger disagreement has not meant a better result here.

HOW THE BANDS ARE SET
Quantiles of |p_model - p_market| over the league's own history, in
probability points:
    no_edge    below the 40th percentile   the model broadly agrees
    coin_flip  40th to 70th                it leans, slightly
    lean       70th to 90th                a real disagreement
    play       90th and above              its strongest tenth
Relative, because the leagues disagree with their markets by very different
amounts (NFL's median is ~10 points, CFB's ~13) and a fixed cutoff would call
half of one league a Play and almost none of another.

WHERE EACH LEAGUE'S HISTORY COMES FROM
  nfl  walk-forward cache x nflverse closing spreads     (1,884 games)
  cfb  walk-forward cache x CFBD closing spreads         (1,704 games)
  mlb  ESPN closing moneylines x the walk-forward (grade_market_weight.mlb);
       data/mlb_divergence snapshots (~87 games, rough) only if those are absent
  nhl  ESPN closing moneylines x the live model (grade_market_weight.nhl);
       BORROWED from mlb only if those rows are absent
  nba  ESPN closing spreads x the live model (grade_market_weight.nba);
       BORROWED from nfl only if those rows are absent
Every league's bands are replaced by the same quantiles over its own settled
paper trades once it has 150 of them (source "ledger").

Writes data/tier_thresholds.json.
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model import grade_market_weight as G  # noqa: E402

OUT = ROOT / "data" / "tier_thresholds.json"
QUANTILES = {"coin_flip": 0.40, "lean": 0.70, "play": 0.90}
BORROW = {"nhl": "mlb", "nba": "nfl"}
#: Enough to place a 90th percentile at all. MLB clears it with ~87 games from
#: one week of odds-watch snapshots, which makes its bands rough -- one reason
#: every band here is PROVISIONAL and is replaced at 150 settled paper trades.
MIN_ROWS = 50


def bands(edges: pd.Series) -> dict:
    a = edges.abs()
    out = {k: round(float(a.quantile(q)), 4) for k, q in QUANTILES.items()}
    return out


def by_band(rows: pd.DataFrame, cut: dict) -> dict:
    """Hit rate of the side the model preferred, per band -- recorded so the
    claim that bands are not edge stays checkable."""
    e = rows.p_model - rows.p_market
    a = e.abs()
    hit = np.where(e > 0, rows.y == 1, rows.y == 0)
    # What the MARKET said that side's chance was. On a spread it is about
    # 0.5, so the hit rate reads against break-even; on a moneyline the
    # model's strongest disagreements are mostly underdogs the market prices
    # well under 0.5, and a 37% hit rate is only bad if the market said more.
    mkt = np.where(e > 0, rows.p_market, 1 - rows.p_market)
    edges_ = [0.0, cut["coin_flip"], cut["lean"], cut["play"], 1.0]
    out = {}
    for name, lo, hi in zip(("no_edge", "coin_flip", "lean", "play"), edges_[:-1], edges_[1:]):
        m = (a >= lo) & (a < hi) if name != "play" else (a >= lo)
        out[name] = {"n": int(m.sum()),
                     "preferred_side_hit_rate": round(float(hit[m].mean()), 4) if m.any() else None,
                     "market_said": round(float(mkt[m].mean()), 4) if m.any() else None}
    return out


def mlb_rows(through: str | None = None) -> pd.DataFrame:
    """Moneyline disagreement from the odds-watch snapshots, priced by the
    walk-forward for the same game with the starters who actually started.

    `through` (YYYY-MM-DD) keeps only games on or before that date. The
    inputs grow every day -- the odds watch adds snapshots and the MLB cron
    adds results -- so without a recorded cut-off the bands could not be
    reproduced a day after they were derived, and the reproduction test went
    red on every cron update. The walk-forward is point-in-time, so a game's
    price does not change when later games are added; the cut-off alone makes
    the rows reproducible."""
    from coverline.execution.matching import local_date
    from coverline.execution.recommend import cover_probability
    from coverline.leagues.mlb.model import MLBModel, load_rules
    from coverline.leagues.mlb.teams import TABLE
    from deploy.mlb_daily_update import load_mlb_caches
    from model.mlb_model import run_walk_forward

    sched, pit = load_mlb_caches(str(ROOT / "model"))
    walk = run_walk_forward(sched, pit)
    walk["key"] = list(zip(walk.date.astype(str), walk.home_team, walk.away_team))
    counts = walk.key.value_counts()
    walk = walk[walk.key.map(counts) == 1]                     # doubleheaders out
    by_key = {k: r for k, r in zip(walk.key, walk.itertuples())}

    latest: dict = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "mlb_divergence" / "*.json"))):
        for r in json.loads(Path(f).read_text()).get("divergences", []):
            if r.get("line_status") != "open" or r.get("market_home_prob") is None:
                continue
            try:
                k = (local_date(r["kickoff"]), TABLE.to_code(r["home_name"]),
                     TABLE.to_code(r["away_name"]))
            except Exception:
                continue
            latest[k] = r                      # files are in time order: last wins

    class _Src:
        def __init__(self, row): self.row = row
        def features(self, gid, asof):
            from coverline.leagues.mlb.model import GameFeatures
            return GameFeatures(exp_home=float(self.row.exp_home),
                                exp_away=float(self.row.exp_away))

    rules = load_rules()
    out = []
    for k, r in latest.items():
        w = by_key.get(k)
        if w is None:
            continue
        p_home, _ = cover_probability(MLBModel(_Src(w), rules=rules).predict("g", "x"), 0.0)
        out.append({"p_model": p_home, "p_market": float(r["market_home_prob"]),
                    "y": int(w.home_score > w.away_score), "season": k[0][:4], "date": k[0]})
    df = pd.DataFrame(out)
    if through is not None and len(df):
        df = df[df.date <= through].reset_index(drop=True)
    return df


#: The board's "check before you trust it" flag (dashboard v2 item 8): a
#: game whose model and market margins sit further apart than this share of
#: the league's backtest did. The brief's rule, applied as written; whether
#: 99% is the right cut is recorded as an open question for the owner.
OUTLIER_QUANTILE = 0.99


def outlier_points(rows: pd.DataFrame) -> float | None:
    """The OUTLIER_QUANTILE of |model margin - market margin| in points, for
    leagues whose history carries both margins (spread leagues); else None."""
    if not {"model_margin", "market_margin"} <= set(rows.columns):
        return None
    gap = (rows.model_margin - rows.market_margin).abs().dropna()
    return round(float(gap.quantile(OUTLIER_QUANTILE)), 2) if len(gap) else None


def history(name: str, mlb_through: str | None = None) -> tuple[pd.DataFrame | None, str]:
    got = G.historical_rows(name)             # kept rows for the ESPN leagues
    if got is not None and len(got[0]):
        rows, meta = got
        return rows, meta["market"]
    if name == "mlb":
        return mlb_rows(mlb_through), ("moneyline, data/mlb_divergence snapshots (latest open "
                            "line per game) x the walk-forward with actual starters")
    return None, ""


def derive(ledger_dir: Path = G.LEDGER_DIR, mlb_through: str | None = None) -> dict:
    """`mlb_through=None` uses every MLB game now available and records the
    last date used, so the result can be rebuilt exactly later."""
    out = {"_provenance": {
        "script": "model/derive_tier_thresholds.py",
        "generated": date.today().isoformat(),
        "unit": "probability points (0-1) of |p_model - p_market|",
        "quantiles": QUANTILES,
        "meaning": ("conviction relative to the league's own history, NOT edge; "
                    "stake comes from data/market_weights.json"),
        "status": "PROVISIONAL until a league has 150 settled paper trades",
    }, "leagues": {}}
    own = {}
    for name in ("nfl", "cfb", "mlb", "nhl", "nba"):
        led = G.ledger_rows(ledger_dir, name)
        if len(led) >= G.LEDGER_MIN_GAMES:
            cut = bands(led.p_model - led.p_market)
            own[name] = {"source": "ledger", "provisional": False, "n": int(len(led)),
                         "basis": "the league's own settled paper trades",
                         **cut, "by_band": by_band(led, cut)}
            continue
        rows, basis = history(name, mlb_through)
        if rows is not None and len(rows) >= MIN_ROWS:
            cut = bands(rows.p_model - rows.p_market)
            own[name] = {"source": "backtest", "provisional": True, "n": int(len(rows)),
                         "basis": basis, **cut, "by_band": by_band(rows, cut)}
            op = outlier_points(rows)
            if op is not None:
                own[name]["outlier_points"] = op
            if "date" in rows.columns:
                own[name]["through"] = str(rows.date.max())
    for name in ("nfl", "cfb", "mlb", "nhl", "nba"):
        if name in own:
            out["leagues"][name] = own[name]
        elif BORROW.get(name) in own:
            src = own[BORROW[name]]
            out["leagues"][name] = {
                "source": f"borrowed:{BORROW[name]}", "provisional": True, "n": 0,
                "basis": (f"no market history for {name.upper()}; bands borrowed "
                          f"from {BORROW[name].upper()} until its own ledger has 150 games"),
                **{k: src[k] for k in QUANTILES}}
    return out


def main(ledger_dir: Path = G.LEDGER_DIR, argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Derive the tier bands.")
    ap.add_argument("--mlb-through", help="YYYY-MM-DD; default: every game available")
    a = ap.parse_args(argv)
    art = derive(ledger_dir, a.mlb_through)
    OUT.write_text(json.dumps(art, indent=2) + "\n")
    for name, g in art["leagues"].items():
        print(f"{name}: {g['source']:12} n={g['n']:5}  coin_flip {g['coin_flip']:.3f}  "
              f"lean {g['lean']:.3f}  play {g['play']:.3f}"
              + (f"  hit by band {[b['preferred_side_hit_rate'] for b in g['by_band'].values()]}"
                 if "by_band" in g else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
