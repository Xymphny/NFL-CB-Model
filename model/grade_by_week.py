#!/usr/bin/env python3
"""Grades by point in the season, every league (accuracy plan 2b).

    python3 model/grade_by_week.py          # network for NFL (NGS, schedules)

Writes data/grade_by_week.json; scripts/export_record.py carries it onto the
Record page. Changes no price.

WHY
One season-long grade hides that the same model is not equally good all
season: ratings start thin, and in some leagues start from nothing. The
full-ensemble NFL walk-forward ran 52.3% ATS in weeks 1-4 and 55.3% in 9-12,
and the only validated CFB signal is weeks 5+. So each league's grade is
split by segment: w_hat against the close (ADR 0024's estimator, the number
that stakes) and how the model's preferred side did at the closing price.

THE ROWS ARE EACH LEAGUE'S OWN MARKET-GRADE ROWS
  nfl  the system the live board runs since plan 1a: the full ensemble on
       every game with NGS and Elo, the week's slate de-bias, closing spreads
       and prices from nflverse. Built by model/nfl_refit_2016_2022.dataset()
       from the walk-forward caches, 2016-2025 weeks 4-17, current vectors.
  cfb  model/grade_market_weight.cfb(): DVOA-only vector over the walk-forward
       cache, 2021-2023 weeks 4-13, p_market ASSUMED 0.5 (no prices).
  nba/nhl/mlb  data/market_grade/*_rows.parquet, joined to the ESPN closes
       for each game's date; "week" is weeks since that season's first
       graded game.

WHAT A SEGMENT CANNOT SAY
Neither football cache holds weeks 1-3, so the NFL and CFB "1-4" segments are
week 4 alone, and say so in `weeks_covered`. The NFL rows are an upper bound
for 2016-2021 (the full ensemble's fit seasons), the CFB rows for every
season. A segment below 30 games gets no w_hat.

PREFERRED SIDE
The side whose model probability beats the devigged market's. For spread
leagues (NFL, CFB, NBA) that is ATS, against 52.4% at -110. For moneyline
leagues (NHL, MLB) it is a win rate on sides of every price, with no single
break-even, and is labelled that way.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core.market_weight import estimate  # noqa: E402

OUT = ROOT / "data" / "grade_by_week.json"
MIN_W_HAT = 30

#: (label, first week, last week) per league.
SEGMENTS = {
    "nfl": [("Weeks 1-4", 1, 4), ("Weeks 5-8", 5, 8), ("Weeks 9-12", 9, 12), ("Weeks 13+", 13, 99)],
    "cfb": [("Weeks 1-4", 1, 4), ("Weeks 5-8", 5, 8), ("Weeks 9-12", 9, 12), ("Weeks 13+", 13, 99)],
    "nba": [("Weeks 1-3", 1, 3), ("Weeks 4-8", 4, 8), ("Weeks 9-16", 9, 16), ("Weeks 17+", 17, 99)],
    "nhl": [("Weeks 1-3", 1, 3), ("Weeks 4-8", 4, 8), ("Weeks 9-16", 9, 16), ("Weeks 17+", 17, 99)],
    "mlb": [("Weeks 1-4", 1, 4), ("Weeks 5-12", 5, 12), ("Weeks 13-20", 13, 20), ("Weeks 21+", 21, 99)],
}
SIDE_LABEL = {"nfl": "ATS", "cfb": "ATS", "nba": "ATS",
              "nhl": "moneyline win rate", "mlb": "moneyline win rate"}


# ---------------------------------------------------------------- rows ----

def nfl_rows() -> tuple[pd.DataFrame, dict]:
    from coverline.execution.board import slate_debias
    from coverline.leagues.nfl import model as nfl
    from coverline.core import pricing as P
    from model import nfl_refit_2016_2022 as R
    d = R.dataset()
    t = d[d.season.isin(R.TRAIN + R.TEST) & d.spread_line.notna()
          & d.home_spread_odds.notna()].copy()
    full, ro = dict(nfl.MARGIN_COEFFICIENTS), dict(nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY)
    t["model_margin"] = [R.margin(r, full, ro) for r in t.itertuples()]
    off = {k: slate_debias(list(zip(g.model_margin, g.spread_line.astype(float))))[0]
           for k, g in t.groupby(["season", "week"])}
    t["model_margin"] += [off[(s, w)] for s, w in zip(t.season, t.week)]
    t = t[t.actual_margin != t.spread_line]
    w = nfl._load_key_number_weights()
    t["p_model"] = [R.p_model(m, -float(l), w) for m, l in zip(t.model_margin, t.spread_line)]
    t["p_market"] = [float(P.devig([P.american_to_decimal(float(h)),
                                    P.american_to_decimal(float(a))], R.DEVIG)[0])
                     for h, a in zip(t.home_spread_odds, t.away_spread_odds)]
    t["y"] = (t.actual_margin > t.spread_line).astype(int)
    meta = {"model": ("full ensemble where NGS and Elo exist, else rating-only; current "
                      "vectors; weekly slate de-bias -- the live board since plan 1a"),
            "market": "closing spread and prices, nflverse; home side; pushes dropped",
            "contamination": ("UPPER BOUND for 2016-2021, the full ensemble's fit seasons; "
                              "the rating-only vector's window is unrecorded")}
    return t[["season", "week", "p_model", "p_market", "y"]], meta


def cfb_rows() -> tuple[pd.DataFrame, dict]:
    from model.grade_market_weight import cfb
    rows, meta = cfb()
    return rows[["season", "week", "p_model", "p_market", "y"]], {
        k: meta[k] for k in ("model", "market", "market_price", "contamination")}


def espn_rows(league: str) -> tuple[pd.DataFrame, dict]:
    from model.grade_market_weight import espn_closes
    rows = pd.read_parquet(ROOT / "data" / "market_grade" / f"{league}_rows.parquet")
    meta = json.loads((ROOT / "data" / "market_grade" / f"{league}_meta.json").read_text())
    closes = espn_closes(league)[["espn_id", "date", "season"]].drop_duplicates("espn_id")
    r = rows.merge(closes[["espn_id", "date"]], on="espn_id", how="left", validate="many_to_one")
    if r.date.isna().any():
        raise ValueError(f"{league}: {int(r.date.isna().sum())} graded rows have no ESPN date")
    day = pd.to_datetime(r.date, utc=True)
    first = day.groupby(r.season).transform("min")
    r["week"] = ((day - first).dt.days // 7 + 1).astype(int)
    return r[["season", "week", "p_model", "p_market", "y"]], {
        k: meta.get(k) for k in ("model", "market", "contamination")}


# --------------------------------------------------------------- grade ----

def segment(rows: pd.DataFrame, lo: int, hi: int) -> dict:
    s = rows[(rows.week >= lo) & (rows.week <= hi)]
    out = {"n": int(len(s)),
           "weeks_covered": ([int(s.week.min()), int(s.week.max())] if len(s) else None)}
    if len(s) >= MIN_W_HAT:
        mw = estimate(s.p_model, s.p_market, s.y)
        out.update(w_hat=round(float(mw.w_hat), 4), se=round(float(mw.se), 4))
    else:
        out.update(w_hat=None, se=None)
    if len(s):
        home = s.p_model > s.p_market
        won = home == (s.y == 1)
        out["preferred_side"] = round(float(won.mean()), 4)
    return out


def grade(league: str, rows: pd.DataFrame, meta: dict) -> dict:
    whole = segment(rows, 0, 999)
    return {"segments": [{"label": lab, **segment(rows, lo, hi)}
                         for lab, lo, hi in SEGMENTS[league]],
            "all": whole,
            "seasons": [int(rows.season.min()), int(rows.season.max())],
            "preferred_side_label": SIDE_LABEL[league],
            "week_means": ("NFL/CFB week number" if league in ("nfl", "cfb")
                           else "weeks since the season's first graded game"),
            **meta}


def main() -> int:
    out = {"_provenance": {
        "script": "model/grade_by_week.py",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "estimator": "src/coverline/core/market_weight.py (ADR 0024)",
        "min_games_for_w_hat": MIN_W_HAT,
        "preferred_side": "the side whose model probability beats the devigged market's"},
        "leagues": {}}
    loaders = {"nfl": nfl_rows, "cfb": cfb_rows,
               "nba": lambda: espn_rows("nba"), "nhl": lambda: espn_rows("nhl"),
               "mlb": lambda: espn_rows("mlb")}
    for league, load in loaders.items():
        rows, meta = load()
        out["leagues"][league] = grade(league, rows, meta)
        g = out["leagues"][league]
        print(f"{league}: " + "  ".join(
            f"{s['label']} n={s['n']} w={s['w_hat']} side={s.get('preferred_side')}"
            for s in g["segments"]))
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item()
                              if isinstance(o, np.generic) else str(o)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
