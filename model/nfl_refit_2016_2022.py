#!/usr/bin/env python3
"""Refit both NFL margin vectors on 2016-2022, grade them on 2023-2025.

    python3 model/nfl_refit_2016_2022.py        # network: NGS, schedules

Writes data/nfl_refit_2016_2022.json. Changes no live coefficient: whether
the refit passes the rule below is recorded in ADR 0031, and putting it on
the board is a separate, approved change (not made as of 2026-09-27).

WHY (accuracy plan 2a)
The NFL grade is labelled UPPER BOUND because no contiguous season range
reproduces the rating-only coefficients, so no graded season is provably
unseen. The full ensemble's docstring says 2016-2021, and 2022 was graded
as held out against it. Neither vector records its window in an artifact.
This one does, and holds out the three seasons neither refit sees.

DATA, ALL WALK-FORWARD (features for week W use weeks < W)
  rating_diff  model/expanded_walk_forward_cache.csv (2014-2023) and, built
               by the same process_season for this refit,
               model/nfl_walk_forward_2024_2025_cache.csv. Weeks 4-17.
  NGS          model/layer2_ngs through the core's alias table (Rams as LA:
               the live board normalises, so the refit does too).
  Elo          compute_elo_walk_forward from 2006, the PRE-game difference
               WITHOUT Elo's +65 home advantage -- the definition the live
               board serves (odds_watch_job and nfl/ensemble.py difference
               the final ratings). The shipped ensemble was fit on the
               per-game column, which includes the +65: a constant 65 x
               0.0348 = 2.26 points that training put in elo_diff and
               serving puts nowhere. The slate de-bias absorbs constants,
               which is why it never showed; a refit should not repeat it.
  close        data/raw/nfl/nflverse_lines.parquet spread_line and prices.

THE COLLINEARITY THE MODEL FILE FLAGS
home_field is 1 at a home site and 0 at a neutral one, so the neutral games
(23 in 2016-2022) are what separate it from the intercept. Kept in, with
their SEs reported: 23 games identify the split, weakly.

PRE-REGISTERED TEST (written before any held-out number was looked at)
Primary: held-out w_hat against the close (core.market_weight.estimate, the
ADR 0024 grade) on 2023-2025, the SYSTEM as the live board runs it -- each
game on the full ensemble when its NGS and Elo exist, else rating-only; the
week's slate de-bias (execution/board.slate_debias over that week's games);
p_model through NFLModel and execution.recommend.cover_probability at the
closing line; p_market power-devigged from the closing prices; pushes
dropped. Current vectors vs refit vectors, same games.
Secondary, reported not gated: the same without the de-bias; per-vector
subsets; ATS by gap band (|model - market| in points).
Ship rule (the plan's): the refit replaces the current vectors only if its
primary held-out w_hat >= the current vectors'. Both vectors ship or neither.

CONTAMINATION, WHICH CUTS ONE WAY
The current rating-only vector's window is unknown and may include 2023, so
its held-out number may be flattered. That can only make the refit look
worse by comparison, never better.
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

from coverline.core import pricing as P  # noqa: E402
from coverline.core.market_weight import estimate  # noqa: E402
from coverline.execution.board import slate_debias  # noqa: E402
from coverline.execution.recommend import cover_probability  # noqa: E402
from coverline.leagues.nfl import model as nfl  # noqa: E402

OUT = ROOT / "data" / "nfl_refit_2016_2022.json"
CACHE = ROOT / "model" / "expanded_walk_forward_cache.csv"
CACHE_2024_2025 = ROOT / "model" / "nfl_walk_forward_2024_2025_cache.csv"
LINES = ROOT / "data" / "raw" / "nfl" / "nflverse_lines.parquet"
TRAIN = list(range(2016, 2023))
TEST = [2023, 2024, 2025]
ELO_FROM = 2006
FULL_TERMS = ["rating_diff", "home_field", "rest_diff", "cpoe_diff", "separation_diff",
              "yac_oe_diff", "ryoe_diff", "elo_diff", "intercept"]
RATING_TERMS = ["rating_diff", "home_field", "rest_diff", "intercept"]
NGS_TERMS = ["cpoe_diff", "separation_diff", "yac_oe_diff", "ryoe_diff"]
GAP_BANDS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
DEVIG = "power"


# ------------------------------------------------------------- dataset ----

def ratings() -> pd.DataFrame:
    if not CACHE_2024_2025.exists():
        from model.build_expanded_walk_forward_cache import process_season
        pd.concat([process_season(s) for s in (2024, 2025)]).to_csv(CACHE_2024_2025, index=False)
    r = pd.concat([pd.read_csv(CACHE), pd.read_csv(CACHE_2024_2025)], ignore_index=True)
    return r[r.season.isin(TRAIN + TEST)]


def elo() -> pd.DataFrame:
    from ingest.nfl_schedules import load_schedules
    from model.elo_rating import compute_elo_walk_forward
    per_game, _ = compute_elo_walk_forward(load_schedules(list(range(ELO_FROM, max(TEST) + 1))))
    per_game["elo_diff"] = per_game.home_elo_pre - per_game.away_elo_pre   # served definition
    return per_game[["season", "week", "home_team", "away_team", "elo_diff"]]


def ngs(frame: pd.DataFrame) -> pd.DataFrame:
    from coverline.leagues.nfl.ngs import normalise_index
    from model.layer2_ngs import compute_team_ngs_features, load_ngs_data
    rows = []
    for season in sorted(frame.season.unique()):
        pre = {k: load_ngs_data(season, k) for k in ("passing", "receiving", "rushing")}
        wk = frame[frame.season == season]
        for week in sorted(wk.week.unique()):
            try:
                g = normalise_index(compute_team_ngs_features(
                    int(season), through_week=int(week), preloaded_data=pre))
            except ValueError:
                continue
            for r in wk[wk.week == week].itertuples():
                if r.home_team in g.index and r.away_team in g.index:
                    h, a = g.loc[r.home_team], g.loc[r.away_team]
                    rows.append({"season": season, "week": week, "home_team": r.home_team,
                                 "away_team": r.away_team,
                                 "cpoe_diff": h.team_cpoe - a.team_cpoe,
                                 "separation_diff": h.team_avg_separation - a.team_avg_separation,
                                 "yac_oe_diff": h.team_yac_over_expected - a.team_yac_over_expected,
                                 "ryoe_diff": h.team_ryoe - a.team_ryoe})
    return pd.DataFrame(rows)


def dataset() -> pd.DataFrame:
    r = ratings()
    d = r.merge(elo(), on=["season", "week", "home_team", "away_team"], how="left")
    d = d.merge(ngs(d), on=["season", "week", "home_team", "away_team"], how="left")
    lines = pd.read_parquet(LINES)
    lines = lines[lines.game_type == "REG"][["season", "week", "home_team", "away_team",
                                             "spread_line", "home_spread_odds",
                                             "away_spread_odds", "location"]]
    d = d.merge(lines, on=["season", "week", "home_team", "away_team"], how="left",
                validate="one_to_one")
    d["full"] = d[NGS_TERMS + ["elo_diff"]].notna().all(axis=1)
    d["intercept"] = 1.0
    return d


# ----------------------------------------------------------------- fit ----

def ols(train: pd.DataFrame, terms: list[str]) -> dict:
    X = train[terms].to_numpy(float)
    y = train["actual_margin"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - len(terms))
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"coefficients": {t: float(c) for t, c in zip(terms, coef)},
            "se": {t: float(np.sqrt(cov[i, i])) for i, t in enumerate(terms)},
            "n": int(len(y)), "neutral_games": int((train.home_field == 0).sum()),
            "residual_sd": float(np.sqrt(s2)),
            "condition_number": float(np.linalg.cond(X))}


def margin(row, full: dict, rating_only: dict) -> float:
    c = full if row.full else rating_only
    m = c["intercept"] + c["rating_diff"] * row.rating_diff + c["rest_diff"] * row.rest_diff
    m += c["home_field"] * row.home_field
    if row.full:
        m += sum(c[t] * getattr(row, t) for t in NGS_TERMS + ["elo_diff"])
    return float(m)


# --------------------------------------------------------------- grade ----

def p_model(mu: float, line: float, weights) -> float:
    """Through the production distribution: NFLModel's sd and key numbers."""
    from coverline.core.distributions import NormalMarginDistribution
    dist = NormalMarginDistribution(
        mu_margin=mu, sd_margin=nfl.MARGIN_SD, mu_total=45.0,
        sd_total=nfl.TOTAL_SD_UNVALIDATED, total_validated=False, discrete=True,
        key_number_weights=weights)
    return float(cover_probability(dist, line)[0])


def grade(test: pd.DataFrame, full: dict, rating_only: dict, debias: bool,
          all_rating_only: bool = False) -> dict:
    t = test.copy()
    if all_rating_only:
        t["full"] = False
    t["model_margin"] = [margin(r, full, rating_only) for r in t.itertuples()]
    t["market_margin"] = t.spread_line.astype(float)
    if debias:
        off = {k: slate_debias(list(zip(g.model_margin, g.market_margin)))[0]
               for k, g in t.groupby(["season", "week"])}
        t["model_margin"] += [off[(s, w)] for s, w in zip(t.season, t.week)]
    t = t[t.actual_margin != t.spread_line]                     # pushes
    weights = nfl._load_key_number_weights()
    t["p_model"] = [p_model(m, -l, weights) for m, l in zip(t.model_margin, t.spread_line)]
    t["p_market"] = [float(P.devig([P.american_to_decimal(float(h)),
                                    P.american_to_decimal(float(a))], DEVIG)[0])
                     for h, a in zip(t.home_spread_odds, t.away_spread_odds)]
    t["y"] = (t.actual_margin > t.spread_line).astype(int)
    out = {}
    for name, part in (("system", t), ("full_ensemble_games", t[t.full]),
                       ("rating_only_games", t[~t.full])):
        if len(part) < 30:
            out[name] = {"n": int(len(part))}
            continue
        mw = estimate(part.p_model, part.p_market, part.y)
        out[name] = {"n": int(len(part)), "w_hat": round(mw.w_hat, 6), "se": round(mw.se, 6)}
    gap = t.model_margin - t.market_margin
    home = gap > 0
    win = (home == (t.y == 1))
    out["ats_by_gap_band"] = []
    for lo, hi in GAP_BANDS:
        m = (gap.abs() >= lo) & (gap.abs() < hi)
        out["ats_by_gap_band"].append({
            "band": f"{lo}-{hi if hi < 99 else '+'}", "n": int(m.sum()),
            "ats": round(float(win[m].mean()), 4) if m.sum() else None,
            "picks_underdog": round(float((home[m] != (t.market_margin[m] > 0)).mean()), 4)
            if m.sum() else None})
    out["ats_all"] = {"n": int(len(t)), "ats": round(float(win.mean()), 4)}
    out["slope_on_market"] = round(float(np.polyfit(t.market_margin, t.model_margin, 1)[0]), 4)
    return out


def main() -> int:
    d = dataset()
    train = d[d.season.isin(TRAIN)]
    test = d[d.season.isin(TEST) & d.spread_line.notna() & d.home_spread_odds.notna()]
    fit_full = ols(train[train.full], FULL_TERMS)
    fit_rating = ols(train, RATING_TERMS)
    current = (dict(nfl.MARGIN_COEFFICIENTS), dict(nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY))
    refit = (fit_full["coefficients"], fit_rating["coefficients"])
    res = {}
    for label, debias in (("with_slate_debias", True), ("raw", False)):
        res[label] = {"current": grade(test, *current, debias),
                      "refit": grade(test, *refit, debias)}
    # ADDED AFTER THE FIRST RUN, AND STRICTER, NOT LOOSER. Weeks 4-17 always
    # have NGS, so the pre-registered "system" grade priced every held-out
    # game on the full ensemble and never tested the rating-only vector. It
    # is graded here on every held-out game priced rating-only, and each
    # vector ships only on its own held-out w_hat.
    res["rating_only_vector"] = {
        label: {"current": grade(test, *current, debias, all_rating_only=True),
                "refit": grade(test, *refit, debias, all_rating_only=True)}
        for label, debias in (("with_slate_debias", True), ("raw", False))}
    prim_cur = res["with_slate_debias"]["current"]["system"]["w_hat"]
    prim_ref = res["with_slate_debias"]["refit"]["system"]["w_hat"]
    ro = res["rating_only_vector"]["with_slate_debias"]
    ro_cur, ro_ref = ro["current"]["system"]["w_hat"], ro["refit"]["system"]["w_hat"]
    art = {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fit_window": {"seasons": [min(TRAIN), max(TRAIN)], "weeks": [4, 17]},
        "held_out": {"seasons": TEST, "weeks": [4, 17], "n_games": int(len(test))},
        "features": ("walk-forward rating_diff, rest_diff, home_field (0 at neutral sites), "
                     "NGS via the core alias table, Elo pre-game difference without the "
                     "+65 home advantage (the served definition)"),
        "refit": {"full_ensemble": fit_full, "rating_only": fit_rating},
        "current": {"full_ensemble": current[0], "rating_only": current[1]},
        "home_edge_equal_teams": {
            "current_full": current[0]["home_field"] + current[0]["intercept"],
            "refit_full": refit[0]["home_field"] + refit[0]["intercept"],
            "current_rating_only": current[1]["home_field"] + current[1]["intercept"],
            "refit_rating_only": refit[1]["home_field"] + refit[1]["intercept"]},
        "held_out_grade": res,
        "primary": {"metric": "held-out w_hat, system, with slate de-bias",
                    "current": prim_cur, "refit": prim_ref,
                    "ship_rule": "refit ships only if refit >= current",
                    "passes_ship_rule": bool(prim_ref >= prim_cur),
                    "note": ("every held-out game had NGS and Elo, so this grades the "
                             "full-ensemble vector only")},
        "rating_only_vector": {"metric": ("held-out w_hat, every game priced rating-only, "
                                          "with slate de-bias"),
                               "current": ro_cur, "refit": ro_ref,
                               "passes_ship_rule": bool(ro_ref >= ro_cur)},
        "adr": "docs/decisions/0031-nfl-refit-with-a-recorded-window.md",
        "script": "model/nfl_refit_2016_2022.py",
    }
    OUT.write_text(json.dumps(art, indent=1) + "\n")
    print(json.dumps({"primary": art["primary"], "rating_only_vector": art["rating_only_vector"],
                      "home_edge": art["home_edge_equal_teams"],
                      "held_out": art["held_out"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
