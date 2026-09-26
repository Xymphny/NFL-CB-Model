#!/usr/bin/env python3
"""Write the dashboard's board, one file per league, from the core.

    python3 scripts/export_board.py                  # all five leagues
    python3 scripts/export_board.py --league nba

Writes data/site/board_{league}.json. The site renders these and computes
nothing (the dashboard brief): every probability, edge, tier and stake here
comes from the league model and the same pricing, tier and staking code the
operator command uses (execution/board.py).

WHAT A BOARD HOLDS
  slate       the games the league's live source has for today (or this NFL
              week), with the model's own view of each -- P(home wins),
              expected margin -- whether or not a book has quoted it yet
  markets     per game, each of the league's primary markets the feed quotes:
              line, best price and book, p_model, p_market, edge, tier, the
              stake the core would size, and the line at open
  grade       the league's market-grade weight and its evidence (ADR 0024)
  tiers       the bands used and whether they are provisional (ADR 0025)
  freshness   when the odds and each input were last current
  refusals    anything the core would not price, and why, in plain words
  context     what that sport's cards headline: probables, rest, early season
  teams       what the league's model actually rates

A league whose inputs are not ready still gets a file: its refusal IS the
board, so the site never shows yesterday's games as if they were today's.

Run by close-capture-job after each capture and by live-inputs-job daily.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core import tiers as T  # noqa: E402
from coverline.core.market_weight import staking_weight  # noqa: E402
from coverline.execution import board as B  # noqa: E402
from coverline.execution import recommend as Rc  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.matching import local_date, match_by_date  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import board_context  # noqa: E402
import board_detail  # noqa: E402

SITE = ROOT / "data" / "site"
BRONZE = ROOT / "data" / "bronze"
LEAGUES = ("nfl", "cfb", "mlb", "nhl", "nba")

SPORT_NOTES = {
    "nfl": "Weekly slate. Spreads price key numbers (3, 7) with a measured correction.",
    "cfb": ("Weekly slate, Thursday to Saturday. Spreads price key numbers (3, 7) "
            "with a measured correction. Neutral-site games are refused: the model "
            "has no home term to remove."),
    "mlb": ("Daily. Lines are quoted on the listed probable starters; a scratch "
            "changes the game being priced."),
    "nhl": ("Daily. Ratings reset every season, so early-season prices carry "
            "little information by design."),
    "nba": ("Daily. The model is team-level: it has no injury or minutes layer, "
            "so late scratches and rest decisions are not in its price."),
}

#: NHL games played below which a card is flagged low-information. A display
#: flag, not a model input: the ratings walk from a league-average start, and
#: ten games is roughly where a team's own record begins to dominate it.
NHL_EARLY_GAMES = 10


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _runner():
    spec = importlib.util.spec_from_file_location(
        "recommend_slate", ROOT / "scripts" / "recommend_slate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recommend_slate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _clean(x):
    """JSON-safe: NaN and inf become null rather than invalid JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _names(league: str) -> dict[str, str]:
    """model code -> display name, the first spelling each table lists."""
    if league == "nfl":
        from coverline.leagues.nfl.events import NAME_TO_CODE as table
    else:
        table = getattr(importlib.import_module(f"coverline.leagues.{league}.teams"),
                        f"{league.upper()}_NAMES")
    out: dict[str, str] = {}
    for name, code in table.items():
        out.setdefault(code, name)
    return out


def credits(store: BronzeStore) -> dict | None:
    """The Odds API's remaining monthly credits, as the capture job last saw
    them (scripts/capture.py write_quota)."""
    f = store.root / "odds" / "_quota.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def latest_snapshot(store: BronzeStore, sport: str):
    rows = list(store.snapshots(sport))
    return (store.root / rows[-1]["path"], rows[-1]["captured_at"]) if rows else (None, None)


def opening_quotes(store: BronzeStore, sport: str, event_ids: set[str]) -> dict:
    """event_id -> (quotes, captured_at) from the first snapshot holding it."""
    out: dict = {}
    for row in store.snapshots(sport):
        missing = event_ids - set(out)
        if not missing:
            break
        path = store.root / row["path"]
        if not path.exists():
            continue
        rec = store.read_snapshot(path)
        qs = normalize(rec["payload"], captured_at=rec["_meta"]["captured_at"])
        for e in missing:
            eq = [q for q in qs if q.event_id == e]
            if eq:
                out[e] = (eq, rec["_meta"]["captured_at"])
    return out


def model_view(dist) -> dict:
    view = {"p_home_win": None, "margin_mean": None, "total_mean": None}
    try:
        view["p_home_win"] = round(Rc.cover_probability(dist, 0.0)[0], 5)
    except Exception:
        pass
    for k, fn in (("margin_mean", "margin_mean"), ("total_mean", "total_mean")):
        try:
            if k == "total_mean" and getattr(dist, "total_validated", True) is False:
                continue          # a withheld total is not shown as a number
            view[k] = round(float(getattr(dist, fn)()), 3)
        except Exception:
            pass
    return view


# ------------------------------------------------------------ per league ----

def _load(league: str, R, day: str | None = None):
    """(model, source, games, slate descriptor, start_of, context_of, freshness)."""
    lg = R.LEAGUES[league]
    if league == "cfb":
        # A CFB week, like an NFL one: Thursday to Saturday on one board.
        from coverline.leagues.cfb import live
        today = day or _today_et()
        try:
            src = live.CFBLiveSource.load(live.season_of(today))
        except live.MissingSeasonData as e:
            raise R.InputsNotReady(str(e)) from None
        week, games = src.week_slate(today)
        s = src.schedule
        start = lambda gid: s.loc[gid].start.strftime("%Y-%m-%dT%H:%M:%SZ")
        snap = src.snapshots[-1]
        fresh = {"ratings_version": snap.path.name,
                 "ratings_computed_at": snap.computed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "ratings_through_week": snap.week - 1,
                 "latest_final": src.latest_final()}
        slate = {"kind": "week", "season": src.season, "week": week,
                 "dates": sorted({g.date for g in games})}
        extra = _context_bundle(lambda: board_context.cfb(src, [g.game_id for g in games]))
        return (live.build_model(src), src, games, slate, start,
                lambda gid: {"neutral_site": bool(s.loc[gid].neutral_site),
                             **extra.get(gid, {})}, fresh)
    if league == "nfl":
        from coverline.leagues.nfl.live import load_schedule
        today = day or _today_et()
        season = int(today[:4])
        sched = load_schedule([season])
        up = sched[(sched.game_type == "REG") & (sched.gameday.astype(str) >= today)]
        if up.empty:
            raise R.InputsNotReady(f"no upcoming NFL regular-season games in {season}")
        week = int(up.week.min())
        try:
            model, src = lg.loader(season, week)
        except Exception as exc:
            raise R.InputsNotReady(f"week {week}: {exc}") from None
        wk = sched[(sched.game_type == "REG") & (sched.week == week)]
        from coverline.execution.matching import ModelGame
        games = [ModelGame(f"{season}-W{week:02d}-{h}-{a}", str(d), h, a)
                 for h, a, d in zip(wk.home_team, wk.away_team, wk.gameday)]
        # nflverse gives Eastern wall-clock kickoffs; every board time is UTC.
        starts = {g.game_id: pd.Timestamp(f"{d} {t}", tz="America/New_York")
                  .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                  for g, d, t in zip(games, wk.gameday.astype(str), wk.gametime.astype(str))}
        extra = _context_bundle(lambda: board_context.nfl(
            wk, [g.game_id for g in games], season, week, starts))
        ctx = {g.game_id: {**_nfl_context(r), **extra.get(g.game_id, {})}
               for g, r in zip(games, wk.itertuples())}
        return (model, src, games, {"kind": "week", "season": season, "week": week},
                lambda gid: starts.get(gid), lambda gid: ctx.get(gid, {}),
                _nfl_fresh(src, today))
    day = day or _today_et()
    model, src, games = lg.loader(day)
    if league == "nba":
        s = src.schedule
        start = lambda gid: s.loc[gid].tip.strftime("%Y-%m-%dT%H:%M:%SZ")
        done = src.history
        fresh = {"latest_final": str(done.tip.max()) if len(done) else None}
        extra = _context_bundle(lambda: board_context.nba(src, [g.game_id for g in games]))
        return (model, src, games, {"kind": "date", "date": day}, start,
                lambda gid: {**_nba_context(src, gid), **extra.get(gid, {})}, fresh)
    if league == "nhl":
        s = src.schedule
        start = lambda gid: s.loc[gid].start.strftime("%Y-%m-%dT%H:%M:%SZ")
        fresh = {"latest_final": str(src.games.start.max()) if len(src.games) else None}
        extra = _context_bundle(lambda: board_context.nhl(src, [g.game_id for g in games]))
        return (model, src, games, {"kind": "date", "date": day}, start,
                lambda gid: {**_nhl_context(src, gid), **extra.get(gid, {})}, fresh)
    if league == "mlb":
        start = lambda gid: src._game(gid).get("start_utc")
        played = src.schedule.dropna(subset=["home_score", "away_score"])
        fresh = {"slate_fetched_at": src.payload.get("fetched_at"),
                 "latest_final": str(played.date.max())}
        extra = _context_bundle(lambda: board_context.mlb(src, [g.game_id for g in games]))
        return (model, src, games, {"kind": "date", "date": day}, start,
                lambda gid: {**_mlb_context(src, gid), **extra.get(gid, {})}, fresh)
    raise ValueError(league)


def attach_line_history(board: dict, store: BronzeStore, sport: str, headline: str,
                        dists: dict) -> None:
    """Brief item 10: every matched game's headline line since open, and the
    model's fair line, from one pass over the captures. Soft-fails to
    nothing: history is display, never a reason a board is missing."""
    try:
        now = pd.Timestamp(board["generated_at"])
        key = B.M.vendor_key(headline)
        want = {g["event_id"]: key for g in board["games"] if g.get("event_id")}
        hist = board_detail.history(store, sport, want, now)
        gaps = board_detail.true_gaps(store, sport)
        for g in board["games"]:
            m = g["markets"].get(headline) or {}
            side = m.get("side") or "home"
            if g.get("event_id"):
                g["line_history"] = _clean(board_detail.for_side(
                    hist.get(g["event_id"], []), side, g.get("start"), gaps))
            if g["game_id"] in dists and m.get("line") is not None:
                g["fair_line"] = board_detail.fair_line(dists[g["game_id"]], side)
            elif m.get("status") == "priced":
                # A moneyline has no line to draw: the sparkline plots the
                # market's probability, and the model's is its fair mark.
                g["fair_p"] = m.get("p_model")
    except Exception as exc:
        print(f"[export_board] line history soft-fail: {type(exc).__name__}: {exc}")


#: CFB weeks bannered EARLY SEASON (dashboard v2 item 8).
CFB_EARLY_WEEKS = 4


def cfb_window(start: str | None) -> str | None:
    """The CFB board's kickoff window, Eastern time (brief item 8)."""
    if not start:
        return None
    t = pd.Timestamp(start).tz_convert("America/New_York")
    day, h = t.strftime("%A"), t.hour
    if day == "Friday":
        return "Friday night"
    if day == "Saturday":
        return ("Saturday noon" if h < 15 else "Saturday afternoon" if h < 19
                else "Saturday prime time" if h < 22 else "Saturday late")
    if day == "Sunday" and h < 6:
        return "Saturday late"
    return f"{day}"


def attach_check_flags(board: dict, headline: str) -> None:
    """"Check before you trust it" (brief item 8): the model's and the
    market's home margins further apart than the league's outlier_points --
    the 99th percentile of |gap| in its backtest. Display only: the tier
    still follows the rule."""
    thr = (board.get("tiers") or {}).get("outlier_points")
    if thr is None:
        return
    for g in board["games"]:
        m = g["markets"].get(headline) or {}
        mu = (g.get("model") or {}).get("margin_mean")
        if m.get("status") != "priced" or m.get("line") is None or mu is None:
            continue
        market = -(m["line"] if m["side"] == "home" else -m["line"])
        gap = abs(mu - market)
        if gap > thr:
            g["check_flag"] = {
                "gap_points": round(gap, 1), "threshold": thr,
                "model_margin": round(mu, 1), "market_margin": round(market, 1),
                "text": ("A gap this size is more often a rating that hasn't caught up than "
                         "an edge. It's shown because the tier rule says so, and it's "
                         "paper-traded like everything else.")}


def _context_bundle(make) -> dict:
    """A league's display context (scripts/board_context.py), soft-failing to
    none: context never blocks a board, and never reaches a price."""
    try:
        return make() or {}
    except Exception as exc:
        print(f"[export_board] context soft-fail: {type(exc).__name__}: {exc}")
        return {}


def regime_evidence() -> str:
    """'wins/n' for early flags that backed a first-year external head coach,
    at the Lean threshold -- read from the artifact that measured it, never
    typed. (The board once quoted 0/9, double-counting Play inside Lean.)"""
    art = json.loads((ROOT / "model" / "coach_regime_results.json").read_text())
    g = next(g for g in art["grades"]
             if g["label"] == "model BACKED the regime team" and g["min_edge"] == 2.5)
    return f"{g['wins']}/{g['n_graded']}"


def apply_regime_cap(entry: dict, headline: str, regimes: dict, week: int) -> None:
    """The legacy board's regime rule, carried into the core's export.

    Early-season flags that BACKED a first-year external head coach's team
    against the market went 0 for 6 in 2016-2023 (model/coach_regime_
    experiment.py); flags fading such teams graded at baseline and are left
    alone. So within the cap window a flag backing a regime team is capped at
    Lean and its stake halved -- a reduction, never a removal, and still
    graded. The weeks and the regime map are the legacy job's own, imported.
    """
    from deploy.odds_watch_job import REGIME_CAP_WEEKS, REGIME_CHIP_WEEKS
    if not regimes or week is None:
        return
    home_r, away_r = regimes.get(entry["home"]), regimes.get(entry["away"])
    if week <= REGIME_CHIP_WEEKS and (home_r or away_r):
        entry["context"]["regime"] = {"home": home_r, "away": away_r}
    m = entry["markets"].get(headline)
    if week > REGIME_CAP_WEEKS or not m or m.get("status") != "priced":
        return
    picked = entry["home"] if m["side"] == "home" else entry["away"]
    if picked not in regimes:
        return
    q = regime_evidence()
    if m["tier"] == "play":
        m["tier"] = "lean"
    m["stake_fraction"] = round(m["stake_fraction"] * 0.5, 5)
    m["cap"] = {"rule": "regime", "reason": (
        f"{picked} first-year staff ({regimes[picked]['coach']}): early flags backing "
        f"new-regime teams went {q} in backtests (2016-2023). Capped at Lean, still graded.")}


def apply_early_season_cap(entry: dict) -> None:
    """NHL tiers are held at Coin flip until both teams have NHL_EARLY_GAMES.

    The graded NHL procedure restarts every team at league average each
    season (ADR 0006, model/fit_nhl_walkforward.py). On opening night the
    model therefore "disagrees" with every favourite by exactly the market's
    own opinion of the two teams, and the tiers would light Plays on
    underdogs for a reason that is the reset, not information. The model's
    price is still shown and still paper-traded -- the ledger records
    probabilities, not tiers, so this changes no grade -- but no card claims
    conviction the ratings have not earned yet. Like the regime cap: a
    reduction, never a removal.
    """
    c = entry.get("context") or {}
    if not c.get("low_information"):
        return
    gp = f"{entry['away']} has played {c.get('away_games_played')}, " \
         f"{entry['home']} {c.get('home_games_played')}"
    for m in entry["markets"].values():
        if m.get("status") != "priced" or m.get("tier") not in ("play", "lean"):
            continue
        m["tier"] = "coin_flip"
        m["stake_fraction"] = 0.0
        m["cap"] = {"rule": "early_season", "reason": (
            f"Early season: {gp}. Hockey ratings restart at league average every "
            f"season, so the gap with the market is mostly the reset. Held at Coin "
            f"flip until both teams have played {NHL_EARLY_GAMES}; still paper-traded.")}


def _nfl_fresh(src, today: str) -> dict:
    try:
        ca, wk, path = src.version_for(today)
        return {"ratings_version": path.name, "ratings_computed_at": ca.isoformat(),
                "ratings_through_week": wk}
    except Exception as exc:
        return {"ratings_version": None, "ratings_note": str(exc)}


def _nfl_context(r) -> dict:
    ctx = {}
    for k in ("home_qb_name", "away_qb_name", "roof", "temp", "wind", "div_game"):
        v = getattr(r, k, None)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            ctx[k] = v if not hasattr(v, "item") else v.item()
    return ctx


def _nba_context(src, gid) -> dict:
    s = src.schedule
    g = s.loc[gid]
    out = {"team_level_model": True}
    for side in ("home", "away"):
        team = g[side]
        prior = s[((s.home == team) | (s.away == team)) & (s.tip < g.tip)]
        if len(prior):
            days = (g.tip.normalize() - prior.tip.max().normalize()).days
            out[f"{side}_rest_days"] = int(days)
            out[f"{side}_back_to_back"] = bool(days <= 1)
    return out


def _nhl_context(src, gid) -> dict:
    g = src.schedule.loc[gid]
    played = src.games[src.games.start < g.start]
    out = {}
    for side, team in (("home", g.home_team_abbr), ("away", g.away_team_abbr)):
        n = int(((played.home_team_abbr == team) | (played.away_team_abbr == team)).sum())
        out[f"{side}_games_played"] = n
    out["low_information"] = min(out["home_games_played"], out["away_games_played"]) < NHL_EARLY_GAMES
    return out


def _mlb_context(src, gid) -> dict:
    g = src._game(gid)
    ctx = {"home_probable": g.get("home_sp_name"), "away_probable": g.get("away_sp_name"),
           "park": g.get("park")}
    # A scratch: the probable on the earliest pull for this date is not the
    # one on the latest. Lines are quoted on the listed starter, so a change
    # is the headline, not a footnote.
    from coverline.leagues.mlb.live import SLATE_DIR
    pulls = sorted(SLATE_DIR.glob(f"{src.day}-*.json"))
    if len(pulls) > 1:
        first = {x["game_key"]: x for x in json.loads(pulls[0].read_text())["games"]}
        f = first.get(gid, {})
        for side in ("home", "away"):
            was, now = f.get(f"{side}_sp_name"), g.get(f"{side}_sp_name")
            if was and now and was != now:
                ctx[f"{side}_probable_changed_from"] = was
    return ctx


def next_slate(league: str, src, slate: dict) -> dict | None:
    """The next date with games, for a league with none today."""
    try:
        if league == "nba":
            s = src.schedule[~src.schedule.completed & ~src.schedule.postponed]
            starts = s.tip
        elif league == "nhl":
            s = src.schedule[~src.schedule.game_state.isin(["OFF", "FINAL", "PPD", "CNCL"])]
            starts = s.start
        else:
            return None
        dates = sorted(local_date(t.strftime("%Y-%m-%dT%H:%M:%SZ")) for t in starts)
        later = [d for d in dates if d > slate["date"]]
        if not later:
            return None
        return {"date": later[0], "games": later.count(later[0])}
    except Exception:
        return None


def teams(league: str, src) -> list[dict]:
    """What the league's model actually rates, as of now."""
    try:
        if league == "nba":
            r, last = src.ratings_before("now")
            # Between seasons the model regresses every rating before pricing
            # (live.NBALiveSource.features). Shown as priced, not as stored.
            c = src.hyperparameters["carryover"] if last is not None and last != src.season else 1.0
            return sorted(({"team": t, "rating": round(v * c, 3),
                            "note": "points of margin vs an average team, as priced"}
                           for t, v in r.items()), key=lambda x: -x["rating"])
        if league == "nhl":
            from model import fit_nhl_walkforward as wf
            played = src.games
            if not len(played):
                return []
            st: dict = {}
            wf.walk_forward(played.rename(columns={"nopull_h": "home_score",
                                                   "nopull_a": "away_score"})[
                ["game_date", "home_team_abbr", "away_team_abbr", "home_score", "away_score"]],
                state=st)
            return sorted(({"team": t, "attack": round(st["attack"][t], 4),
                            "defence": round(st["defence"].get(t, 0.0), 4)}
                           for t in st["attack"]), key=lambda x: -(x["attack"] - x["defence"]))
        if league == "mlb":
            from coverline.leagues.mlb.live import state_before
            if src._state is None:
                src._state = state_before(src.schedule, src.pitching, src.day)
            st = src._state
            rows = []
            for g in src.payload["games"]:
                for side in ("home", "away"):
                    sp = g.get(f"{side}_sp")
                    if sp:
                        rows.append({"pitcher": g.get(f"{side}_sp_name"), "team": g[f"{side}_team"],
                                     "ra27": round(st.sp_ra27(sp), 3), "park": g.get("park"),
                                     "park_factor": round(st.park_factor(g.get("park")), 3)})
            return rows
        if league == "cfb":
            # The newest snapshot, which is what prices the coming week. Elo
            # is left off: the live source does not read it (cfb/live.py).
            snap = src.snapshots[-1]
            keep = ("total_rating", "offense_voa", "defense_voa")
            return sorted(({"team": r["team"], **{k: (round(r[k], 4) if isinstance(r.get(k), float)
                                                      else r.get(k)) for k in keep}}
                           for r in json.loads(snap.path.read_text())["ratings"]),
                          key=lambda x: -(x["total_rating"] or 0))
        if league == "nfl":
            # The fields the rating-only vector reads (MARGIN_COEFFICIENTS_V1)
            # and the ones the card explains it with.
            keep = ("total_rating", "offense_voa", "defense_voa", "special_teams_voa",
                    "rating_p05", "rating_p95")
            _, _, path = src.version_for(_today_et())
            ratings = {r["team"]: r for r in json.loads(Path(path).read_text())["ratings"]}
            return sorted(({"team": t, **{k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k))
                                          for k in keep}}
                           for t, r in ratings.items()), key=lambda x: -(x["total_rating"] or 0))
    except Exception:
        return []
    return []


def build(league: str, store: BronzeStore, R, day: str | None = None) -> dict:
    board = {"league": league, "generated_at": _now(), "note": SPORT_NOTES[league],
             "slate": None, "games": [], "refusals": [], "freshness": {}, "teams": []}
    w, grade = staking_weight(league)
    board["grade"] = {"weight": w, **({k: grade.get(k) for k in
                      ("w_hat", "se", "n", "source", "seasons", "contamination")}
                      if grade else {"source": None})}
    try:
        bands = T.thresholds(league)
    except T.NoThresholds:
        bands = None
    board["tiers"] = (None if bands is None else
                      {k: bands[k] for k in ("coin_flip", "lean", "play", "source", "provisional")})
    if bands is not None and bands.get("outlier_points") is not None:
        board["tiers"]["outlier_points"] = bands["outlier_points"]

    lg = R.LEAGUES[league]
    snap, captured = latest_snapshot(store, lg.vendor_sport)
    board["freshness"]["odds_captured_at"] = captured
    board["freshness"]["credits"] = credits(store)

    try:
        model, src, games, slate, start_of, context_of, fresh = _load(league, R, day)
    except Exception as exc:                  # InputsNotReady and anything else
        board["refusals"].append({"scope": "league", "reason": str(exc)})
        return board
    board["slate"] = slate
    board["freshness"].update(fresh)
    board["teams"] = teams(league, src)
    markets = list(model.primary_markets)
    default = R.LEAGUES[league].default_market
    headline = next((m for m in markets if B.M.vendor_key(m) == B.M.vendor_key(default)),
                    markets[0])
    if snap is None and games:
        board["refusals"].append({"scope": "league", "reason":
            "No odds captured yet for this league. The model's view is shown; "
            "market prices appear once the capture job has run."})
    stale = stale_odds(captured, board["generated_at"]) if games else None
    if stale:
        board["refusals"].append({"scope": "league", "reason": stale})
    if not games:
        board["next_slate"] = next_slate(league, src, slate)

    quotes, matched = [], {}
    if snap is not None:
        rec = store.read_snapshot(snap)
        quotes = normalize(rec["payload"], captured_at=captured)
        if league == "nfl":
            from coverline.leagues.nfl.events import match_events
            ms = match_events(quotes, season=slate["season"], week=slate["week"],
                              known_game_ids={g.game_id for g in games})
            matched = {m.game_id: m.event_id for m in ms}
        else:
            table = importlib.import_module(lg.team_table).TABLE
            days = set(slate.get("dates") or [slate.get("date")])
            day_q = [q for q in quotes if q.commence_time and local_date(q.commence_time) in days]
            ms, refs = match_by_date(day_q, table, games)
            matched = {m.game_id: m.event_id for m in ms}
            for r in refs:
                board["refusals"].append({"scope": "event", "event_id": r.event_id,
                                          "reason": r.reason})
    opens = opening_quotes(store, lg.vendor_sport, set(matched.values())) if matched else {}
    names = _names(league)
    regimes = {}
    if league == "nfl":
        from deploy.odds_watch_job import load_regime_map
        regimes = load_regime_map(slate["season"], str(ROOT / "data"))

    fair: dict = {}
    for g in games:
        entry = {"game_id": g.game_id, "home": g.home, "away": g.away,
                 "home_name": names.get(g.home, g.home), "away_name": names.get(g.away, g.away),
                 "start": start_of(g.game_id), "markets": {}, "headline": headline,
                 "context": {}, "refusal": None}
        try:
            entry["context"] = _clean(context_of(g.game_id))
        except Exception:
            pass
        if entry["start"] and pd.Timestamp(entry["start"]) <= pd.Timestamp(board["generated_at"]):
            board_context.finish_injuries(entry["context"], None)
            # STARTED, not refused. The core rightly will not price a game
            # after kickoff, but that is the game's state, not a failure: as
            # a refusal each started game became its own red banner (the
            # message names the game and the time, so none grouped), which
            # on a Saturday is fifty of them.
            entry["started"] = True
            board["games"].append(_clean(entry))
            continue
        try:
            dist = model.predict(g.game_id, "now")
        except Exception as exc:
            entry["refusal"] = f"{type(exc).__name__}: {str(exc).strip(chr(39))[:220]}"
            board_context.finish_injuries(entry["context"], None)
            board["games"].append(_clean(entry))
            continue
        entry["model"] = model_view(dist)
        # Matchup detail (brief items 9-10): the distribution the board just
        # priced with, as bars the site draws and never computes.
        entry["model"]["margin_pmf"] = board_detail.margin_pmf(league, dist)
        entry["model"]["totals"] = board_detail.totals_view(league, dist)
        fair[g.game_id] = dist
        ev = matched.get(g.game_id)
        if ev is None:
            entry["refusal"] = ("no market price captured for this game" if snap is not None
                                else None)
        else:
            entry["event_id"] = ev
            eq = [q for q in quotes if q.event_id == ev]
            oq, oat = opens.get(ev, (None, None))
            for m in markets:
                v = B.price_market(dist, eq, m, weight=w, bands=bands,
                                   open_quotes=oq, open_at=oat)
                entry["markets"][m] = v.to_dict()
        if league == "nfl":
            apply_regime_cap(entry, headline, regimes, slate.get("week"))
        if league == "nhl":
            apply_early_season_cap(entry)
        # The card's injury list is cut to its tier's rule only now, once the
        # tier is known. Nothing in pricing reads context; a test builds the
        # board with and without it and compares every model field
        # (tests/core/test_board_context.py).
        m = entry["markets"].get(headline) or {}
        board_context.finish_injuries(entry["context"],
                                      m.get("tier") if m.get("status") == "priced" else None)
        board["games"].append(_clean(entry))

    attach_line_history(board, store, lg.vendor_sport, headline, fair)
    attach_check_flags(board, headline)
    logo = board_context.logos(league)
    for g in board["games"]:
        for side in ("home", "away"):
            if g.get(side) in logo:
                g[f"{side}_logo"] = logo[g[side]]
    if league == "cfb":
        for g in board["games"]:
            g["window"] = cfb_window(g.get("start"))
        wk = (board.get("slate") or {}).get("week")
        board["early_season"] = bool(wk is not None and wk <= CFB_EARLY_WEEKS)
    board["games"].sort(key=lambda e: (e["start"] or "", e["game_id"]))
    return board


#: The capture job polls every league in season at least once a day
#: (scripts/capture.py EARLY_UTC_HOUR). Odds older than a day plus slack mean
#: it has stopped -- credits, key, or the cron -- and every "market" price on
#: the board is yesterday's.
STALE_ODDS_HOURS = 26


def stale_odds(captured: str | None, now: str) -> str | None:
    """A refusal line when the newest snapshot is older than a day, else None."""
    if not captured:
        return None
    t = pd.Timestamp(captured)
    n = pd.Timestamp(now)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    n = n.tz_localize("UTC") if n.tzinfo is None else n
    hours = (n - t).total_seconds() / 3600
    if hours <= STALE_ODDS_HOURS:
        return None
    return (f"Odds are {hours:.0f} hours old: the last capture was {captured}, and "
            "captures run at least daily in season, so the capture job has stopped. "
            "Market prices and tiers below are from then, not now.")


#: Why a game has no price, in the words of the CFB side panel (brief
#: item 8). Matched on the refusal the core wrote; anything else is "other".
NOT_PRICED = (
    ("no_line", "No line captured yet",
     "The model's number is ready; the tier appears once a line is captured."),
    ("unrated", "FCS opponent, no rating", "The model won't price a team it doesn't rate."),
    ("whole_number", "Whole-number spread",
     "Priced since the measured key-number table (ADR 0028); refused only where a league has none."),
    ("started", "Already kicked off", "Never priced after kickoff."),
    ("neutral", "Neutral site", "The model has no home term to remove, so these are refused."),
    ("other", "Other refusal", "The reason is on the game."),
)


def _why_unpriced(g: dict) -> str | None:
    if g.get("started"):
        return "started"
    m = (g.get("markets") or {}).get(g.get("headline")) or {}
    if m.get("status") == "priced":
        return None
    r = (g.get("refusal") or m.get("refusal") or "").lower()
    if not r or "no market price captured" in r:
        return "no_line"
    if "not rated" in r or "no rating" in r:
        return "unrated"
    if "integer line" in r or "whole-number" in r:
        return "whole_number"
    if "neutral site" in r:
        return "neutral"
    return "other"


def not_priced(board: dict) -> list[dict]:
    """[{key, label, note, games, examples}] -- every category, zero included."""
    by: dict = {}
    for g in board.get("games", []):
        k = _why_unpriced(g)
        if k:
            by.setdefault(k, []).append(f"{g.get('away', '?')} @ {g.get('home', '?')}")
    return [{"key": k, "label": lab, "note": note, "games": len(by.get(k, [])),
             "examples": by.get(k, [])[:3]} for k, lab, note in NOT_PRICED]


def summarise(board: dict) -> dict:
    """The circuit's state and counts, so the site renders them rather than
    deriving them.

      up        games are priced against the market
      degraded  games exist but some or all cannot be priced (no odds yet,
                event refusals, a game-level refusal)
      refused   the league cannot be priced at all; the refusal says why
      idle      nothing on the slate today, and nothing wrong
    """
    started = [g for g in board.get("games", []) if g.get("started")]
    # The state is about the games still to be played: a slate half over is
    # not "degraded" because its first games have kicked off.
    games = [g for g in board.get("games", []) if not g.get("started")]
    head = [g["markets"].get(g.get("headline")) for g in games]
    priced = [m for m in head if m and m.get("status") == "priced"]
    tiers = {t: sum(1 for m in priced if m.get("tier") == t) for t in T.TIERS}
    league_ref = [r for r in board.get("refusals", []) if r.get("scope") == "league"]
    if not games and league_ref:
        state = "refused"
    elif not games:
        state = "idle"
    elif priced and len(priced) == len(games) and not board.get("refusals"):
        state = "up"
    else:
        state = "degraded"
    # Game-level refusals grouped by reason, so the evidence strip can say
    # "4 games refused: ..." in plain words. A reason's line-specific prefix
    # ("line -4.0: ") is dropped so the same refusal groups as one.
    import re
    grouped: dict[str, int] = {}
    for g in games:
        m = g["markets"].get(g.get("headline")) or {}
        r = g.get("refusal") or (m.get("refusal") if m.get("status") == "refused" else None)
        if r:
            r = re.sub(r"^line [-+\d.]+: ", "", r)
            grouped[r] = grouped.get(r, 0) + 1
    return {"state": state, "games": len(games), "priced": len(priced), "tiers": tiers,
            "started": len(started),
            "not_priced": not_priced(board),
            "refusals": len(board.get("refusals", [])),
            "reason": league_ref[0]["reason"] if league_ref else None,
            "game_refusals": [{"reason": r, "games": n} for r, n in
                              sorted(grouped.items(), key=lambda kv: -kv[1])]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=LEAGUES, action="append")
    ap.add_argument("--out", default=str(SITE))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = BronzeStore(BRONZE)
    R = _runner()
    failed = []
    for league in a.league or LEAGUES:
        try:
            board = build(league, store, R)
        except Exception as exc:
            traceback.print_exc()
            failed.append(league)
            board = {"league": league, "generated_at": _now(), "games": [],
                     "refusals": [{"scope": "league",
                                   "reason": f"export failed: {type(exc).__name__}: {exc}"}]}
        board["status"] = summarise(board)
        (out / f"board_{league}.json").write_text(json.dumps(_clean(board), indent=1) + "\n")
        priced = sum(1 for g in board["games"] if any(
            v.get("status") == "priced" for v in g.get("markets", {}).values()))
        print(f"{league}: {len(board['games'])} games, {priced} priced, "
              f"{len(board['refusals'])} refusal(s)"
              + (f" -- {board['refusals'][0]['reason'][:90]}" if board["refusals"] else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
