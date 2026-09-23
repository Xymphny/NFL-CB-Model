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

SITE = ROOT / "data" / "site"
BRONZE = ROOT / "data" / "bronze"
LEAGUES = ("nfl", "cfb", "mlb", "nhl", "nba")

SPORT_NOTES = {
    "nfl": "Weekly slate. Spreads price key numbers (3, 7) with a measured correction.",
    "cfb": "Weekly slate.",
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
        raise R.InputsNotReady(
            "The CFB live source prices from a historical cache (2021–2023) "
            "and cannot price a current game. No CFB board until a live CFB "
            "source exists.")
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
        ctx = {g.game_id: _nfl_context(r) for g, r in zip(games, wk.itertuples())}
        return (model, src, games, {"kind": "week", "season": season, "week": week},
                lambda gid: starts.get(gid), lambda gid: ctx.get(gid, {}),
                {"ratings_snapshot": f"{season}-week-{week:02d}"})
    day = day or _today_et()
    model, src, games = lg.loader(day)
    if league == "nba":
        s = src.schedule
        start = lambda gid: s.loc[gid].tip.strftime("%Y-%m-%dT%H:%M:%SZ")
        done = src.history
        fresh = {"latest_final": str(done.tip.max()) if len(done) else None}
        return (model, src, games, {"kind": "date", "date": day}, start,
                lambda gid: _nba_context(src, gid), fresh)
    if league == "nhl":
        s = src.schedule
        start = lambda gid: s.loc[gid].start.strftime("%Y-%m-%dT%H:%M:%SZ")
        fresh = {"latest_final": str(src.games.start.max()) if len(src.games) else None}
        return (model, src, games, {"kind": "date", "date": day}, start,
                lambda gid: _nhl_context(src, gid), fresh)
    if league == "mlb":
        start = lambda gid: src._game(gid).get("start_utc")
        played = src.schedule.dropna(subset=["home_score", "away_score"])
        fresh = {"slate_fetched_at": src.payload.get("fetched_at"),
                 "latest_final": str(played.date.max())}
        return (model, src, games, {"kind": "date", "date": day}, start,
                lambda gid: _mlb_context(src, gid), fresh)
    raise ValueError(league)


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
        if league == "nfl":
            # The fields the rating-only vector reads (MARGIN_COEFFICIENTS_V1)
            # and the ones the card explains it with.
            keep = ("total_rating", "offense_voa", "defense_voa", "special_teams_voa",
                    "rating_p05", "rating_p95")
            return sorted(({"team": t, **{k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k))
                                          for k in keep}}
                           for t, r in src.ratings.items()), key=lambda x: -(x["total_rating"] or 0))
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

    lg = R.LEAGUES[league]
    snap, captured = latest_snapshot(store, lg.vendor_sport)
    board["freshness"]["odds_captured_at"] = captured

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
    if snap is None:
        board["refusals"].append({"scope": "league", "reason":
            "No odds captured yet for this league. The model's view is shown; "
            "market prices appear once the capture job has run."})

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
            day_q = [q for q in quotes if q.commence_time and local_date(q.commence_time) == slate["date"]]
            ms, refs = match_by_date(day_q, table, games)
            matched = {m.game_id: m.event_id for m in ms}
            for r in refs:
                board["refusals"].append({"scope": "event", "event_id": r.event_id,
                                          "reason": r.reason})
    opens = opening_quotes(store, lg.vendor_sport, set(matched.values())) if matched else {}
    names = _names(league)

    for g in games:
        entry = {"game_id": g.game_id, "home": g.home, "away": g.away,
                 "home_name": names.get(g.home, g.home), "away_name": names.get(g.away, g.away),
                 "start": start_of(g.game_id), "markets": {}, "headline": headline,
                 "context": {}, "refusal": None}
        try:
            entry["context"] = _clean(context_of(g.game_id))
        except Exception:
            pass
        try:
            dist = model.predict(g.game_id, "now")
        except Exception as exc:
            entry["refusal"] = f"{type(exc).__name__}: {str(exc).strip(chr(39))[:220]}"
            board["games"].append(entry)
            continue
        entry["model"] = model_view(dist)
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
        board["games"].append(_clean(entry))

    board["games"].sort(key=lambda e: (e["start"] or "", e["game_id"]))
    return board


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
        (out / f"board_{league}.json").write_text(json.dumps(_clean(board), indent=1) + "\n")
        priced = sum(1 for g in board["games"] if any(
            v.get("status") == "priced" for v in g.get("markets", {}).values()))
        print(f"{league}: {len(board['games'])} games, {priced} priced, "
              f"{len(board['refusals'])} refusal(s)"
              + (f" -- {board['refusals'][0]['reason'][:90]}" if board["refusals"] else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
