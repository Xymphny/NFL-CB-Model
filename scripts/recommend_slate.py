#!/usr/bin/env python3
"""Price a slate end to end: model -> market -> stake -> ledger.

This is the operator-facing command. Everything else in the new core is a
component; this is the thing you actually run.

    python3 scripts/recommend_slate.py --league nfl --week 2 --bankroll 5000
    python3 scripts/recommend_slate.py --league mlb --date 2026-09-22 --bankroll 5000
    python3 scripts/recommend_slate.py --league nhl --bankroll 5000        # today, ET
    python3 scripts/recommend_slate.py --league nba --bankroll 5000 --commit

Football slates are weeks; baseball, hockey and basketball slates are dates
(US Eastern, the date printed on the schedule). --week and --date are refused
for the league they do not belong to rather than silently ignored.

DRY BY DEFAULT. Nothing is written to the ledger without --commit, because
the ledger is append-only and a slate run that turns out to be misconfigured
cannot be taken back out of it.

WHAT THE STAKING WEIGHT IS, PER LEAGUE -- READ THIS BEFORE STAKING ANYTHING
The weight blends the model toward the market: p_used = w*p_model +
(1-w)*p_market. The only weight the repository can compute comes from
evidence/attempts.yaml, and every attempt there is graded MODEL AGAINST
MODEL -- does a change beat the version before it. None is graded against a
book. Measured against the market on 1,885 NFL games, the model's margin adds
nothing the spread does not already hold (slope -0.064, SE 0.072).

So the attempt-log weight is used ONLY for the football leagues, where it is
what this command has always used, and every run says what it does not
measure. MLB, NHL and NBA have no market-facing evidence of any kind, and
default to a weight of 0: the model's price and the market's are both
computed and recorded, and nothing is staked. That is paper trading, and it
is the only honest default until a market-relative grade exists. --commit
still records every row as not placed, which is how that grade gets built.
--market-weight overrides it and prints why that is a guess.

IT USES THE ROBUST SHRINKAGE WEIGHT, NOT THE POOLED ONE (football)
The robust figure is the one that survives losing any single attempt; both
are printed on every run. --pooled overrides this and prints a warning.

IT DOES NOT PLACE BETS. It prints recommendations and, with --commit, records
them. Placing is a human action and the fill gets recorded separately.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from coverline.core import evidence as E  # noqa: E402
from coverline.core import markets as M  # noqa: E402
from coverline.execution import recommend as Rc  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger  # noqa: E402
from coverline.execution.matching import local_date, match_by_date  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402

LEDGER_DIR = ROOT / "data" / "ledger"
BRONZE_DIR = ROOT / "data" / "bronze"

#: Refusals a loader raises when its inputs are missing or behind. Each
#: carries the exact command that fixes it, so the runner prints the message
#: and exits 2 -- the same code as a missing odds snapshot.
class InputsNotReady(Exception):
    pass


# ------------------------------------------------------------ loaders ----

def load_nfl(season: int, week: int):
    from coverline.leagues.nfl.live import RatingsSnapshotSource, load_schedule
    from coverline.leagues.nfl.model import NFLModel
    fixture = ROOT / "tests" / "fixtures" / f"nfl_{season}_week{week:02d}_schedule.csv"
    sched = pd.read_csv(fixture) if fixture.exists() else load_schedule([season])
    src = RatingsSnapshotSource.for_week(season, week, sched)
    return NFLModel(src), src


def load_cfb(season: int, week: int):
    from coverline.leagues.cfb.model import CFBModel
    from coverline.leagues.cfb.sources import CachedWalkForwardSource
    src = CachedWalkForwardSource.load()
    return CFBModel(src), src


def load_mlb(day: str):
    from coverline.leagues.mlb import live
    try:
        src = live.MLBLiveSource.load(day)
    except (live.NoSlate, live.StaleResults) as e:
        raise InputsNotReady(str(e)) from None
    return live.build_model(src), src, src.slate(day)


def _season_ending(day: str) -> int:
    d = datetime.fromisoformat(day)
    return d.year + 1 if d.month >= 8 else d.year


def load_nhl(day: str):
    from coverline.leagues.nhl import live
    try:
        src = live.NHLLiveSource.load(_season_ending(day))
    except (live.MissingSeasonData, live.StaleResults) as e:
        raise InputsNotReady(str(e)) from None
    return live.build_model(src), src, src.slate(day)


def load_nba(day: str):
    from coverline.leagues.nba import live
    try:
        src = live.NBALiveSource.load(live.season_of(datetime.fromisoformat(day).date()))
    except (live.SeasonGap, live.StaleResults) as e:
        raise InputsNotReady(str(e)) from None
    return live.build_model(src), src, src.slate(day)


@dataclass(frozen=True)
class League:
    slate: str                    # "week" or "date"
    vendor_sport: str             # the key capture.py stores snapshots under
    default_market: str
    loader: Callable
    #: Where a staking weight could come from. None means there is NO
    #: market-facing evidence at all and the default weight is 0.
    weight_source: str | None
    team_table: str | None = None  # module holding TABLE, for date leagues


LEAGUES: dict[str, League] = {
    "nfl": League("week", "americanfootball_nfl", "spreads", load_nfl, "attempt_log"),
    "cfb": League("week", "americanfootball_ncaaf", "spreads", load_cfb, "attempt_log"),
    "mlb": League("date", "baseball_mlb", "moneyline", load_mlb, None,
                  "coverline.leagues.mlb.teams"),
    "nhl": League("date", "icehockey_nhl", "moneyline", load_nhl, None,
                  "coverline.leagues.nhl.teams"),
    "nba": League("date", "basketball_nba", "spread", load_nba, None,
                  "coverline.leagues.nba.teams"),
}

#: Kept for callers that imported the old name.
LOADERS = {k: v.loader for k, v in LEAGUES.items()}


def latest_snapshot(store: BronzeStore, league: str) -> Path | None:
    """Newest snapshot for the league.

    capture.py stores snapshots under the VENDOR's sport key
    ("americanfootball_nfl"), and this used to look them up under the
    league's short name ("nfl") -- so a captured snapshot was never found and
    the runner reported none. Both spellings are checked, vendor key first.
    """
    for sport in (LEAGUES[league].vendor_sport, league):
        rows = list(store.snapshots(sport))
        if rows:
            return store.root / rows[-1]["path"]
    return None


def today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


# ------------------------------------------------------------- weight ----

def choose_weight(args, lg: League) -> float:
    if lg.weight_source == "attempt_log":
        log = E.load_attempts()
        weight = log.weight() if args.pooled else log.robust_weight()
        print(f"attempt log: {log.summary()}")
        if args.pooled:
            print(f"  USING POOLED WEIGHT {weight:.4f}. It rests on one attempt -- "
                  f"the robust figure is {log.robust_weight():.4f}.")
        else:
            print(f"  using robust weight {weight:.4f} "
                  f"(pooled would be {log.weight():.4f})")
        print("  NOTE: this weight is graded model-against-model, never against "
              "a book. Against the market the NFL margin's slope is -0.064 "
              "(SE 0.072) -- no measured edge over the spread.")
    else:
        weight = 0.0
        print(f"attempt log: not used for {args.league.upper()} -- no attempt "
              "there was graded against a market.")
        print("  PAPER TRADING: robust weight not applicable, weight 0. Model "
              "and market prices are computed and recorded; nothing is staked.")

    if args.market_weight is not None:
        print(f"  --market-weight {args.market_weight:.4f} OVERRIDES the "
              f"default {weight:.4f}. No market-facing grade supports any "
              "particular value; this is a guess with money on it.")
        weight = args.market_weight
    if weight <= 0.0:
        print("\nNothing will be staked at this weight.")
    return weight


# --------------------------------------------------------------- main ----

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=sorted(LEAGUES))
    ap.add_argument("--season", type=int, default=2026, help="football only")
    ap.add_argument("--week", type=int, help="football only")
    ap.add_argument("--date", help="mlb/nhl/nba: YYYY-MM-DD, US Eastern (default today)")
    ap.add_argument("--snapshot", help="a bronze odds snapshot; default is the newest")
    ap.add_argument("--market", help="default: the league's primary market")
    ap.add_argument("--bankroll", type=float, required=True)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--max-fraction", type=float, default=0.02)
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--pooled", action="store_true",
                    help="use the pooled shrinkage weight instead of the robust one")
    ap.add_argument("--market-weight", type=float, default=None,
                    help="override the staking weight (0..1); prints a warning")
    ap.add_argument("--commit", action="store_true",
                    help="write the signals to the append-only ledger")
    args = ap.parse_args(argv)
    lg = LEAGUES[args.league]

    if lg.slate == "week":
        if args.date is not None:
            ap.error(f"{args.league} slates are weeks; --date does not apply")
        if args.week is None:
            ap.error(f"--week is required for {args.league}")
    else:
        if args.week is not None:
            ap.error(f"{args.league} slates are dates; use --date, not --week")
        args.date = args.date or today_et()
    if args.market_weight is not None and not 0.0 <= args.market_weight <= 1.0:
        ap.error("--market-weight must be between 0 and 1")

    weight = choose_weight(args, lg)

    try:
        if lg.slate == "week":
            model, source = lg.loader(args.season, args.week)
            games = None
            label = f"{args.season} week {args.week}"
        else:
            model, source, games = lg.loader(args.date)
            label = args.date
    except InputsNotReady as e:
        print(f"\n{args.league.upper()} inputs are not ready:\n  {e}")
        return 2

    market = args.market or lg.default_market
    try:
        M.require_offered(market, model.primary_markets, args.league)
    except ValueError as e:
        print(f"\n{e}")
        return 2
    print(f"\n{args.league.upper()} {label}: {model.__class__.__name__}, "
          f"markets {list(model.primary_markets)}, pricing {market}")
    if games is not None:
        print(f"model slate: {len(games)} game(s) not yet started")

    store = BronzeStore(BRONZE_DIR)
    snap_path = Path(args.snapshot) if args.snapshot else latest_snapshot(
        store, args.league)
    if snap_path is None or not Path(snap_path).exists():
        print(f"\nNo odds snapshot for {args.league}. Capture one first:")
        print("  see src/coverline/execution/capture.py, or pass --snapshot")
        return 2

    payload = store.read_snapshot(snap_path)["payload"]
    quotes = normalize(payload, captured_at=str(snap_path))
    events = sorted({q.event_id for q in quotes})
    print(f"snapshot {Path(snap_path).name}: {len(events)} events, "
          f"{len(quotes)} quotes")

    # Resolve the book's event ids to model game ids. Without this the
    # runner has a model and a market and no way to know they describe the
    # same game.
    if lg.slate == "week":
        from coverline.leagues.nfl.events import match_events, unmatched
        known = set(source.game_ids()) if hasattr(source, "game_ids") else None
        matches = match_events(quotes, season=args.season, week=args.week,
                               known_game_ids=known)
        skipped = [str(e) for e in unmatched(quotes, matches)]
    else:
        import importlib
        table = importlib.import_module(lg.team_table).TABLE
        on_day = [q for q in quotes if local_date(q.commence_time) == args.date]
        other = len({q.event_id for q in quotes}) - len({q.event_id for q in on_day})
        if other:
            print(f"  {other} event(s) on other dates ignored")
        matches, refusals = match_by_date(on_day, table, games)
        skipped = [f"{r.event_id}: {r.reason}" for r in refusals]
    print(f"matched {len(matches)} events to model games"
          + (f"; {len(skipped)} unmatched" if skipped else ""))
    # Named, not counted. A slate that silently prices 6 of 16 games is the
    # shape of a problem that goes unnoticed for weeks.
    for e in skipped[:12]:
        print(f"    unmatched: {e}")

    ledger = BetLedger(LEDGER_DIR) if args.commit else None
    placed = declined = withheld = 0

    for m in matches:
        ev = m.event_id
        try:
            dist = model.predict(m.game_id, "now")
        except Exception as exc:
            print(f"  {m.game_id}: no prediction ({type(exc).__name__}: "
                  f"{str(exc)[:160]}) -- skipped")
            continue
        sigs = Rc.recommend(dist=dist, quotes=quotes, event_id=ev,
                            market=market, league=args.league,
                            bankroll=args.bankroll, shrinkage=weight,
                            ledger=ledger, primary_markets=model.primary_markets,
                            kelly_multiple=args.kelly,
                            max_bankroll_fraction=args.max_fraction,
                            min_edge=args.min_edge)
        if not sigs:
            withheld += 1
            continue
        for s in sigs:
            if s.placed:
                placed += 1
                line = f"{s.line:+.1f}" if s.line is not None else "ML"
                print(f"  BET  {s.selection:24} {line:>6} @ {s.book:12} "
                      f"stake {s.stake:>9,.2f}  edge {s.edge_used:+.2%}")
            else:
                declined += 1
        if weight <= 0.0:
            # Paper trading: show where model and market disagree most, so a
            # run at weight 0 is still informative.
            top = max(sigs, key=lambda s: s.p_model - s.p_market)
            who = (f"{m.away}@{m.home}" if hasattr(m, "home")
                   else f"{m.away_code}@{m.home_code}")
            print(f"  paper {who}: {top.selection} model "
                  f"{top.p_model:.3f} vs market {top.p_market:.3f} "
                  f"({top.p_model - top.p_market:+.3f})")

    print(f"\n{placed} recommended, {declined} declined, "
          f"{withheld} events withheld (no priceable market)")
    if ledger is not None:
        print(f"written to {LEDGER_DIR}")
        print(f"conversion so far: {ledger.conversion()}")
    else:
        print("Dry run -- nothing written. Add --commit to record these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
