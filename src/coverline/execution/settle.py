"""Resolve every recorded signal against the final score.

WHAT THIS IS FOR
Every league paper-trades until a grade against the market says otherwise
(ADR 0024), and that grade needs, for each priced signal, the model's
probability, the market's, and what happened. The ledger holds the first two.
This writes the third, as an Outcome row -- for placed bets and paper trades
alike, since a game resolves the same way whether or not money was on it.

WHERE FINAL SCORES COME FROM
The files the live sources already read, keyed the way the matcher keys model
games -- so a signal's game_id is looked up, never re-derived from team names:

  mlb  model/mlb_schedule_*.csv   game_key (a DOUBLEHEADER key that appears
                                  twice is ambiguous and never settled)
  nhl  data/raw/nhl/nhl_*.parquet game_id
  nba  data/raw/sportsdataverse/nba_*.parquet  id, completed games only
  nfl  nflverse games.csv         "{season}-W{week:02d}-{home}-{away}"
  cfb  data/raw/cfb/espn_*.parquet  game_id (ESPN's event id), completed
                                    games only

WHAT IT WILL NOT DO
Settle a signal with no game_id or no side -- rows from before those fields
existed. They are counted as unsettleable, not guessed at. Nor settle twice:
an outcome is written once.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from coverline.execution.ledger import BetLedger, Outcome, Signal

_ROOT = Path(__file__).resolve().parents[3]
NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

Finals = Mapping[str, tuple[int, int]]


def result_for(sig: Signal, home: int, away: int) -> str:
    """win / loss / push for one signal, from the side it backed.

    The line is that side's own number, as quoted: a home -3.5 needs the
    home team to win by four. A moneyline is a line of zero, so a tie pushes
    -- the same convention pricing uses (recommend.price_candidate).
    """
    line = 0.0 if sig.line is None else float(sig.line)
    if sig.side == "home":
        v = (home - away) + line
    elif sig.side == "away":
        v = (away - home) + line
    elif sig.side == "over":
        v = (home + away) - line
    elif sig.side == "under":
        v = line - (home + away)
    else:
        raise ValueError(f"{sig.signal_id}: no side recorded; cannot settle")
    return "win" if v > 0 else ("push" if v == 0 else "loss")


@dataclass
class SettleReport:
    settled: list[str] = field(default_factory=list)
    already: int = 0
    pending: int = 0                    # game not final yet
    unsettleable: int = 0               # no game_id or side on the row

    def summary(self) -> str:
        return (f"{len(self.settled)} settled, {self.already} already, "
                f"{self.pending} awaiting a final, {self.unsettleable} "
                "unsettleable (rows without game_id/side)")


def settle(ledger: BetLedger, finals: Finals, league: str,
           at: str | None = None) -> SettleReport:
    at = at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = {o.signal_id for o in ledger.outcomes()}
    rep = SettleReport()
    for sig in ledger.signals():
        if sig.league != league:
            continue
        if sig.signal_id in done:
            rep.already += 1
            continue
        if not sig.game_id or not sig.side:
            rep.unsettleable += 1
            continue
        score = finals.get(sig.game_id)
        if score is None:
            rep.pending += 1
            continue
        h, a = int(score[0]), int(score[1])
        ledger.record_outcome(Outcome(signal_id=sig.signal_id, at=at,
                                      result=result_for(sig, h, a),
                                      home_score=h, away_score=a))
        rep.settled.append(sig.signal_id)
    return rep


# ------------------------------------------------------------- finals ----

def mlb_finals(root: Path = _ROOT) -> dict[str, tuple[int, int]]:
    frames = [pd.read_csv(p) for p in (root / "model" / "mlb_schedule_cache.csv",
                                       root / "model" / "mlb_schedule_current.csv")
              if p.exists()]
    d = pd.concat(frames, ignore_index=True).dropna(subset=["home_score", "away_score"])
    # A key written twice is a doubleheader the cron could not tell apart
    # (ATL202606170). Which score is which cannot be known, so neither is used.
    d = d[~d.game_key.duplicated(keep=False)]
    return {k: (int(h), int(a)) for k, h, a in zip(d.game_key, d.home_score, d.away_score)}


def nhl_finals(root: Path = _ROOT) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(glob.glob(str(root / "data" / "raw" / "nhl" / "nhl_*.parquet"))):
        d = pd.read_parquet(p)
        for g, h, a in zip(d.game_id, d.home_score, d.away_score):
            out[str(g)] = (int(h), int(a))
    return out


def nba_finals(root: Path = _ROOT) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(glob.glob(str(root / "data" / "raw" / "sportsdataverse" / "nba_*.parquet"))):
        d = pd.read_parquet(p)
        if "status_type_completed" in d.columns:
            d = d[d.status_type_completed.fillna(False).astype(bool)]
        d = d.dropna(subset=["home_score", "away_score"])
        for g, h, a in zip(d["id"], d.home_score, d.away_score):
            out[str(g)] = (int(h), int(a))
    return out


def cfb_finals(root: Path = _ROOT) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(glob.glob(str(root / "data" / "raw" / "cfb" / "espn_*.parquet"))):
        d = pd.read_parquet(p)
        # A scheduled game carries 0-0 here; the completion flag is what makes
        # that zero safe (model/ingest/cfb_espn.py).
        d = d[d.completed.astype(bool)].dropna(subset=["home_score", "away_score"])
        for g, h, a in zip(d.game_id, d.home_score, d.away_score):
            out[str(g)] = (int(h), int(a))
    return out


def nfl_finals(url: str = NFLVERSE_URL) -> dict[str, tuple[int, int]]:
    d = pd.read_csv(url)
    d = d[d.game_type == "REG"].dropna(subset=["home_score", "away_score"])
    return {f"{s}-W{int(w):02d}-{h}-{a}": (int(hs), int(as_))
            for s, w, h, a, hs, as_ in zip(d.season, d.week, d.home_team,
                                          d.away_team, d.home_score, d.away_score)}


#: The Odds API sport key each league's snapshots are stored under.
#: tests/core/test_settle.py holds this equal to the runner's own table.
VENDOR = {
    "nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf",
    "mlb": "baseball_mlb", "nhl": "icehockey_nhl", "nba": "basketball_nba",
}

FINALS: dict[str, Callable[[], Finals]] = {
    "mlb": mlb_finals, "nhl": nhl_finals, "nba": nba_finals, "nfl": nfl_finals,
    "cfb": cfb_finals,
}
