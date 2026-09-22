"""Match odds-feed events to model games, for leagues that play on dates.

The NFL and CFB have their own matcher keyed on season and week
(leagues/nfl/events.py). MLB, the NHL and the NBA play on calendar dates, often
the same two teams on consecutive nights and sometimes twice in one day, so
they need a matcher keyed on a local date that handles doubleheaders.

THREE THINGS THIS REFUSES TO DO, EACH FROM A BUG ALREADY FOUND IN THIS REPO

1. NO FALLBACK TO A NAME. The legacy MLB board looks a team up in the
   probables feed and, when the lookup misses, carries the Odds API's full
   name forward in the field meant for a code (``pb.get("home_team") or
   game.get("home_team")``). Measured across the committed divergence files,
   27 of 346 rows -- 11 of 17 on 2026-09-21 -- ended up that way, and none of
   them carries a model probability. Nothing flagged it, because a name in a
   code field is a string like any other. Here an unknown name RAISES.

2. NO GUESSING WHICH GAME OF A DOUBLEHEADER. Two events with the same teams
   on the same date are ordered by start time and paired with the model's
   game numbers in order -- but only when both sides agree on how many games
   there are. When they do not, or when the model side has two games under
   one id (the schedule cache has one: ATL202606170, the June 17 Braves-Giants
   doubleheader, both games keyed as game 0), the whole group is REFUSED and
   named. Pricing the wrong game of a doubleheader prices the wrong starters.

3. NO UTC DATES. The Odds API gives kickoffs in UTC, so a 7:10 pm Pacific
   first pitch is on the NEXT UTC day. Dates are taken in US Eastern, which
   puts every North American start on the calendar day it is played. A game
   played abroad at an unusual hour may land on a different date and will
   then be reported as unmatched -- never matched to the wrong game.

Name normalisation is canonicalisation, not fuzzy matching: accents stripped
(so "Montréal" and "Montreal" agree), full stops removed ("St. Louis" and
"St Louis" agree), case and spacing folded. Two different names that
canonicalise the same are refused at table construction.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from coverline.execution.normalize import Quote

EASTERN = ZoneInfo("America/New_York")


class UnknownTeamName(KeyError):
    """A name the table does not know. Stops the run -- see module docstring."""


def canonical(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace(".", " ").lower()
    return " ".join(folded.split())


class TeamTable:
    """Feed name -> model code, with a canonicalised lookup."""

    def __init__(self, league: str, names: Mapping[str, str],
                 verified_against_odds_api: bool) -> None:
        self.league = league
        self.verified = verified_against_odds_api
        self._by_canon: dict[str, str] = {}
        for name, code in names.items():
            key = canonical(name)
            if key in self._by_canon and self._by_canon[key] != code:
                raise ValueError(
                    f"{league}: {name!r} canonicalises to {key!r}, already "
                    f"mapped to {self._by_canon[key]} -- refusing to guess"
                )
            self._by_canon[key] = code

    def __len__(self) -> int:
        return len(set(self._by_canon.values()))

    def codes(self) -> set[str]:
        return set(self._by_canon.values())

    def to_code(self, name: str) -> str:
        try:
            return self._by_canon[canonical(name)]
        except KeyError:
            hint = ("" if self.verified else
                    " This table has never been checked against a live Odds "
                    "API payload, so a first-run mismatch is expected: add "
                    "the feed's spelling deliberately rather than loosening "
                    "the match.")
            raise UnknownTeamName(
                f"{self.league}: no code for {name!r}.{hint}") from None


def local_date(commence_time: str) -> str:
    """The US Eastern calendar date of a UTC timestamp."""
    ts = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    return ts.astimezone(EASTERN).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class ModelGame:
    """One game as the model knows it."""

    game_id: str
    date: str          # YYYY-MM-DD, the local date the game is played
    home: str
    away: str
    order: int = 0     # game number within a doubleheader, 0-based


@dataclass(frozen=True)
class Match:
    event_id: str
    game_id: str
    home: str
    away: str
    commence_time: str


@dataclass(frozen=True)
class Refusal:
    event_id: str
    reason: str


def _events(quotes: Iterable[Quote]) -> dict[str, tuple[str, str, str]]:
    """event_id -> (home name, away name, commence time)."""
    out: dict[str, tuple[str, str, str]] = {}
    for q in quotes:
        out.setdefault(q.event_id, (q.home_team, q.away_team, q.commence_time))
    return out


def match_by_date(
    quotes: Sequence[Quote],
    table: TeamTable,
    games: Sequence[ModelGame],
) -> tuple[list[Match], list[Refusal]]:
    """Pair feed events with model games. Returns (matches, refusals).

    Every event ends up in exactly one of the two lists, so a slate that
    prices 6 of 16 games says which 10 and why.
    """
    by_key: dict[tuple[str, str, str], list[ModelGame]] = defaultdict(list)
    for g in games:
        by_key[(g.date, g.home, g.away)].append(g)

    grouped: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    matches: list[Match] = []
    refusals: list[Refusal] = []
    for eid, (home, away, ct) in _events(quotes).items():
        try:
            key = (local_date(ct), table.to_code(home), table.to_code(away))
        except UnknownTeamName as exc:
            # A refusal, not an exception. One unrecognised name -- likely on
            # the tables not yet checked against the feed -- used to abort
            # the whole slate with a traceback, breaking the promise above.
            # It is still NAMED, with the table's own hint, so a systematic
            # gap is as loud as before; it just no longer takes the other
            # games down with it.
            refusals.append(Refusal(eid, str(exc).strip("'\"")))
            continue
        grouped[key].append((ct, eid))

    for key, evs in grouped.items():
        evs.sort()
        model = sorted(by_key.get(key, []), key=lambda g: g.order)
        date, home, away = key
        if not model:
            refusals.extend(Refusal(e, f"no model game for {away}@{home} on {date}")
                            for _, e in evs)
            continue
        ids = [g.game_id for g in model]
        if len(set(ids)) != len(ids):
            refusals.extend(Refusal(
                e, f"{away}@{home} {date}: the model has {len(ids)} games under "
                   f"{len(set(ids))} id(s) {sorted(set(ids))} -- which is which "
                   "cannot be known") for _, e in evs)
            continue
        if len(model) != len(evs):
            refusals.extend(Refusal(
                e, f"{away}@{home} {date}: feed has {len(evs)} event(s), model "
                   f"has {len(model)} game(s) -- refusing to guess the pairing")
                for _, e in evs)
            continue
        for (ct, eid), g in zip(evs, model):
            matches.append(Match(event_id=eid, game_id=g.game_id, home=home,
                                 away=away, commence_time=ct))
    return matches, refusals
