"""Matching odds-feed events to model games.

THE GAP THIS CLOSES
An odds snapshot identifies a game by the book's own event id and by full
team names ("Los Angeles Rams"). The model identifies it by
"2026-W02-LA-NYG". Without a translation the runner has a model and a market
and no way to know they are about the same game, which is the state the slate
runner was in when it was first written.

A SECOND MAPPING IS A SECOND CHANCE TO DISAGREE
deploy/odds_watch_job.py already carries a name-to-code table. Writing another
one here creates exactly the failure mode found in ADR 0004: two sources with
different vocabularies, joined by membership, degrading silently when they
drift. So this table is checked against the legacy one by a guard test, and
the canonical codes are checked against the ratings snapshot's.

The alternative -- importing the legacy table -- would tie the new core to a
deploy module it is otherwise independent of, and would mean a change there
silently changes behaviour here. A duplicated table that is TESTED to agree
is the lesser evil, and the test is what makes it so.

UNKNOWN NAMES RAISE
No fuzzy matching, no partial-string fallback. A relocation or a rebrand
should stop the run, not quietly match a team to its predecessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from coverline.execution.normalize import Quote

#: Full name as the odds feed writes it -> canonical code.
#: Kept in step with deploy/odds_watch_job.ODDS_TEAM_TO_ABBR by a guard test.
NAME_TO_CODE: dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


class UnknownTeamName(KeyError):
    """The feed used a name nothing maps. Stop rather than guess."""


def to_code(name: str) -> str:
    try:
        return NAME_TO_CODE[name.strip()]
    except KeyError:
        raise UnknownTeamName(
            f"{name!r} is not a known team name. A relocation or rebrand "
            "should stop the run, not quietly match to a predecessor -- add "
            "it to NAME_TO_CODE deliberately."
        ) from None


@dataclass(frozen=True)
class EventMatch:
    """One odds-feed event, resolved to a model game id."""

    event_id: str
    game_id: str
    home_code: str
    away_code: str


def match_events(quotes: Sequence[Quote], *, season: int, week: int,
                 known_game_ids: Iterable[str] | None = None) -> list[EventMatch]:
    """Resolve every event in a snapshot to a model game id.

    `known_game_ids` filters to games the model can actually price. Passing it
    is what stops the runner pricing a game whose features do not exist --
    silently producing nothing rather than saying which games were dropped.
    """
    known = set(known_game_ids) if known_game_ids is not None else None
    seen: dict[str, EventMatch] = {}

    for q in quotes:
        if q.event_id in seen or not q.home_team or not q.away_team:
            continue
        home, away = to_code(q.home_team), to_code(q.away_team)
        gid = f"{season}-W{week:02d}-{home}-{away}"
        if known is not None and gid not in known:
            continue
        seen[q.event_id] = EventMatch(event_id=q.event_id, game_id=gid,
                                      home_code=home, away_code=away)
    return sorted(seen.values(), key=lambda m: m.game_id)


def unmatched(quotes: Sequence[Quote], matches: Sequence[EventMatch]) -> list[str]:
    """Events in the snapshot that resolved to nothing the model can price.

    Returned rather than logged, so a caller has to decide what to do about
    them. A slate that silently prices 6 of 16 games is the shape of a
    problem that goes unnoticed for weeks.
    """
    matched = {m.event_id for m in matches}
    return sorted({q.event_id for q in quotes} - matched)
