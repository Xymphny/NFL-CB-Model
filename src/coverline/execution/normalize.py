"""Raw API payloads to priceable rows.

Kept separate from the client so that what lands in bronze is exactly what the
API returned. Normalisation is code that will change; raw data is not allowed
to change with it, and re-deriving rows from stored snapshots must always be
possible.

The output is deliberately long-format -- one row per bookmaker per outcome --
because that is the shape devigging needs: core.pricing removes the margin
from a complete two-sided or n-way market, and a wide format encourages the
single most common CLV mistake, which is passing one side's price on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Quote:
    """One bookmaker's price on one outcome of one market."""

    event_id: str
    sport: str
    commence_time: str
    home_team: str
    away_team: str
    bookmaker: str
    market: str
    outcome: str
    price_decimal: float
    point: float | None
    last_update: str | None
    captured_at: str

    @property
    def is_priceable(self) -> bool:
        return self.price_decimal > 1.0


def normalize(payload: Iterable[dict[str, Any]], captured_at: str,
              sport: str | None = None) -> list[Quote]:
    """Flatten an /odds response into quotes.

    Tolerant of missing optional fields, strict about the ones pricing needs:
    a quote without a usable decimal price is dropped rather than defaulted,
    because a fabricated price is worse than a missing one.
    """
    out: list[Quote] = []
    for event in payload or []:
        eid = event.get("id")
        if not eid:
            continue
        ev_sport = event.get("sport_key") or sport or ""
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        for book in event.get("bookmakers") or []:
            bkey = book.get("key", "")
            for market in book.get("markets") or []:
                mkey = market.get("key", "")
                last = market.get("last_update") or book.get("last_update")
                for oc in market.get("outcomes") or []:
                    price = oc.get("price")
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        continue
                    point = oc.get("point")
                    out.append(Quote(
                        event_id=eid, sport=ev_sport, commence_time=commence,
                        home_team=home, away_team=away, bookmaker=bkey,
                        market=mkey, outcome=oc.get("name", ""),
                        price_decimal=price,
                        point=float(point) if point is not None else None,
                        last_update=last, captured_at=captured_at,
                    ))
    return out


def two_sided(quotes: Sequence[Quote], *, event_id: str, market: str,
              bookmaker: str) -> list[Quote]:
    """Both sides of one market at one book, ordered deterministically.

    Returns [] rather than a partial market. core.pricing.devig cannot remove
    a margin from one side, and handing it an incomplete market is the mistake
    this function exists to make impossible.
    """
    sides = [q for q in quotes
             if q.event_id == event_id and q.market == market
             and q.bookmaker == bookmaker and q.is_priceable]
    if len(sides) < 2:
        return []
    return sorted(sides, key=lambda q: q.outcome)


def best_price(quotes: Sequence[Quote], *, event_id: str, market: str,
               outcome: str) -> Quote | None:
    """The best available price across books -- the line-shopping primitive.

    Consistency across books is irrelevant here, which is exactly why line
    shopping may draw on any source while the CLV baseline may not (ADR 0002).
    """
    candidates = [q for q in quotes
                  if q.event_id == event_id and q.market == market
                  and q.outcome == outcome and q.is_priceable]
    return max(candidates, key=lambda q: q.price_decimal, default=None)
