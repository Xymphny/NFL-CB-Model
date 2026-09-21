"""One vocabulary for markets, because there were two and they never met.

THE HALF-CONNECTION THIS CLOSES
Leagues declare what they price in their own words -- "spread", "moneyline",
"total", "runline", "puck_line". Quotes arrive from the vendor in the
vendor's words -- "spreads", "h2h", "totals". Nothing in this repository
translated between them, and nothing ever had to, because
``LeagueModel.primary_markets`` had NO PRODUCTION CONSUMER AT ALL: only tests
read it. The recommender takes a market string, matches it against the
vendor key on a quote, and never asks the league whether it prices that
market.

So the declared market list was documentation, not a control. A caller could
price any league on any market and get a signal back. That is the same shape
as ADR 0011 -- a market withheld in a list the object does not enforce --
except that one was caught because the distribution refused. This one had no
object to refuse.

WHY THE MAPPING IS MANY-TO-ONE
Three leagues call their handicap "spread", the MLB calls it the "runline"
and the NHL the "puck line", and the vendor calls all three "spreads". The
difference is not cosmetic: a runline is always 1.5 and a puck line is always
1.5, while an NFL spread moves, which is why the NHL model prices SHAPE and
the NFL model prices a mean. Collapsing them at the vendor boundary is
correct; collapsing them in the league's own vocabulary would lose that.

So the forward map is many-to-one and the reverse is one-to-many, and callers
that need the reverse get a tuple rather than a guess.
"""

from __future__ import annotations

from typing import Sequence

#: Internal market name -> the vendor's market key.
VENDOR_KEY: dict[str, str] = {
    "spread": "spreads",
    "runline": "spreads",
    "puck_line": "spreads",
    "moneyline": "h2h",
    "total": "totals",
}

#: Vendor key -> every internal name that maps to it.
INTERNAL_NAMES: dict[str, tuple[str, ...]] = {}
for _internal, _vendor in VENDOR_KEY.items():
    INTERNAL_NAMES.setdefault(_vendor, ())
    INTERNAL_NAMES[_vendor] += (_internal,)


class MarketNotOffered(ValueError):
    """Raised when a league is asked to price a market it does not offer.

    Not a KeyError and not silence. A caller who asks for NFL totals is
    either wrong or has just enabled something without saying so, and both
    deserve to stop rather than to receive a number.
    """


class UnknownMarket(ValueError):
    """Raised on a market name in neither vocabulary.

    Separate from MarketNotOffered because the remedies differ: one means
    "this league does not price that", the other means "nothing prices that
    and the name is probably a typo".
    """


def vendor_key(market: str) -> str:
    """Translate an internal market name to the vendor's key.

    Accepts a vendor key unchanged, so a caller already holding one is not
    forced to round-trip it. That tolerance is deliberate and narrow: it
    exists so this module can be introduced without rewriting every call
    site at once, and the tolerance itself is tested.
    """
    if market in VENDOR_KEY:
        return VENDOR_KEY[market]
    if market in INTERNAL_NAMES:
        return market
    raise UnknownMarket(
        f"{market!r} is neither an internal market name "
        f"({sorted(VENDOR_KEY)}) nor a vendor key ({sorted(INTERNAL_NAMES)})"
    )


def require_offered(market: str, primary_markets: Sequence[str],
                    league: str = "") -> str:
    """Check a league offers this market, and return its internal name.

    A VENDOR key is resolved against what the league offers, so asking for
    "spreads" on the NHL resolves to "puck_line" and on the NFL to "spread".
    If a vendor key matches nothing the league offers, that is a refusal, not
    a guess -- the NHL without its rules layer offers no handicap at all, and
    pricing one anyway is exactly the failure this module exists to stop.
    """
    offered = tuple(primary_markets)
    where = f"{league} " if league else ""
    if market in offered:
        return market
    if market in INTERNAL_NAMES:
        matches = [m for m in INTERNAL_NAMES[market] if m in offered]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise MarketNotOffered(
                f"{where}does not offer any market under the vendor key "
                f"{market!r}. It offers {sorted(offered)}."
            )
        raise UnknownMarket(
            f"{where}offers more than one market under the vendor key "
            f"{market!r} ({sorted(matches)}), so the vendor key is ambiguous "
            "here and the internal name must be given."
        )
    if market in VENDOR_KEY:
        raise MarketNotOffered(
            f"{where}does not offer {market!r}. It offers {sorted(offered)}."
        )
    raise UnknownMarket(
        f"{market!r} is neither an internal market name nor a vendor key"
    )
