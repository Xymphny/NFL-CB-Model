"""One game, one market, as the dashboard shows it -- computed here, not there.

The site renders what this returns and computes nothing of its own (the
dashboard brief). Every number comes from the same functions the operator
command uses: the league model's distribution, execution.recommend's cover
and total probabilities, core.pricing's devig, core.tiers, and the staking
engine through recommend.price_candidate.

WHICH LINE
Books hang different numbers. The board prices the CONSENSUS line -- the
point most books quote for the home (or over) side -- and devigs each book's
two-sided market at that line; p_market is the median across books. The best
price is the highest decimal on the model's preferred side at that line.

WHICH SIDE
The one the model prefers: the side whose model probability exceeds the
market's. Edge is that difference in probability points, and its size sets
the tier (ADR 0025). The stake is what the staking engine sizes at the best
price with the league's market-grade weight (ADR 0024) -- zero today, and the
card says why rather than showing a figure.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from coverline.core import markets as M
from coverline.core import pricing as P
from coverline.core import tiers as T
from coverline.execution import recommend as Rc
from coverline.execution.normalize import Quote

#: Staking parameters the operator command defaults to, so the board's stake
#: is the one the command would size.
KELLY = 0.25
MAX_FRACTION = 0.02
DEVIG = "power"


@dataclass
class MarketView:
    market: str                      # internal name: spread / moneyline / total / ...
    status: str                      # "priced" or "refused"
    refusal: str | None = None
    line: float | None = None        # the preferred side's own number
    side: str | None = None          # home / away / over / under
    selection: str | None = None     # the feed's name for that side
    best_price_decimal: float | None = None
    best_price_american: float | None = None
    best_book: str | None = None
    books: int = 0
    p_model: float | None = None     # preferred side
    p_market: float | None = None    # preferred side, median devigged across books
    edge: float | None = None        # p_model - p_market, probability points
    tier: str | None = None
    stake_fraction: float = 0.0      # of bankroll; 0 until the league is graded
    unsized_reason: str | None = None
    open_line: float | None = None
    open_p_market: float | None = None
    open_at: str | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _side_of(q: Quote) -> str | None:
    o = q.outcome.strip().lower()
    if o in ("over", "under"):
        return o
    h = Rc._is_home_outcome(q)
    return None if h is None else ("home" if h else "away")


def consensus(quotes: Sequence[Quote], vendor: str) -> dict | None:
    """Per book, the two sides at the consensus line; None if no book has a
    complete market. Keyed by the anchor side ("home" or "over")."""
    by_book: dict[str, dict[str, Quote]] = defaultdict(dict)
    for q in quotes:
        if q.market != vendor:
            continue
        s = _side_of(q)
        if s:
            by_book[q.bookmaker][s] = q
    anchor, other = ("over", "under") if vendor == "totals" else ("home", "away")
    complete = {b: s for b, s in by_book.items() if anchor in s and other in s}
    if not complete:
        return None
    lines = Counter(s[anchor].point for s in complete.values())
    # Most common line; ties go to the line nearer zero (spreads) or lower
    # (totals) so the choice is deterministic.
    top = max(lines.values())
    line = sorted((l for l, c in lines.items() if c == top),
                  key=lambda x: (abs(x) if x is not None else 0))[0]
    at_line = {b: s for b, s in complete.items() if s[anchor].point == line}
    return {"anchor": anchor, "other": other, "line": line, "books": at_line}


def market_probability(books: dict, anchor: str, other: str) -> float:
    """Median devigged anchor-side probability across books."""
    ps = [float(P.devig([s[anchor].price_decimal, s[other].price_decimal], DEVIG)[0])
          for s in books.values()]
    return float(statistics.median(ps))


def price_market(dist, quotes: Sequence[Quote], market: str, *, weight: float,
                 bands: dict | None, open_quotes: Sequence[Quote] | None = None,
                 open_at: str | None = None) -> MarketView:
    vendor = M.vendor_key(market)
    view = MarketView(market=market, status="refused")
    c = consensus(quotes, vendor)
    if c is None:
        view.refusal = "no book quotes a complete market"
        return view
    anchor, other, line = c["anchor"], c["other"], c["line"]
    view.books = len(c["books"])
    pm_anchor = market_probability(c["books"], anchor, other)

    any_book = next(iter(c["books"].values()))
    try:
        if vendor == "totals":
            p_anchor, _ = Rc.total_probability(dist, float(line), over=True)
        else:
            home_line = 0.0 if line is None else float(line)
            if line is not None and not Rc._can_price_push(dist, home_line, total=False):
                raise Rc.PushPriceWithheld(
                    "an integer line and no key-number correction (ADR 0022)")
            p_anchor, _ = Rc.cover_probability(dist, home_line)
    except Rc.PushPriceWithheld as exc:
        view.refusal = f"line {line}: {exc}"
        return view
    except Exception as exc:                       # e.g. a withheld total
        view.refusal = f"{type(exc).__name__}: {str(exc)[:160]}"
        return view

    edge_anchor = p_anchor - pm_anchor
    side = anchor if edge_anchor >= 0 else other
    view.side = side
    view.p_model = p_anchor if side == anchor else 1 - p_anchor
    view.p_market = pm_anchor if side == anchor else 1 - pm_anchor
    view.edge = view.p_model - view.p_market
    sel = any_book[side]
    view.selection = sel.outcome
    view.line = sel.point
    best_book, best = max(((b, s[side]) for b, s in c["books"].items()),
                          key=lambda kv: kv[1].price_decimal)
    view.best_book, view.best_price_decimal = best_book, best.price_decimal
    view.best_price_american = round(P.decimal_to_american(best.price_decimal))
    view.tier = T.tier(view.edge, bands) if bands else None
    view.status = "priced"

    # The stake the operator command would size at the best price.
    try:
        cand = Rc.price_candidate(dist=dist, quotes=quotes, event_id=best.event_id,
                                  market=vendor, bookmaker=best_book,
                                  selection=best.outcome, bankroll=1.0,
                                  shrinkage=weight, kelly_multiple=KELLY,
                                  max_bankroll_fraction=MAX_FRACTION, min_edge=0.0)
        view.stake_fraction = float(cand.plan.bankroll_fraction) if cand.plan.placed else 0.0
    except Exception:
        view.stake_fraction = 0.0
    if weight <= 0:
        view.unsized_reason = "Unsized — not yet graded against the market"
    elif view.stake_fraction == 0:
        view.unsized_reason = "Graded, but no stake at this price"

    if open_quotes:
        o = consensus(open_quotes, vendor)
        if o is not None:
            view.open_at = open_at
            view.open_line = o["books"][next(iter(o["books"]))][side].point
            po = market_probability(o["books"], o["anchor"], o["other"])
            view.open_p_market = po if side == o["anchor"] else 1 - po
    return view
