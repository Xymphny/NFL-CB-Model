"""Where the pieces meet: a distribution, a market, and a stake.

This is the integration layer. Everything else in the project produces one
input to it -- a league produces a ScoreDistribution, the ingest layer
produces quotes, the evidence log produces a shrinkage weight, staking turns
the result into a size -- and none of that does anything until something joins
them into a recommendation.

FOUR RULES IT ENFORCES, EACH FROM SOMETHING ALREADY LEARNED HERE

1. VOIDING OUTCOMES ARE CONDITIONED OUT, NOT IGNORED. A spread bet has three
   outcomes and so does a moneyline -- the push on one, the tie on the other.
   The quantity comparable to a devigged two-way market price is
   P(cover | not push), and on a key number the push is worth 7-8% -- the
   measured NFL weights put P(margin = 3) near 0.078 against a rounded
   normal's 0.027. Comparing raw P(cover) to a market probability silently
   prices a two-way bet with three-way probabilities. The same applies to
   moneylines, which this module got wrong until MLB made it obvious: the
   negative binomial puts about 11% on an exact tie, and a game with a
   positive expected margin reported a sub-50% win probability because of it.

2. IT REFUSES TO PRICE A PUSH IT KNOWS IS WRONG. If the distribution reports
   no key-number correction and the line sits on an integer, the push
   probability is known to be wrong by roughly a factor of three. That is
   withheld rather than quoted.

3. EVERY CANDIDATE BECOMES A LEDGER ROW. Including the ones that do not clear
   the threshold, because the signals that never convert are what make
   execution cost measurable at all.

4. IT RECOMMENDS; IT DOES NOT BET. No network, no book API, no side effects
   beyond writing the ledger. A human places the bet and records the fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from coverline.core import pricing as P
from coverline.core import markets as M
from coverline.core.interfaces import ScoreDistribution
from coverline.core.staking import StakePlan, size_bet
from coverline.execution.ledger import BetLedger, Signal
from coverline.execution.normalize import Quote, two_sided


class PushPriceWithheld(ValueError):
    """The line sits on an integer and the distribution cannot price the push."""


@dataclass(frozen=True)
class Candidate:
    """One side of one market, priced against the model."""

    event_id: str
    market: str
    selection: str
    line: float | None
    bookmaker: str
    price_decimal: float

    p_model: float
    p_market: float
    push_probability: float

    plan: StakePlan

    @property
    def edge_claimed(self) -> float:
        return self.plan.edge_claimed

    @property
    def edge_used(self) -> float:
        return self.plan.edge_used


def cover_probability(dist: ScoreDistribution, line: float) -> tuple[float, float]:
    """(P(cover | not push), P(push)) for a home spread of `line`.

    `line` is the home margin the bet needs beaten: -3 means the home team
    must win by more than 3. Sign follows the model's margin convention, so a
    home favourite has a negative line, as books quote it.
    """
    threshold = -line
    p_push = dist.margin_pmf(threshold) if dist.is_discrete else 0.0
    p_cover = 1.0 - dist.margin_cdf(threshold)
    denom = 1.0 - p_push
    if denom <= 0:
        raise ValueError("push probability of 1 leaves nothing to bet on")
    return p_cover / denom, p_push


def total_probability(dist: ScoreDistribution, line: float,
                      over: bool) -> tuple[float, float]:
    """(P(this side | not push), P(push)) for a total of `line`.

    A SEPARATE FUNCTION BECAUSE A TOTAL IS NOT A MARGIN, which is not as
    obvious as it sounds: until this existed, price_candidate routed every
    market through cover_probability, so an Over 44.5 was priced as
    P(margin > -44.5) and came back 0.9982 against a truth of 0.4801. MLB is
    the only league that currently offers totals, and it would have bet every
    Over at maximum stake.
    """
    p_push = dist.total_pmf(line) if dist.is_discrete else 0.0
    p_over = 1.0 - dist.total_cdf(line)
    denom = 1.0 - p_push
    if denom <= 0:
        raise ValueError("push probability of 1 leaves nothing to bet on")
    p = p_over if over else 1.0 - p_over - p_push
    return p / denom, p_push


def _is_home_outcome(q: Quote) -> bool | None:
    """True for the home side, False for the away side, None if neither.

    Compared on the team names the quote itself carries, so a league that
    renames a team cannot silently swap a side.
    """
    name = q.outcome.strip().casefold()
    if name == (q.home_team or "").strip().casefold():
        return True
    if name == (q.away_team or "").strip().casefold():
        return False
    return None


def _can_price_push(dist: ScoreDistribution, line: float,
                    total: bool = False) -> bool:
    """A distribution without measured key numbers cannot price an integer line.

    Not a style preference: the plain rounded normal understates P(margin=3)
    by nearly threefold, so the conditioning above would be materially wrong
    in a direction that inflates the apparent edge.

    A TOTAL's line is already in the distribution's own units, so it is
    tested as it stands rather than negated -- negating it, as this function
    did for every market, made an integer total of 44 look like -44 and
    answered the wrong question about the wrong quantity.
    """
    if not float(line if total else -line).is_integer():
        return True
    if not dist.is_discrete:
        return True
    # A family whose pmf IS the atom needs no key-number table. Only a
    # rounded continuous distribution does, and conflating the two withheld
    # every integer total in baseball and hockey.
    if bool(getattr(dist, "pmf_is_exact", False)):
        return True
    return bool(getattr(dist, "has_key_number_correction", False))


def price_candidate(
    *,
    dist: ScoreDistribution,
    quotes: Sequence[Quote],
    event_id: str,
    market: str,
    bookmaker: str,
    selection: str,
    bankroll: float,
    shrinkage: float,
    devig_method: P.Method = "power",
    **staking,
) -> Candidate:
    """Price one side against one book's full two-sided market."""
    sides = two_sided(quotes, event_id=event_id, market=market, bookmaker=bookmaker)
    if not sides:
        raise ValueError(
            f"{bookmaker} has no complete two-sided {market} market for "
            f"{event_id}; devigging one side is not possible"
        )
    idx = next((i for i, q in enumerate(sides) if q.outcome == selection), None)
    if idx is None:
        raise ValueError(f"{selection!r} is not an outcome of that market")

    quote = sides[idx]
    line = quote.point
    is_total = quote.outcome.strip().lower() in ("over", "under")

    if line is not None and not _can_price_push(dist, line, total=is_total):
        raise PushPriceWithheld(
            f"{event_id} {market} sits on the integer line {line} and this "
            "distribution has no measured key-number correction. Its push "
            "probability is wrong by roughly threefold, so the edge would be "
            "too, in the flattering direction."
        )

    if is_total:
        if line is None:
            raise ValueError(f"{event_id} {market} Over/Under carries no line")
        p_model, p_push = total_probability(
            dist, line, over=quote.outcome.strip().lower() == "over")
    else:
        # WHICH SIDE IS THIS? Until this branch existed, the away side was
        # priced by feeding ITS line into a function that always answers for
        # the HOME team. On a home favourite at -6.5 that returned 0.5314 for
        # the away side against a truth of 0.8214, and the two sides of a
        # no-push market summed to 1.08 instead of 1. Every away-side
        # handicap and moneyline price in every league was wrong, in a
        # direction that flips with the sign of the line.
        home = _is_home_outcome(quote)
        if home is None:
            raise ValueError(
                f"{selection!r} matches neither {quote.home_team!r} nor "
                f"{quote.away_team!r}; the side cannot be determined and "
                "guessing it is how the away side was wrong before"
            )
        # The quoted point is that side's line. Convert to a HOME line, which
        # is the only convention cover_probability speaks.
        home_line = (line if line is not None else 0.0) if home else -(
            line if line is not None else 0.0)
        p_home, p_push = cover_probability(dist, home_line)
        # A moneyline is a spread of zero. Routing it through the same
        # function is not tidiness -- it is the fix for a real bug. It used
        # to special-case moneylines as (1 - cdf(0), 0.0), conditioning
        # nothing out. On a football spread the push is the obvious voiding
        # outcome; on a moneyline it is the TIE, and just as voiding. MLB
        # exposed it: a game with a +0.42 expected margin reported
        # P(home) = 0.49 because 10.8% of the mass sat on margin = 0.
        # Conditioned, the same game is 0.5495 -- a 5.9 point error.
        p_model = p_home if home else 1.0 - p_home
    p_market = float(P.devig([q.price_decimal for q in sides], devig_method)[idx])

    plan = size_bet(p_model=p_model, p_market=p_market,
                    decimal_price=sides[idx].price_decimal,
                    bankroll=bankroll, shrinkage=shrinkage, **staking)

    return Candidate(event_id=event_id, market=market, selection=selection,
                     line=line, bookmaker=bookmaker,
                     price_decimal=sides[idx].price_decimal,
                     p_model=p_model, p_market=p_market, push_probability=p_push,
                     plan=plan)


def to_signal(c: Candidate, *, league: str, bankroll: float,
              best_available: Quote | None = None,
              books_surveyed: int | None = None,
              at: str | None = None) -> Signal:
    """Turn a priced candidate into a ledger row, placed or not.

    A candidate the staking engine declined is recorded as `below_threshold`
    rather than dropped. Dropping it is what makes execution cost
    unmeasurable.
    """
    at = at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sid = f"{c.event_id}|{c.market}|{c.selection}|{c.bookmaker}|{at}"
    common = dict(
        signal_id=sid, at=at, league=league, event_id=c.event_id,
        market=c.market, selection=c.selection, line=c.line,
        p_model=c.p_model, p_market=c.p_market, p_used=c.plan.p_used,
        shrinkage=(c.plan.p_used - c.p_market) / (c.p_model - c.p_market)
        if c.p_model != c.p_market else 0.0,
        edge_claimed=c.edge_claimed, edge_used=c.edge_used,
        # Carried through rather than dropped. It was computed, it decided
        # the price, and a ledger row without it cannot be re-checked for the
        # conditioning error that motivated computing it.
        push_probability=c.push_probability,
    )
    if not c.plan.placed:
        return Signal(**common, disposition="not_placed",
                      not_placed_reason="below_threshold", notes=c.plan.reason)
    return Signal(
        **common, disposition="placed", book=c.bookmaker,
        price_decimal=c.price_decimal, stake=c.plan.stake,
        bankroll_at_placement=bankroll,
        best_available_decimal=(best_available.price_decimal
                                if best_available else None),
        books_surveyed=books_surveyed, notes=c.plan.reason,
    )


def recommend(
    *,
    dist: ScoreDistribution,
    quotes: Sequence[Quote],
    event_id: str,
    market: str,
    league: str,
    bankroll: float,
    shrinkage: float,
    ledger: BetLedger | None = None,
    primary_markets: Sequence[str] | None = None,
    **staking,
) -> list[Signal]:
    """Price every book and side for one market, and record every candidate.

    Returns the signals. Writing them is optional so a caller can inspect a
    slate before committing anything to an append-only ledger.

    `primary_markets` IS THE LEAGUE'S OWN LIST, and passing it turns that list
    from documentation into a control. Until this parameter existed,
    LeagueModel.primary_markets had no production consumer at all -- only
    tests read it -- so a caller could price any league on any market and get
    a signal back. See core/markets.py.

    It is optional because making it required would break every existing call
    site in one commit, which is how a safety feature gets reverted. A call
    that omits it behaves exactly as before and is not protected, which is
    stated here rather than left to be discovered.
    """
    if primary_markets is not None:
        M.require_offered(market, primary_markets, league)
    vendor = M.vendor_key(market)
    books = sorted({q.bookmaker for q in quotes
                    if q.event_id == event_id and q.market == vendor})
    out: list[Signal] = []

    for book in books:
        sides = two_sided(quotes, event_id=event_id, market=vendor, bookmaker=book)
        for q in sides:
            try:
                c = price_candidate(
                    dist=dist, quotes=quotes, event_id=event_id, market=vendor,
                    bookmaker=book, selection=q.outcome, bankroll=bankroll,
                    shrinkage=shrinkage, **staking)
            except PushPriceWithheld:
                continue     # withheld, not a recommendation of any kind
            best = max((x for x in quotes
                        if x.event_id == event_id and x.market == vendor
                        and x.outcome == q.outcome),
                       key=lambda x: x.price_decimal, default=None)
            out.append(to_signal(c, league=league, bankroll=bankroll,
                                 best_available=best, books_surveyed=len(books)))

    if ledger is not None:
        for s in out:
            ledger.record(s)
    return out
