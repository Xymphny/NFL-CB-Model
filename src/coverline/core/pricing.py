"""Odds conversion, margin removal, expected value, and closing line value.

WHY THIS MODULE IS THE ONE THAT MATTERS
---------------------------------------
The strategy audit found that closing line value is the right way to evaluate
this project -- roughly a hundredfold variance reduction against win/loss
records, which is the difference between validating a change in one season and
never validating it -- but ONLY when it is measured against a devigged fair
close. Measured against the posted close, CLV includes the bookmaker's margin
and systematically overstates edge. A published worked example shows the same
bet scoring +4.01% naive and +0.76% devigged: a five-fold overstatement, and
the entire apparent edge.

That is why four methods live here instead of one. They disagree by over a
percentage point on a moderately lopsided two-way market, which is a large
fraction of any edge this project will ever have, so the CHOICE of method is
not a detail -- and a caller who cannot see the disagreement cannot know when
it matters. ``devig_all`` returns every method plus the spread between them,
and ``method_spread_exceeds`` is the guardrail: when the methods disagree by
more than the edge being claimed, the edge is a statement about devigging
rather than about the game.

WHAT IS UNVERIFIED AND TREATED AS SUCH
--------------------------------------
Shin's method is implemented by solving for z numerically rather than by a
closed form. Published closed forms for the two-outcome case circulate but
could not be checked against a primary source, and a closed form that is subtly
wrong is worse than a solver that is slow: the solver's answer can be verified
by substitution, which ``_shin_residual`` does and the tests assert on.

TWO SPECIFIED FIGURES WERE WRONG, AND THE CODE WAS NOT BENT TO MATCH THEM
-------------------------------------------------------------------------
The research that specified this module quoted four fair probabilities for a
-300/+240 market: 71.84 multiplicative, 72.47 power, 72.80 additive, 73.05
shin. Multiplicative and additive reproduce exactly. Power and shin do not, and
the implementations here are right:

  * power is DEFINED by sum(p_i^k) = 1. The solver finds k = 1.0793, at which
    sum(p^k) = 1.0 to twelve decimals and p_fav = 73.31. No k produces 72.47
    while satisfying the constraint, so 72.47 is not a power-method answer.
  * shin is DEFINED by one z satisfying pi_i^2/S = (1-z)p_i^2 + z*p_i for every
    outcome at once. This implementation recovers a single consistent z to
    machine precision on 2-, 3- and 4-way markets, giving 72.79, not 73.05.

Checking against the defining equation rather than the quoted number is what
surfaced this. tests/core/test_pricing.py asserts the equations, deliberately
not the figures.

A SOURCE CONFLICT, NOW RESOLVED BY MEASUREMENT
----------------------------------------------
One practitioner source states Shin reduces to the additive method for
two-outcome markets; another's worked example contradicts it. The first is
right. Shin and additive agree to 1e-16 on every two-way market tested and
diverge on three-way markets, which is what a two-outcome algebraic identity
looks like. Practical consequence: on two-way markets there are three distinct
methods here, not four, and reporting shin and additive as independent
corroboration of each other would be double-counting one answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import optimize

Method = Literal["multiplicative", "additive", "power", "shin"]
METHODS: tuple[Method, ...] = ("multiplicative", "additive", "power", "shin")


# ------------------------------------------------------------ conversion ----

def american_to_decimal(american: float) -> float:
    """-110 -> 1.9091, +240 -> 3.40."""
    if american == 0:
        raise ValueError("american odds of 0 are not a price")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def decimal_to_implied(decimal: float) -> float:
    """Raw implied probability, margin INCLUDED. Not a fair probability."""
    if decimal <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    return 1.0 / decimal


def american_to_implied(american: float) -> float:
    return decimal_to_implied(american_to_decimal(american))


def booksum(decimals: Sequence[float]) -> float:
    """Sum of raw implied probabilities. Above 1.0 by the bookmaker's margin."""
    return float(sum(decimal_to_implied(d) for d in decimals))


def hold(decimals: Sequence[float]) -> float:
    """The bookmaker's theoretical hold as a fraction of handle.

    Note this is NOT booksum - 1. A book with a 4.76% overround holds 4.55% of
    handle, and quoting the overround as the hold overstates the cost by about
    5% of itself.
    """
    s = booksum(decimals)
    return (s - 1.0) / s


# ---------------------------------------------------------------- devig ----

def devig_multiplicative(decimals: Sequence[float]) -> np.ndarray:
    """Scale every implied probability by the same factor. Simplest, and the
    one that assumes margin is proportional to probability -- which biases the
    favourite's fair price low relative to the other methods."""
    raw = np.array([decimal_to_implied(d) for d in decimals], dtype=float)
    return raw / raw.sum()


def devig_additive(decimals: Sequence[float]) -> np.ndarray:
    """Subtract the excess equally from each outcome. Assumes margin is
    constant in probability terms, which biases the opposite way to
    multiplicative and can produce negative probabilities on extreme
    longshots -- guarded below."""
    raw = np.array([decimal_to_implied(d) for d in decimals], dtype=float)
    fair = raw - (raw.sum() - 1.0) / len(raw)
    if np.any(fair <= 0.0):
        raise ValueError(
            "additive devig produced a non-positive probability; the market is "
            "too lopsided for this method -- use power or shin"
        )
    return fair


def devig_power(decimals: Sequence[float]) -> np.ndarray:
    """Raise each implied probability to a common power k with sum(p^k) = 1.

    Also called the logarithmic method. k > 1 when the book is overround, which
    shrinks longshots more than favourites -- the behaviour usually wanted,
    because the favourite-longshot bias is in that direction.
    """
    raw = np.array([decimal_to_implied(d) for d in decimals], dtype=float)

    def residual(k: float) -> float:
        return float(np.sum(raw**k) - 1.0)

    lo, hi = 0.05, 20.0
    if residual(lo) * residual(hi) > 0:
        raise ValueError("power devig failed to bracket a solution")
    k = optimize.brentq(residual, lo, hi, xtol=1e-14, rtol=1e-15)
    return raw**k


def _shin_fair(raw: np.ndarray, z: float) -> np.ndarray:
    s = raw.sum()
    inner = z * z + 4.0 * (1.0 - z) * (raw * raw) / s
    return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))


def _shin_residual(raw: np.ndarray, z: float) -> float:
    return float(_shin_fair(raw, z).sum() - 1.0)


def devig_shin(decimals: Sequence[float]) -> np.ndarray:
    """Shin's method: margin arises from the book protecting against a fraction
    z of insider money. Solved numerically for z; see module docstring for why
    no closed form is used.

    This is the defensible default for markets with three or more outcomes,
    where multiplicative and additive both behave badly.
    """
    raw = np.array([decimal_to_implied(d) for d in decimals], dtype=float)
    if raw.sum() <= 1.0:
        # No margin to remove (or an arb). Shin's z is undefined; fall back to
        # normalising, and say so rather than returning something invented.
        return raw / raw.sum()

    lo, hi = 1e-12, 1.0 - 1e-9
    f_lo, f_hi = _shin_residual(raw, lo), _shin_residual(raw, hi)
    if f_lo * f_hi > 0:
        raise ValueError("shin devig failed to bracket a solution for z")
    z = optimize.brentq(lambda zz: _shin_residual(raw, zz), lo, hi,
                        xtol=1e-15, rtol=1e-15)
    fair = _shin_fair(raw, z)
    # Verify by substitution rather than trusting the solve.
    if abs(fair.sum() - 1.0) > 1e-9:
        raise ValueError(f"shin solution did not normalise: sum={fair.sum()!r}")
    return fair


_DISPATCH = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(decimals: Sequence[float], method: Method = "power") -> np.ndarray:
    """Remove the bookmaker margin. Default is power: the most-recommended
    two-way choice, and better behaved than additive on lopsided markets."""
    if method not in _DISPATCH:
        raise ValueError(f"unknown devig method {method!r}; have {list(_DISPATCH)}")
    if len(decimals) < 2:
        raise ValueError("devigging needs at least two outcomes")
    return _DISPATCH[method](decimals)


@dataclass(frozen=True)
class DevigComparison:
    """Every method's answer for one outcome, plus how much they disagree."""

    fair_by_method: dict[str, float]
    spread: float
    """Max minus min fair probability across methods, in probability points."""

    @property
    def consensus(self) -> float:
        """Mean across methods that succeeded. Not a recommendation -- a
        summary. Prefer naming a method explicitly."""
        return float(np.mean(list(self.fair_by_method.values())))

    def exceeds(self, claimed_edge: float) -> bool:
        """True when method disagreement is as large as the edge being claimed.

        The guardrail. If flipping devig method moves the fair price as much as
        the edge does, the edge is a statement about devigging, not the game.
        """
        return self.spread >= abs(claimed_edge)


def devig_all(decimals: Sequence[float], index: int = 0) -> DevigComparison:
    """Devig by every method and report the outcome at `index` under each.

    Methods that legitimately cannot run (additive on a lopsided market) are
    omitted rather than substituted, so the spread is computed over real
    answers only.
    """
    out: dict[str, float] = {}
    for name in METHODS:
        try:
            out[name] = float(devig(decimals, name)[index])
        except ValueError:
            continue
    if not out:
        raise ValueError("no devig method succeeded on this market")
    values = list(out.values())
    return DevigComparison(fair_by_method=out, spread=float(max(values) - min(values)))


# ------------------------------------------------------- value and CLV ----

def expected_value(fair_prob: float, decimal_price: float) -> float:
    """EV per unit staked. 0.03 means a 3% edge.

    This is also exactly the ratio form of CLV, (p_fair - p_bet) / p_bet, so
    edge and CLV share an axis and can be plotted against realised ROI
    together.
    """
    if not 0.0 < fair_prob < 1.0:
        raise ValueError("fair_prob must lie strictly between 0 and 1")
    return fair_prob * decimal_price - 1.0


def kelly_edge(fair_prob: float, decimal_price: float) -> float:
    """The b*p - q form. Identical to expected_value; named separately because
    staking code reads better when it says what it means."""
    return expected_value(fair_prob, decimal_price)


@dataclass(frozen=True)
class ClosingLineValue:
    """Three numbers, because they diagnose different capabilities.

    Beating the NUMBER (getting +3.5 when the close was +3) is a timing and
    market-reading skill. Beating the PRICE (getting -105 when the close was
    -110) is a shopping and execution skill. A bettor can be good at one and
    bad at the other, and a single blended figure hides which.
    """

    line_points: float | None
    """Points of line beaten. None for markets without a line, e.g. moneylines."""

    price_cents: float
    """American-odds cents beaten at the same line."""

    ev_pct: float
    """EV against the DEVIGGED fair close. The one that predicts profit."""

    naive_ev_pct: float
    """EV against the posted close, margin included. Kept only to show the gap."""

    valid: bool
    """False where no sharp closing line exists -- props above all.

    Prop markets are not made by market-making books, so their closing prices
    are not a sharp forecast and CLV against them is, in a practitioner's
    words, at best an informed guess. Recording CLV as invalid is not the same
    as recording it as zero, and code that averages CLV must filter on this.
    """

    @property
    def devig_overstatement(self) -> float:
        """How much the naive figure overstates. The size of the mistake."""
        return self.naive_ev_pct - self.ev_pct


def closing_line_value(
    *,
    bet_decimal: float,
    close_decimals: Sequence[float],
    outcome_index: int = 0,
    bet_line: float | None = None,
    close_line: float | None = None,
    method: Method = "power",
    market_has_sharp_close: bool = True,
) -> ClosingLineValue:
    """Compute CLV for one settled or pending bet.

    `close_decimals` is the FULL two-sided (or n-way) closing market, because
    the margin cannot be removed from one side alone. Passing a single price
    here is the most common way to get CLV wrong.
    """
    fair = float(devig(close_decimals, method)[outcome_index])
    naive = float(decimal_to_implied(close_decimals[outcome_index]))

    line_points = None
    if bet_line is not None and close_line is not None:
        line_points = float(bet_line - close_line)

    close_american = decimal_to_american(close_decimals[outcome_index])
    bet_american = decimal_to_american(bet_decimal)

    return ClosingLineValue(
        line_points=line_points,
        price_cents=float(bet_american - close_american),
        ev_pct=expected_value(fair, bet_decimal),
        naive_ev_pct=naive * bet_decimal - 1.0,
        valid=market_has_sharp_close,
    )
