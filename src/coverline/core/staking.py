"""Bet sizing, and the shrinkage that used to live in the ship/no-ship gate.

THE STRUCTURAL CHANGE THIS MODULE ENCODES
-----------------------------------------
The old design put conservatism in the wrong place. A significance gate decided
whether a model change shipped at all, and at this project's sample sizes that
gate had essentially no power -- the minimum detectable effect on ~2,000 NFL
games is about +3.1 points of cover rate, an entire career's edge from one
change, so a candidate worth a real +1 point cleared t > 1.96 roughly 14% of
the time. A rule that fires on one in seven genuine improvements is not strict;
it is off. Worse, conditional on passing, the measured effect is inflated
roughly 2.7x, so the few changes that did ship were then overbet.

Under Kelly the loss function is mean squared error, not a hypothesis test:
expected growth shortfall is E[(mu_hat - mu)^2] / (2 sigma^2). That makes this
an estimation problem, and the growth-optimal estimator of an uncertain edge is
a shrunk one, not a thresholded one. Hard thresholding is dominated.

So: candidates are no longer withheld or shipped. They are shipped at a WEIGHT,
and the weight is computed, not chosen -- see ``shrinkage_weight``. Withholding
still exists, but it now means "this quantity is not measurable at all", not
"this quantity failed a test".

WHERE THE WEIGHT COMES FROM
---------------------------
``core.evidence`` holds the attempt log and computes the weight from it. As of
2026-09-21 that log has 5 recovered attempts, RMS t = 1.580, and produces
b = 0.5997 -- so candidates ship at roughly 60% weight rather than at 0 or 1.

``DEFAULT_SHRINKAGE`` is still deliberately absent, and should stay absent. A
module constant would be importable without reading the log, and the discipline
is that the weight is derived, dated and auditable rather than chosen. Callers
pass ``evidence.current_weight()`` explicitly, which fails loudly if the log is
missing instead of falling back to a guess.

Two properties of that weight matter at the call site. It is a CEILING while
the log contains recovered attempts, because a reconstructed log is missing
whatever was never written up and what goes unwritten skews toward nulls. And
it can legitimately be 0.0: an RMS t near 1 is what pure noise produces, and
the correct response is to bet nothing, which ``size_bet`` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ------------------------------------------------------------- shrinkage ----

def shrinkage_weight(t_stats: Sequence[float]) -> float:
    """Empirical-Bayes weight b = 1 - 1/E[t^2], from every candidate tested.

    Pass the t-statistic of EVERY model change evaluated, including the ones
    that failed. Passing only the winners inflates E[t^2] and returns a weight
    that ships noise at full size -- the selection effect this is meant to undo.

    The return value doubles as a diagnostic on the whole research pipeline.
    If the root-mean-square t across all attempts is near 1.0, then E[t^2] is
    near 1.0, b is near 0, and the arithmetic is saying that nothing tested so
    far is distinguishable from noise. That is information about the pipeline,
    not a number to tune away.
    """
    if len(t_stats) == 0:
        raise ValueError(
            "shrinkage_weight needs the t-statistics of every candidate tested, "
            "including failures; an empty log cannot produce a weight"
        )
    mean_t2 = float(np.mean(np.asarray(t_stats, dtype=float) ** 2))
    if mean_t2 <= 1.0:
        # b would be <= 0: the candidates collectively carry no signal.
        return 0.0
    return 1.0 - 1.0 / mean_t2


def shrink_probability(p_model: float, p_market: float, weight: float) -> float:
    """Move the model's probability toward the devigged market prior.

    Shrink the PROBABILITY, not the stake. The two are not equivalent: halving
    a stake scales exposure linearly while leaving the claimed edge intact in
    every downstream diagnostic, so a log full of half-Kelly bets still reports
    the unshrunk edge and still looks calibrated-but-unlucky when it loses.
    Shrinking the probability propagates into EV, CLV and calibration together,
    which is what makes the log honest about what was actually believed.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    for name, p in (("p_model", p_model), ("p_market", p_market)):
        if not 0.0 < p < 1.0:
            raise ValueError(f"{name} must lie strictly between 0 and 1")
    return weight * p_model + (1.0 - weight) * p_market


# ----------------------------------------------------------------- kelly ----

def kelly_fraction(p: float, decimal_price: float) -> float:
    """Full-Kelly stake as a fraction of bankroll. Negative means do not bet.

    f* = (p*b - q) / b, with b = decimal - 1.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if decimal_price <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    b = decimal_price - 1.0
    return (p * b - (1.0 - p)) / b


def growth_rate(f: float, p: float, decimal_price: float) -> float:
    """Expected log growth per bet at stake fraction f. The thing Kelly maximises.

    Used by the tests to verify the overbetting asymmetry numerically rather
    than asserting it from a remembered formula.
    """
    if f < 0.0 or f >= 1.0:
        raise ValueError("f must lie in [0, 1)")
    b = decimal_price - 1.0
    return p * math.log(1.0 + f * b) + (1.0 - p) * math.log(1.0 - f)


@dataclass(frozen=True)
class StakePlan:
    """What to bet and, more importantly, the reasoning that produced it."""

    stake: float
    """Currency amount. Zero when the bet should not be placed."""

    bankroll_fraction: float
    p_model: float
    p_used: float
    """The shrunk probability actually staked on. Log THIS, not p_model."""

    p_market: float
    edge_claimed: float
    """EV under the raw model probability. What the model thinks."""

    edge_used: float
    """EV under the shrunk probability. What was actually bet on."""

    full_kelly_fraction: float
    kelly_multiple: float
    """Fraction of full Kelly actually staked. Above 1.0 is overbetting."""

    reason: str
    """Why the stake is what it is -- especially why it is zero."""

    @property
    def placed(self) -> bool:
        return self.stake > 0.0


def size_bet(
    *,
    p_model: float,
    p_market: float,
    decimal_price: float,
    bankroll: float,
    shrinkage: float,
    kelly_multiple: float = 0.25,
    max_bankroll_fraction: float = 0.02,
    min_edge: float = 0.0,
) -> StakePlan:
    """The four stages, in the order that matters.

    1. shrink the probability toward the market prior
    2. compute full Kelly on the SHRUNK probability
    3. apply the fractional-Kelly multiple
    4. apply a hard bankroll cap

    Stage 1 before stage 2 is the load-bearing ordering, and the two orderings
    are not equivalent. Which is larger depends on the price, crossing at even
    money: below decimal 2.0 shrinking first is the more conservative, above it
    the more aggressive. That matters because virtually every bet this project
    places is priced below 2.0, where shrinking first can correctly VETO a bet
    that stake-scaling would still place. Worked example at decimal 1.8, a 60%
    model probability, a 50% market and weight 0.40: shrink-first returns a
    Kelly fraction of -0.035, i.e. no bet, while Kelly-then-scale returns
    +0.040 and stakes it.

    The second reason is bookkeeping. Scaling the stake leaves p_model intact
    in every downstream diagnostic, so EV, CLV and calibration all describe a
    bet that was never placed.

    ``kelly_multiple`` defaults to 0.25 rather than 0.5. Quarter Kelly retains
    about 44% of the growth rate of full Kelly, and the asymmetry is severe in
    the other direction: staking twice the true Kelly fraction drives expected
    growth to approximately zero even though every individual bet is +EV. Since
    the true edge here is an estimate with real error, the multiple is
    protecting against being wrong about f*, not merely smoothing variance.
    """
    if bankroll <= 0.0:
        raise ValueError("bankroll must be positive")
    if not 0.0 < kelly_multiple <= 1.0:
        raise ValueError("kelly_multiple must lie in (0, 1]")
    if not 0.0 < max_bankroll_fraction <= 1.0:
        raise ValueError("max_bankroll_fraction must lie in (0, 1]")

    p_used = shrink_probability(p_model, p_market, shrinkage)
    edge_claimed = p_model * decimal_price - 1.0
    edge_used = p_used * decimal_price - 1.0

    full = kelly_fraction(p_used, decimal_price)

    def plan(frac: float, reason: str) -> StakePlan:
        return StakePlan(
            stake=round(max(0.0, frac) * bankroll, 2),
            bankroll_fraction=max(0.0, frac),
            p_model=p_model,
            p_used=p_used,
            p_market=p_market,
            edge_claimed=edge_claimed,
            edge_used=edge_used,
            full_kelly_fraction=full,
            kelly_multiple=(frac / full) if full > 0 else 0.0,
            reason=reason,
        )

    if full <= 0.0:
        return plan(0.0, "no edge after shrinking toward the market")
    if edge_used < min_edge:
        return plan(0.0, f"shrunk edge {edge_used:.4f} below floor {min_edge:.4f}")

    frac = full * kelly_multiple
    if frac > max_bankroll_fraction:
        return plan(
            max_bankroll_fraction,
            f"capped at {max_bankroll_fraction:.1%} of bankroll "
            f"(fractional Kelly wanted {frac:.2%})",
        )
    return plan(frac, f"{kelly_multiple:.2f} Kelly on a shrunk edge of {edge_used:.2%}")
