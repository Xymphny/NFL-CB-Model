---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: execution-grade
---

# 0016. Average CLV in probability points, never in American cents

## Context and Problem Statement

ADR 0002 settled *which* close CLV is measured against: the devigged fair
close, never the posted one, because the margin cannot be removed from one
side alone. It did not settle *what scale* the resulting numbers are averaged
on, and `ClosingLineValue` reports three of them.

One of those three cannot be averaged at all.

## Decision Drivers

- CLV is the metric this project's whole snapshot architecture exists to
  produce, and a mean of it is the headline a bankroll gets judged by.
- American odds are the scale a bettor actually reads on a screen, so
  dropping them is not an option either.

## Considered Options

- **Report the cents mean with a caveat.** A caveat beside a number that
  looks like a CLV is not a control.
- **Drop `price_cents`.** Loses the figure a bettor can reconcile against
  their own account.
- **Keep it, refuse to average it, and add a scale that can be.** Chosen.

## Decision Outcome

`ClosingLineValue` gains **`prob_points`**: the devigged fair probability minus
the bet's implied probability. Linear, continuous, and comparable across the
whole price range. `clv_summary` reports its mean as
`mean_clv_probability_points` and reports **no mean in cents**, with the
reason stated in the payload rather than only in a docstring.

**American odds are discontinuous at even money** — nothing exists between
−100 and +100 — and non-linear everywhere else. Measured on real price pairs:

| move | probability points | American cents |
|---|---|---|
| 2.050 → 1.952 | 2.45 | **210.0** |
| 1.200 → 1.180 | 1.41 | 55.6 |
| 1.910 → 1.870 | 1.12 | 5.1 |

That is **19 times** as many cents per probability point at the top of the
table as at the bottom. And the ranking genuinely inverts: a deep favourite
moving 1.05 → 1.04 covers 0.91 probability points and reads as 500 cents,
more than the 2.45-point move above it.

A mean over such numbers is dominated by whichever bets happened to sit near
even money. It would look like a CLV, carry a sign, and mean nothing — and
moneylines near even money are exactly where NHL and MLB bets cluster.

## Consequences

**The legacy board is unaffected.** `deploy/generate_performance.py` computes
`avg_clv` as `close - line`, which is line points — linear, and correctly
labelled "pts" in the frontend. The defect was confined to the new core's
`price_cents`, which is where the fix is.

**Two properties now pin the devigging itself**, which nothing had: taking
exactly the posted closing price must show negative EV, because that price
carries the hold, and a CLV implementation that forgot to devig would report
exactly zero there. Betting the fair close is exactly break-even on both
scales.

## Recovery

`prob_points` is additive; removing it and restoring the cents mean in
`clv_summary` returns the previous behaviour, which reports a number that
cannot be interpreted.

## Revisit Triggers

- **`test_american_cents_are_not_comparable_and_probability_points_are`
  fails.** The cents-per-point ratio has stopped varying by an order of
  magnitude, which would mean the price distribution has narrowed enough that
  the warning no longer bites.
- **A consumer starts averaging `price_cents`.** The field is kept for
  reconciliation against a bettor's own account and for nothing else.
- **Real graded bets accumulate.** Both scales can then be compared against
  realised profit, which is the only test that settles which one predicts it.
