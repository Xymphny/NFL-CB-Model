---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-calibration
---

# 0011. A withheld market must be refused by the object, not only by the list

## Context and Problem Statement

NFL, CFB and NBA all withhold totals from `primary_markets`. All three went on
answering `total_mean()`, `total_cdf()` and `total_pmf()` with a placeholder,
and `sample()` — which returns a scoreline, and therefore a total — worked
too.

A caller iterating `primary_markets` was safe. A caller asking the
**distribution** was not, and the distribution is what the seam is for: ADR
0001 made everything downstream import `core` and talk to a
`ScoreDistribution` rather than to a league. Withholding a market in a list
the object does not enforce is a half-connection, and half-connections are
what this rebuild exists to prevent.

## Decision Drivers

- The placeholders are not harmless. Two of the three are measurably wrong by
  about a third, and in both cases the contradicting number was already
  committed to this repository.
- A market that failed its gate should not become reachable by a longer route.
- Refusing must not break spread staking, which needs draws and never touches
  the total.

## Considered Options

- **Correct the constants.** Makes the numbers right and the markets look
  ready. NFL's totals model graded `supported: false`; a correct sd on a
  failed model is still a failed model.
- **Return NaN.** Propagates silently, which is the failure mode being fixed.
- **Refuse loudly.** Chosen.

## Decision Outcome

`UnvalidatedTotal` in `core.distributions`, and a `total_validated` flag on
`NormalMarginDistribution` defaulting to **True** so that a league which has
graded a total needs no ceremony. The three that have not now say so, and
every method depending on the total raises — including `sample()`, because a
scoreline encodes a total as surely as `total_mean()` does.

`sample_margin()` is the escape hatch. Correlated slate staking on spreads
needs draws and never touches the total dimension, so refusing the total must
not refuse that.

**The measurements, which are why this is not pedantry:**

| league | shipped `sd_total` | measured | ratio |
|---|---|---|---|
| CFB | 14.0 | 18.79 over 3,863 games | **1.342** |
| NFL | 10.0 | 13.353 (its own `totals_validation.json` RMSE) | **1.335** |
| NBA | 18.0 | not measurable here | — |

CFB's comparison is **exact** rather than indicative: its `mu_total` is a
constant, so the residual and unconditional dispersions are the same quantity
and no model is needed to compare them. Its nominal 95% interval covers
**87.0%**, the 80% covers 67.7%, and the 50% covers 40.7%.

NFL's number came from the very artifact that withheld the market. `data/
totals_validation.json` records a model RMSE of 13.353 beside a shipped sd of
10.0, and has done since the market was withheld.

**The labels were the wrong way round.** CFB's `TOTAL_MEAN_PLACEHOLDER = 52.0`
is accurate to seven hundredths of a point (bias −0.07, t = −0.23). The number
carrying the weaker label, `TOTAL_SD_UNVALIDATED`, is the dangerous one. A
label is not evidence in either direction.

**NBA is recorded as unmeasured, not as passing.** `sd_total` should be the
residual around predicted totals, and the feature source producing those
totals is not in this repository. The unconditional sd of 20.11 bounds a
residual only if the model has skill, which is the thing in question.

**Nothing was corrected.** A right sd on an ungraded mean is still an ungraded
total.

## Consequences

The conformance battery now accepts a documented refusal and rejects a silent
answer, and `tests/core/test_unvalidated_totals.py` asserts the invariant on
the real leagues: **the market list and the object must say the same thing.**
Offering a market the object refuses is a crash waiting for a live slate;
refusing one it could price is dead weight.

A caller that was quietly pricing NFL, CFB or NBA totals through the
distribution will now get an exception instead of a number. That is the
intended outcome and it is the reason this is an ADR rather than a commit.

## Recovery

Constructing any of the three distributions with `total_validated=True`
restores the previous behaviour immediately, and the invariant test then fails
until `"total"` is added to that league's `primary_markets` — which is the
correct order of operations once a total model has been graded.

## Revisit Triggers

- **Any league grades a total.** Set `total_validated=True` and add the market
  in the same change; the invariant test enforces that they move together.
- **A caller needs joint scorelines from one of these leagues.** There is no
  escape hatch for that on purpose. It means the total matters to them, and
  the total is not modelled.
- **NBA's feature source lands in this repository.** Its `sd_total` becomes
  measurable and stops being an unknown.
