---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: core-distributions
---

# 0001. Rebuild around a single distribution seam

## Context and Problem Statement

Three leagues were implemented largely independently, and a fourth and fifth
(NHL, NBA) are being added. Every league had its own version of the things that
touch money -- turning a model number into a price, comparing it to a market,
deciding a stake -- so a fix in one never reached the others. Two audits found
the same thing from different directions: the execution layer, not the model,
is what binds realised profit, and it barely existed.

Adding two more leagues to that structure multiplies the divergence by five.

## Considered Options

- **Keep per-league pipelines, add NHL and NBA in the same style.** Cheapest
  today. Guarantees five copies of the pricing logic.
- **Share code where convenient, by extracting helpers as duplication is
  noticed.** What was already being done. Produces helpers shaped around
  whichever league needed them first.
- **Define one interface every league must satisfy, and put everything
  downstream of it in shared code.** More work up front, and forces the
  low-count sports (NHL, MLB) and the near-normal ones (NFL, CFB, NBA) to agree
  on a common vocabulary.

## Decision Outcome

Chosen: the single interface. Every league produces a `ScoreDistribution`;
everything downstream of it -- devigging, expected value, CLV, Kelly, bankroll,
calibration, grading -- imports only `core` and never a league package.

Enforcement is an annotated `REGISTRY: dict[str, LeagueModel]` checked by
`mypy --strict`, not discipline. `@runtime_checkable` + `isinstance` was
explicitly rejected as the mechanism: the typing documentation states it checks
only member presence, not signatures, so a league with a wrongly-typed
`predict` would pass it and fail in production.

The interface carries `is_discrete` and `*_pmf` even though general forecasting
libraries (GluonTS, skpro) do not. Four of the five leagues price on integer
lines, so the probability mass sitting exactly on a line is a first-class
quantity. An interface that leaves atom mass implicit cannot price a push, and
pushes are most of what distinguishes a -3 from a -2.5.

## Consequences

Better: one implementation of everything that touches money, so five leagues
cost less to run than three did. A sixth league is a package plus a registry
entry, with no pricing code written.

Worse: the two distribution families have to agree on a vocabulary that fits
neither perfectly. `BivariatePoissonDistribution` exposes a `margin_sd` that is
an honest summary but not how anyone reasons about hockey. Accepted.

Harder to undo: once pricing, staking and grading all import `core`, changing
the interface is a five-league change. That is the intended cost -- it is what
makes divergence expensive instead of free.

## Recovery

Not applicable. Nothing was dropped; the legacy pipeline is untouched and still
runs the board. The new core is additive and can be deleted wholesale
(`src/coverline/`, `tests/core/`) with no effect on published output.

## Revisit Triggers

- If a sixth sport genuinely fits neither distribution family and needs a third
  implementation, check whether the interface is carrying its weight or just
  imposing a vocabulary.
- If `ScoreDistribution` grows past roughly a dozen methods, it has stopped
  being a seam and become a base class, and the split should be reconsidered.
- If any league ends up needing to import another league, the dependency arrow
  has broken and this decision is not being honoured.
