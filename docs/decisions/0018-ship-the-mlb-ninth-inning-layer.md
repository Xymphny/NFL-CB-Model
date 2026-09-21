---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-mlb
---

# 0018. Ship the MLB ninth-inning layer, and reopen the moneyline on the gate that closed it

## Context and Problem Statement

ADR 0017 withheld the MLB moneyline because the shipped model's conditioned
P(home) averaged 0.5063 against an actual home win rate of 0.5315 — a 2.5
point understatement, systematic, on every game. It named the repair and did
not build it. The linescores were then pulled so the rule could be measured
rather than inferred. This is the build, the grade, and one invariant that
caught a subtle leak in it.

## Decision Drivers

- The withholding was for a **bias**, so the gate that reopens the market
  should be a bias test, not a harder and different one.
- A large headline t is a reason to decompose, as it was in hockey.
- No MLB season is pristine, and the grade has to say so.

## Considered Options

- **Correct the margin only.** Cheaper, and it leaves margin and total coming
  from different objects, so a total priced from it would be quietly wrong.
- **Compose a joint from a measured gain table.** Chosen, mirroring ADR 0008.

## Decision Outcome

`NinthInningLayer` and `MLBFinalScoreDistribution` in
`src/coverline/leagues/mlb/rules.py`. The layer maps the **state after the top
of the ninth** — home runs through eight minus away runs through nine — to a
joint distribution over runs gained by each side. Tune on 2021-2023, graded
**once** on 2024-2025, zero hyperparameter trials.

| | value |
|---|---|
| scoreline log-likelihood, both layers | **t = +32.94** |
| of which simply removing an impossible tie | **96%** |
| the measured layer, over and above | **t = +2.49** |
| **moneyline bias, baseline** | **+0.0256, t = 3.61** |
| **moneyline bias, with the layer** | **+0.0025, t = 0.35** |
| binary log-loss on the winner | t = 1.79, **does not clear** |

**The market reopens on the bias gate.** That is the defect ADR 0017 named,
and on a holdout the layer has never seen the residual bias is
indistinguishable from zero. Requiring a log-loss win instead would be moving
the goalposts: it is a different and harder bar than the one that closed the
market.

**And the log-loss result is recorded as a refusal, not omitted.** The layer
removes a bias without demonstrably adding discrimination. That is enough to
reopen a market closed for a bias and **not** enough to claim an edge, and
`what_this_does_not_show` says so in the artifact.

**The moneyline returns only with the layer wired**, exactly as the NHL puck
line does. A model constructed without it still carries the bias, so it still
withholds the market.

## Consequences

**The scales are the trap, and they are constructor arguments.** `exp_home` is
fitted to **observed** home runs, which are truncated in 45% of games. Feeding
those rates into a layer that applies the truncation again counts it twice and
produces a model that looks *better* calibrated than it is. The measured scales
are 0.9293 for the home eight-inning rate and 0.9687 for the away nine-inning
rate — the away one is not 1.0 either, because extra innings give the away team
a tenth. A scale of 1.0 raises rather than being silently accepted.

**Clipping cannot be naive, and the first version was.** Rows are measured and
validated per state. A deficit beyond the measured range borrows the edge row,
whose cells were validated against *that* state — so a cell taking a four-run
deficit to a one-run **win** takes a five-run deficit to a **tie**, which extra
innings forbid. It leaked 0.0004 of mass onto an impossible outcome: small,
and exactly the kind of small this whole layer exists to remove. A clipped row
now drops the cells that would finish level and renormalises.

**Contamination, stated.** `R_HOME` and `R_AWAY` were measured on all five
seasons, so the holdout is not pristine for dispersion. They are held fixed and
identical across both arms, which isolates the layer, but this grade is weaker
than the NHL rules grade, which had genuinely untouched seasons.

**The attempt log gets the decomposed component**, t = 2.49, per ADR 0014.
Weight moves to 0.9327 pooled and 0.9168 robust.

## Recovery

`model/ingest/mlb_linescores.py` refetches; `model/fit_mlb_rules.py` re-grades;
`model/export_fitted.py` rebuilds `data/mlb_rules.json`. Deleting that file
returns `primary_markets` to `("runline", "total")`, which is the state ADR
0017 left.

## Revisit Triggers

- **The residual bias becomes significant again.** The gate that reopened the
  market is the gate that closes it.
- **Binary log-loss clears.** That is a stronger claim than this market was
  reopened on and deserves its own record.
- **`R_HOME` or `R_AWAY` are refitted.** Both arms depend on them, and the
  runline's error cancellation from ADR 0017 particularly so.
- **A pristine MLB season becomes available.** Everything here is in sample
  for dispersion; a clean season would let the layer be graded properly rather
  than carefully.
