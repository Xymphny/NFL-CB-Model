---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-cfb
---

# 0019. The CFB cache carries results its league forbids

## Context and Problem Statement

ADRs 0007 and 0017 both found a model pricing an outcome its league does not
permit — a tied NHL game, a tied MLB game — and both times it was found by
accident while looking at something else. That is two for two, so the
generalisation was worth running deliberately: does any league's **data** carry
outcomes its own rules disallow?

One does.

## Decision Drivers

- A tie breaks nothing downstream. Means exist, variances exist, models
  converge. It shows up only if something looks.
- `MARGIN_SD` is what every CFB probability divides by, and therefore what
  every stake divides by.

## Considered Options

- **Repair the 76 rows.** Would mean inventing 76 results. Refused.
- **Drop them at the point of measurement and guard the frame.** Chosen.

## Decision Outcome

**76 of 1,731 rows — 4.39% — in `model/cfb_full_walk_forward_cache.csv` carry
a final margin of zero.** College football has not permitted a tie since 1996.
Every one of them is also recorded as `actual_home_win = False`, so a tie
became an away win, which biases any straight-up accuracy computed from the
cache as well as any dispersion.

Two upstream causes in `model/cfb_schedule_cache.csv`:

- **83 rows stored as 0-0** where no score was ever fetched. Arkansas–Rice,
  2021 week 1, is recorded 0-0; Arkansas won 38-17.
- **59 rows frozen at an intermediate score.** Auburn 22-22 Alabama, 2021 —
  a game Alabama won 24-22 in four overtimes.

**Both shipped constants are corrected**, and both old values are recorded:

| | was | is | change |
|---|---|---|---|
| `MARGIN_SD` | 17.5401 | **17.8780** | +1.93% |
| `DVOA_ONLY_MEAN_RESIDUAL` | 2.1644 | **2.2485** | +0.0841 |

The direction matters. A dispersion that is **too small** makes every
probability too confident and oversizes every stake that divides by it. This is
the dangerous direction, and it was 1.93% wrong.

Recomputing after finding a data fault is the same look with a corrected
estimator, not a second look — the precedent set when the NHL rates grade went
from t = 3.02 to t = 2.44 after a lookahead fix.

**The data is NOT repaired.** The cache stays as it was fetched, because
rewriting 76 rows by hand would be inventing results.
`model/fit_data_checks.py` gains `check_no_impossible_ties`, which refuses a
frame with more ties than its league permits, and the measurement scripts
exclude them explicitly at the point of use. The NFL's allowance is small and
non-zero, at 1%, because it is the one league that genuinely has ties.

## Consequences

**The CFB calibration figures in ADR 0009 were measured through the fault.**
That record is not amended — 0012 refused to amend 0005 for the same reason —
and the corrected figures are here: `n` falls from 1,731 to 1,655, the
non-circular split check gives a dispersion ratio of 0.974 against the previous
0.988, coverage at 95% is 0.947, and Bartlett still finds no evidence the
spread varies by season (p = 0.17 against the previous 0.23). The conclusion of
ADR 0009 stands; its numbers move slightly.

**Parity is unaffected.** The CFB parity claim compares the new core's
coefficient output against the legacy model's, and neither reads
`actual_margin`. A corrupt outcome column cannot change a prediction.

## Recovery

Reverting the two constants and the `check_no_impossible_ties` calls restores
the previous behaviour, which measures dispersion through 76 impossible games.

## Revisit Triggers

- **The schedule cache is refetched.** The 83 unfetched rows should fill in and
  the 59 frozen ones should resolve; the guard will say whether they did.
- **Another league's tie rate exceeds its allowance.** The check is in
  `fit_data_checks.py` for every league, not only for CFB.
- **The NFL's allowance is ever hit.** At 1% against a true rate near 0.2%,
  hitting it means something other than real ties.
- **A cache carries a different impossibility.** Ties are the cheapest to
  check and are unlikely to be the only kind; nothing here looks for a
  seven-point safety or a one-point touchdown.
