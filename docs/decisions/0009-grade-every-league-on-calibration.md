---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-calibration
---

# 0009. Grade every league on calibration, not only on beating a book

## Context and Problem Statement

NHL and NBA were graded on the likelihood of realised scorelines. NFL, CFB and
MLB never were. What they had instead was parity with the legacy pipeline —
which says the rewrite reproduces the old numbers, and nothing about whether
the old numbers were well specified — and, for NFL, an ATS validation, which
asks whether the model beats a book and recorded `supported: false`.

Those are different questions, and the missing one is the one that touches
money on every bet rather than only on the bets that get placed. A model can
be honestly negative against the market and still be badly calibrated. **The
dispersion constant is what every probability divides by, and therefore what
every Kelly fraction divides by.** If it is too small, every stake is too
large, and an ATS record cannot see that at all.

## Decision Drivers

- Five leagues should be readable on one scale. Three of them were not on it.
- `MARGIN_SD` is a single number per league doing a great deal of work with no
  check on whether a single number is enough.
- The check is cheap on data already committed, and a failure would be
  decisive.

## Considered Options

- **Wait for market prices and grade everything against the close.** The right
  test of edge, arriving Thursday, and not a test of calibration at all.
- **Grade on held-out seasons only.** Cleanest, and there are no held-out
  seasons: every league's coefficients were fitted on the full cache.
- **Grade in sample and report it as one-sided.** Chosen. A distribution that
  is miscalibrated in sample is miscalibrated; one that looks fine has proved
  nothing, and the split checks below are the part that is not circular.

## Decision Outcome

`model/grade_distributions.py`, measuring three things per league: the
dispersion ratio, interval coverage at 50/80/95, and likelihood against a
league-average baseline — the same baseline NHL and NBA faced.

**CFB's constant is doing its job, and the obvious reading of that was a
tautology.** `MARGIN_SD` was measured as the residual sd of the exact cache it
is judged against, so the pooled ratio is 1.0000 by construction. Re-estimated
on 2021-2022 and checked on 2023, which it never saw: 17.62 against a realised
17.40, ratio 0.988, 95% interval covering 95.1%. Bartlett finds no evidence
the spread varies across the three seasons (p = 0.23). That is a real pass,
and the circular number is labelled in the artifact so it cannot be read as
one.

**NFL's constant is a compromise, and the season table is the finding.**
Pooled ratio 1.034 with the 95% interval covering 93.2% — tails thinner than
13.2979 implies. Season by season the residual sd runs from **11.73 in 2022 to
15.15 in 2021**, and Bartlett rejects equal variance across ten seasons at
**p = 0.014**, against a sampling SE of about 0.70 for one season's estimate.

This is one test chosen after looking at the season table, which is a garden of
forking paths, and ten seasons is not many. It is recorded as suggestive.

**`MARGIN_SD` is not changed.** Knowing the value moves is not the same as
being able to forecast it. A season-varying sd has to predict next season's
dispersion and be graded on that, and no such forecast exists. Replacing a
compromise with an ungraded guess is the trade this project exists to refuse.

**The spread moves; the bias does not.** NFL season mean residuals run −1.90 to
+1.55 and are consistent with noise at p = 0.24, so there is nothing to
de-bias. CFB is the mirror image: +2.11, +1.94, +2.44 across its three
seasons, with no evidence of variation at p = 0.88. `DVOA_ONLY_MEAN_RESIDUAL`
was already recorded and already uncorrected; what is new is that it is
systematic rather than a period effect.

**All three clear the league-average baseline in sample** — NFL t = +7.67,
CFB t = +10.55, MLB t = +9.70. Weak evidence, reported as weak.

## Consequences

Every league is now on one scale, and two of the three carry a named
calibration defect in the constant rather than in a commit message.

The NFL finding has a concrete consequence that is not acted on here: in a
season where true dispersion is 11.73 and the model uses 13.2979, every edge
is understated and every stake too small; in a season at 15.15 the reverse,
and the reverse is the one that costs a bankroll. Which season is which is not
knowable in advance today.

## Recovery

`model/grade_distributions.py` rebuilds `model/distribution_grades.json` from
committed caches. Deleting both returns the repository to the state before this
record — three leagues with parity and no calibration evidence.

## Revisit Triggers

- **NFL dispersion starts looking constant.** A single `MARGIN_SD` becomes
  defensible, and `test_nfl_dispersion_is_not_constant_across_seasons` is
  where that will surface.
- **A dispersion forecast becomes gradeable.** That is the only thing that
  turns this finding into a change.
- **CFB's out-of-sample ratio leaves 0.9-1.1.** Its constant has stopped
  describing seasons it was not measured on.
- **Held-out seasons become available for any league.** Every number here is
  in sample for the coefficients, and a real holdout replaces the whole record.
- **MLB gets interval coverage.** It is graded on likelihood here but not on
  coverage, because the normal machinery does not apply to counts and the
  count equivalent was not built.
