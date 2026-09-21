---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: evidence-attempt-log
---

# 0014. Put a ceiling on what counts as an attempt

## Context and Problem Statement

`evidence/attempts.yaml` produces the shrinkage weight `b = 1 − 1/E[t²]` that
sizes every bet. Its eligibility rule was already explicit: an attempt counts
only if it is a **held-out paired gain against the incumbent, expressed as
estimate / standard_error**.

Earlier today this project produced a held-out paired gain against the
incumbent of **t = 39.99** — the NHL rules layer — and later **t = 34.95** on
a second holdout. By the letter of the rule, both qualified.

Logging them would have moved `E[t²]` from 14.2 to **215** and `b` from 0.9296
to **0.9954**. That is shrinkage that has stopped shrinking.

## Decision Drivers

- The weight is **quadratic in t**, so one outsized row decides it and every
  ordinary attempt becomes noise beside it.
- Every league now gets a distributional grade, so rows of this shape will
  keep arriving. The failure is not a one-off; it is a direction of travel.
- The 2% bankroll cap bounds the damage but does not prevent it: at a marginal
  edge the inflated weight raises a stake by about 7%.

## Considered Options

- **Ban large t values.** Crude, and wrong in principle — a large effect can
  be real.
- **Split the log by metric.** The log already mixes MAE, rank correlation,
  straight-up accuracy and log-likelihood, deliberately: `t` is unitless and
  the population is "changes this project attempts."
- **Require an outsized t to explain itself, and decompose first.** Chosen.

## Decision Outcome

`evidence.py` refuses any attempt with `|t| >= 12` that carries no
`large_t_justification`. The error tells the reader what to check before
writing one.

**The reason t = 39.99 does not belong is not its size.** 82% of that gain is
the **overtime rule**: the incumbent was assigning about a sixth of its
probability to a tied final score, which the league does not permit. Fixing
something already known to be wrong has a **certain sign before the grade**,
so its t measures sample size rather than surprise — and this weight is an
estimate of how much surprise the project's attempts typically contain.

**So the rule is not "no large t" but "decompose until what remains was
uncertain."** The row that was logged is the goalie-pull component, **t =
5.82**, whose sign genuinely was in doubt. The weight moves 0.9296 → 0.9359
pooled and 0.9069 → 0.9203 robust, which is what logging one ordinary attempt
should do.

The earlier table measured **t = 6.75** for the same component on a different
holdout and was superseded. That is the same question twice, so it is not
logged twice.

## Consequences

**Two figures in the log were corrected while it was open.** The
`nhl-walkforward-rates` row now reads **t = 2.44**, not 3.02: the league base
rate was the mean over the whole frame including unplayed games, and
recomputing after a defect fix is the same look with a corrected estimator,
not a second look. Both numbers stay recorded. That row's explanatory note
also blamed a negative score correlation for hockey's closeness, which ADR
0007 refuted the same day; the t is unaffected and only the sentence beside it
was wrong.

**A dead escape hatch in the README check was removed.**
`test_the_shrinkage_weights_match_the_attempt_log` accepted either the real
attempt count **or** the literal string `"Seven attempts"`, left behind when
the log grew past seven. A stale README would have passed the test that exists
to catch a stale README.

**The ceiling is a tripwire, not a policy.** It cannot tell a decomposable
effect from an indivisible one. It can only make sure somebody looks.

## Recovery

Deleting `LARGE_T` and its check restores the previous behaviour; the log
itself is unchanged apart from one added row and two corrected values, both
recorded in place.

## Revisit Triggers

- **A genuinely indivisible large effect appears.** Then a
  `large_t_justification` is written, and this record is the context for
  judging whether it is a good one.
- **`test_the_decomposed_component_is_what_got_logged` fails.** Something
  crossed the ceiling and the weight it produced needs reading.
- **The log outgrows its recovered rows.** The weight stops being a ceiling
  and this whole mechanism can be re-examined on prospectively logged data.
- **Market-graded attempts start arriving.** They are the population the
  weight is really meant to describe, and once there are enough of them the
  question of whether model-specification grades belong at all can be settled
  rather than bounded.
