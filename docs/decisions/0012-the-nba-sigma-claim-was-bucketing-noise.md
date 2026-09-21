---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-calibration
---

# 0012. The NBA sigma claim was bucketing noise

## Context and Problem Statement

ADR 0005 recorded that NBA margin variance correlates about **0.60** with
spread magnitude, against roughly −0.06 in the NFL, and concluded that a
constant sigma is "defensible in football and wrong in basketball." That
sentence was written into `NormalMarginDistribution`'s docstring in `core`,
where it has been justifying a design decision ever since.

ADR 0006 downgraded it to **inconclusive**: a fitted sigma slope of −0.111 and
a held-out correlation of +0.015 said flat, but the record noted that these
ratings "separate games only from 0.9 to 8.6 points on average, where real NBA
spreads reach the high teens," so the effect was ruled out where measured and
untested where claimed.

Both halves of that turn out to be wrong.

## Decision Drivers

- The claim sits in `core`, not in a league, so it is the most widely read
  sentence in the repository about why the API has the shape it has.
- A 0.60 correlation is either a reason or it is not, and nobody had asked
  which.

## Considered Options

- **Wait for market spreads.** ADR 0006's plan, and correct if the ratings
  really did not reach the relevant range.
- **Check whether the ratings reach it, and check what a bucketed correlation
  is worth.** Chosen, because both are answerable today.

## Decision Outcome

**"0.9 to 8.6" were bucket means, not a range.** The predicted spread
magnitude over 3,540 walk-forward games runs from **0.003 to 23.3 points**,
which is where real NBA spreads live. The question was answerable all along.

**The answer is that dispersion does not vary with the spread.** Per-game
correlation between predicted spread magnitude and absolute residual is
**−0.016**; against squared residual, −0.008. Residual sd by decile is flat —
13.67, 14.38, 14.13, 13.79, 13.86, 13.49, 13.46, 13.63, 13.07, 14.23 — with no
trend. Above 15 points of predicted spread the residual sd is 15.10 on 68
games, about one standard error above the pooled 13.78.

**And the 0.60 was never evidence.** It is a correlation over a handful of
bucket means, and a handful of points correlate strongly by accident. Pushing
**constant-variance** noise through the same five buckets on this data gives a
median absolute correlation of **0.397** and a 90th percentile of **0.801**.
"About 0.60" is unremarkable at that bucket count. The observed value here is
**−0.730** — larger, opposite in sign, and equally meaningless.

This is the same error the attempt log exists to catch, in a different
costume: a statistic reported without the distribution it would have under the
null.

**`margin_sd` stays a constructor argument**, for a better reason than the one
it had. A league should own its dispersion rather than inherit a class
constant, and `NBAModel` takes sigma as a **function** so the question stays
open rather than being closed by a default. What is settled is that the 0.60
was never a reason to build it that way.

**NBA is under-confident on its graded holdout.** Sigma is fitted on the tune
seasons at 14.1666; the holdout realised 12.99, a ratio of 0.917, with the
nominal 95% interval covering 96.5% and the 80% covering 84.8%. Intervals are
too wide, so edges are understated and stakes too small. That is the safe
direction and it is not free.

## Consequences

The docstring in `core` no longer asserts a number that does not survive its
own null, and `model/distribution_grades.json` carries the noise ceiling so
the next person can see what a bucketed correlation is worth before quoting
one.

ADR 0005 and ADR 0006 are **not** amended. They were right for what they knew
and the correction belongs in its own record, which is the same rule applied
when 0006 superseded 0005 rather than editing it.

## Recovery

`model/grade_distributions.py` rebuilds every number here from the retained
NBA parquet and the graded artifact. Nothing depends on state that exists on
one machine.

## Revisit Triggers

- **Market spreads land.** They reach further than model spreads and are the
  better test; this record says the effect is absent up to 23 points, not that
  it is absent everywhere.
- **`test_nba_is_under_confident_on_its_graded_holdout` flips.** The error has
  changed direction, which is the dangerous one.
- **A rating scheme with more separation ships.** The per-game correlation
  should be recomputed against it rather than assumed to carry over.
- **Any other bucketed correlation appears in this repository.** It needs its
  noise ceiling beside it, and this is the record that says why.
