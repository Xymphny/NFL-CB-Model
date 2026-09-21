---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: evidence-attempt-log
---

# 0003. Reconstruct the attempt log, and treat the weight it yields as a ceiling

## Context and Problem Statement

The strategy audit retired the ship/no-ship significance gate — at ~2,000 NFL
games its minimum detectable effect is about +3.1 points of cover rate, so a
change worth a real +1 point cleared t > 1.96 roughly 14% of the time, and
whatever did pass was inflated ~2.7x. The replacement is empirical-Bayes
shrinkage: candidates ship at weight b = 1 - 1/E[t²].

That weight needs the t-statistic of every candidate ever evaluated, failures
included. No such log existed. The project's README ledger recorded what was
FOUND, which is the wrong variable — it is a record of conclusions, and the
weight is a property of the attempt distribution. So `staking.py` shipped with
no default weight and sizing could not be used on real money at all.

## Considered Options

- **Start logging prospectively and wait.** Clean, and leaves sizing blocked
  for however many months it takes to accumulate enough attempts.
- **Pick a conservative weight by judgement until data exists.** Fast, and
  reintroduces exactly the judgement call the mechanism was meant to replace.
- **Reconstruct the log from committed artifacts.** The results JSONs and the
  README ledger already record t-statistics for work done; recovering them
  gives a usable weight now, at the cost of an incomplete sample.

## Decision Outcome

Chosen: **reconstruct, and label the result a ceiling rather than an estimate.**

Five eligible attempts were recovered, all held-out paired gains against the
incumbent:

| attempt | t | decision |
|---|---|---|
| preseason-prior-regression | +2.28 | shipped |
| cfb-returning-production-rescale | +1.93 | not shipped |
| elo-term-for-ngs-absent-games | +1.44 | not shipped |
| qb-continuity-conditional-prior | −0.17 | rejected |
| half-life-100-vs-default-6 | −1.21 | shipped but contradicted |

RMS t = 1.580, E[t²] = 2.4980, **b = 0.5997**. Candidates ship at ~60% weight.

**One estimand, strictly.** Only held-out paired gains are eligible.
Training-side coefficient t-statistics are recorded under `excluded` with
reasons. This is not fastidiousness: the same QB-continuity hypothesis measured
t = +2.63 on train and t = −0.17 held out, in the same week. Admitting
train-side statistics — this repo contains values up to +8.51 — would produce a
weight above 0.75 from numbers that predicted nothing, and a test pins that.

**Why a ceiling.** A reconstructed log contains only attempts someone wrote up,
and what goes unwritten skews toward nulls — nobody documents a failure they
were not already invested in. Missing failures bias E[t²] up, so they bias b
up. `AttemptLog.is_ceiling` returns True while any recovered attempt remains,
and the summary string says CEILING. Treat 0.5997 as an upper bound on what is
defensible, not as the answer.

**The selection effect is quantified rather than warned about.** Dropping the
two non-positive attempts moves b from 0.5997 to 0.7272 — a 21% inflation from
losing two rows out of five. `drop_negative_for_demonstration` exists solely so
a test can measure that on this project's own data.

## Consequences

Better: sizing is unblocked and rests on arithmetic with a date and a source
for every input. Two numbers that were previously invisible are now load-bearing
and audited — the count of failures, and whether the log was recovered or kept.

Worse: the weight is built on five observations, which is few. It will move,
possibly a lot, as prospective attempts accumulate. Anyone reading a stake size
should know it descends from a five-row table.

Newly visible: `half-life-100-vs-default-6` sits in the log at t = −1.21 with
decision `shipped_but_contradicted`. A shipped parameter whose held-out grade
says it is worse than the default it replaced is an open item, and putting it in
the log means it can no longer be forgotten — removing it would raise the weight,
which the tests would catch.

## Recovery

Nothing dropped. The recovered rows cite their sources
(`model/*_results.json`, README ledger) and can be re-derived from those
artifacts if `evidence/attempts.yaml` is lost. The log is classified `training`
in the ledger, not `derived`: it encodes judgement about which attempts count
and under which estimand, and rerunning code does not reproduce it.

## Revisit Triggers

- **When prospectively-logged attempts outnumber recovered ones**, re-examine
  whether the ceiling framing still applies; `is_ceiling` flips automatically
  once no recovered rows remain.
- **If RMS t drifts toward 1.0** as attempts accumulate, b approaches 0 and the
  arithmetic is saying the pipeline produces nothing distinguishable from
  noise. That is a finding about the research programme, not a number to tune.
- **If the log ever contains no failures**, suspect the log rather than
  celebrating; a test asserts at least two non-positive attempts.
- **If a second estimand becomes unavoidable** — say a CLV-based gate replaces
  MAE-based ones — do not mix them in one E[t²]. Keep parallel logs and decide
  deliberately which drives sizing.
