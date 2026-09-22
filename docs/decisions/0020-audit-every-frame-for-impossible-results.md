---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: data-integrity
---

# 0020. Audit every frame for impossible results, on purpose

## Context and Problem Statement

Three separate times a model or a cache was caught holding an outcome its
league does not permit: a tied NHL game (ADR 0007), a tied MLB game (ADR
0017), and 76 tied CFB games in the cache that produced two shipped constants
(ADR 0019). Every one was found while looking at something else.

ADR 0019 closed with the observation that ties are the cheapest impossibility
to check and unlikely to be the only kind. Three finds by accident is an
argument for looking on purpose.

## Decision Drivers

- The faults are invisible to every statistical check. A tie moves a mean
  slightly and nothing else; models converge either way.
- The rulebook is a free oracle. It needs no distribution and no holdout.

## Considered Options

- **Check the improbable as well.** A 2-0 NFL game is legal and has happened
  once since 1940. Flagging it needs a distribution, and a distribution is the
  thing being validated. Out of scope, and said so.
- **Check only the impossible, across every frame.** Chosen.

## Decision Outcome

`model/audit_data_integrity.py` walks all 29 retained frames past every check
that applies: ties against what each league permits, team scores the rules
cannot produce, self-play, and duplicate keys. It exits non-zero on failure so
it can gate a pipeline, and a test runs it to confirm the exit code means
something — this repository pushed a red commit once by reading grep's exit
status instead of a check's.

**It found two more corrupt CFB rows.** Football cannot score exactly one
point: a safety is two, a field goal three, a touchdown six, and the one-point
play exists only as a conversion after a touchdown. `model/cfb_schedule_cache.csv`
contains Florida Atlantic **1-0** Georgia Southern and Kansas State **1-1**
TCU, both in 2021 — the same season as the bulk of ADR 0019's faults.

**An absent result is not a false one, and conflating them is a real mistake I
made.** The first version of the audit reported all five NBA season files as
failing. Each carries exactly one row at 0-0 with
`status_type_completed = False`: a cancelled game, correctly excluded by the
fit's own filter. The audit now applies the completion filter and reports what
it excluded.

**That distinction is the whole of ADR 0019 in one sentence.** The CFB schedule
cache has **no completion column**, so its 83 unfetched games sit in it as 0-0
results indistinguishable from played ones. A frame that can say a game was not
played is a frame whose zeros are safe. The audit now annotates every frame
lacking that column, and most of them lack it.

## Consequences

**A margin-only cache cannot be audited for scores, and one of the corrupt rows
is inside the one that ships constants.** `cfb_full_walk_forward_cache.csv`
stores `actual_margin` and not the two scores. Kansas State 1-1 TCU appears as
a margin of zero and is caught by the tie check; Florida Atlantic 1-0 Georgia
Southern appears as a margin of **+1** and is undetectable. So at least one
impossible game remains inside the frame that produced `MARGIN_SD`, and no
check here can find it. The remedy is for that cache to carry both scores, and
until it does this limit is asserted rather than assumed away.

**The data is still not repaired.** Same reason as ADR 0019: rewriting rows by
hand would be inventing results.

## Recovery

Deleting `model/audit_data_integrity.py` and `model/data_integrity.json`
returns the repository to checking impossibility only where somebody happened
to look.

## Revisit Triggers

- **The CFB schedule cache is refetched.** The unfetched, frozen and
  impossible-score rows should all resolve, and the audit will say whether
  they did.
- **A cache starts carrying both scores where it carried a margin.** The score
  check then applies to it and the structural limit above goes away.
- **Any frame's tie rate crosses its allowance.** The NFL's is 1% against a
  true rate near 0.2%; hitting it means something other than real ties.
- **A new impossibility is worth adding.** Nothing here checks a score
  reachable in principle but not by the sequence that produced it, and nothing
  checks inning or period sums against their totals.
