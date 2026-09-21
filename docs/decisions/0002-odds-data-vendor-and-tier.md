---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: execution-odds-ingest
---

# 0002. Buy The Odds API at the 100K tier, and never split the CLV baseline

## Context and Problem Statement

The execution layer needs market prices for three distinct jobs, and the design
research left one question open that blocked committing to any vendor: whether
The Odds API carries a sharp book at all. Without one there is no fair-price
baseline on the affordable tiers, and CLV — the metric the whole evaluation
approach now rests on — cannot be computed honestly.

A cheaper idea was also on the table: stack the free tiers of two different
providers and pay nothing.

## Considered Options

- **Two providers' free tiers (500 credits/month each).** Zero cost.
- **$30 / 20K tier.** Covers closing capture with large headroom.
- **$59 / 100K tier.** Covers all three jobs with roughly 75% spare.
- **$119 / 5M tier.** What the design research recommended, on the assumption
  of heavy props polling and a 5-minute-resolution historical backfill.

## Decision Outcome

Chosen: **$59 / 100K**, on The Odds API, with a single never-changing source
for the CLV baseline.

**The blocking question is resolved: Pinnacle IS carried, in the `eu` region.**
Their bookmaker page adds a caveat worth recording — "Odds are from public
website which may incur a delay." Acceptable for a closing baseline; NOT
acceptable for steam-chasing, and no design should assume otherwise.

**Free-tier stacking was rejected on methodology, not cost.** CLV is worth
adopting because of variance reduction: it validates a change in one season
where win/loss records need thousands of bets. That reduction depends on the
reference being the same instrument every time. Splitting closes across two
providers injects a between-source variance term into the exact metric chosen
for its low variance. It is worse than it first looks, because the book panel
determines the overround, so changing panels changes the devig result — and
measurement during this build showed devig METHOD alone moves the fair
probability about 1.5 points on a moderately lopsided market, comparable to an
entire edge. Changing the panel underneath is a larger perturbation than that.
Stacking free tiers would spend the metric's main advantage to save $360/year.

**Legitimate two-source design, for the record:** split by FUNCTION, not by
quota. One provider is the canonical CLV baseline and never changes. A second
source may feed line shopping, where only the maximum available price matters
and consistency is irrelevant. That is sound; quota-splitting is not.

**Why not $30**, given closing capture is only ~2.3% of quota: the moment line
shopping runs — the job the strategy audit identified as worth +2.31 points of
ROI, more than the model edge itself — steady-state lands around 25,390
credits/month, which is 1.3x over the 20K tier.

**Why not $119**: that recommendation assumed a 5-minute-resolution historical
backfill. Closing-snapshot backfill is far cheaper because one historical
request covers every game at that timestamp — all five leagues, one season
each, is 92,160 credits, under a single month of the 100K tier.

Credit arithmetic, derived from the published formulas (current odds =
markets x regions; historical = 10 x markets x regions; event props = 1 credit
per unique market returned, per region, PER EVENT):

| job | credits/month | share of 100K |
|-----|---------------|---------------|
| closing capture, eu+us, peak season | 2,262 | 2% |
| line shopping, 15-min window | 12,960 | 13% |
| props, 1 snapshot/game NBA+NHL | 10,168 | 10% |
| steady state | 25,390 | 25% |

The ~75K monthly headroom absorbs one additional historical season per month,
so three seasons of CLV history accumulate by month three at no extra cost.

**Do not subscribe before the ingestion client exists.** The backfill is a
one-shot bulk job that needs code to consume it, and a subscription idling
during development is wasted money.

## Consequences

Better: a real fair-price baseline, so CLV becomes computable rather than
aspirational. Enough headroom that credit budgeting is not a daily concern.

Worse: ~$708/year, roughly 6–14% of the strategy audit's estimated $5–12K
annual profit range. That is a real bite out of a thin edge and should be
revisited if the edge does not materialise.

Constraint accepted: the CLV baseline source is now load-bearing and cannot be
changed without invalidating comparability of the CLV series across the change.
Any future switch needs an overlap period measuring both.

## Recovery

Nothing dropped. If the subscription lapses, captured snapshots remain in
`data/bronze/odds/` as immutable raw data per the ledger's taxonomy, and the
CLV series stays computable over the period covered. Resubscribing resumes
capture; it does not backfill the gap, and a gap in the series must be recorded
rather than interpolated.

## Revisit Triggers

- **If props become a serious part of the book**, the per-event credit model
  changes the arithmetic materially (MLB at 3 snapshots/game alone is 29% of
  quota) and $119 comes back into consideration.
- **If steady-state usage exceeds ~70K/month**, headroom for historical
  backfill is gone and the tier should be re-examined before it starts
  silently truncating capture.
- **If the Pinnacle feed's delay proves material** for closing-line capture —
  measurable by comparing captured closes against a known-good source on a
  sample — the baseline choice needs rethinking, not just the tier.
- **If realised profit after a full season is below the data cost**, this
  subscription is a net loss and the whole execution layer's scope should be
  reconsidered before renewing.
