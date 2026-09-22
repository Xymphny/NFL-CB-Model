---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-cfb
---

# 0021. Repair the CFB scores from a second source

## Context and Problem Statement

ADR 0019 found 76 impossible tied rows in the CFB caches and refused to repair
them: "rewriting 76 rows by hand would be inventing results." ADR 0020 found
two more with a score of one point and made the same refusal. Both were right
**while there was no second source**, and both named a refetch as the first
revisit trigger.

## Decision Drivers

- A tie check finds a frozen score only when the freeze happens to land level.
  There was no reason to think 76 was the whole of it.
- `MARGIN_SD` and `DVOA_ONLY_MEAN_RESIDUAL` are measured through these rows.

## Considered Options

- **Keep excluding.** Measures constants on the survivors of a corrupt set
  and never learns how large the set is.
- **Refetch and adjudicate.** Chosen — the move the NHL work made when
  sportsdataverse's files turned out to carry a constant score on every row.

## Decision Outcome

`model/ingest/cfb_espn.py` pulls 2021-2025 from ESPN, which keys on the **same
event ids** as the existing cache — so the join needs no name matching and no
fuzzy dates, which is the usual way a cross-source check goes wrong — and
carries the **completion flag** the original lacks.

**The corruption is more than twice what the tie check could see.** In the
constants cache, **159 of 1,731 rows (9.19%)** carry a wrong score against the
4.39% that happened to land level. **58 of them flip the winner**, so
`actual_home_win` was wrong on 3.4% of the cache as well. Across the whole
schedule cache, **315 rows** in five seasons.

The pattern is a **frozen** score, not a missing one: Vanderbilt 27-28 UConn
was really 30-28; Missouri 16-23 Florida was really 24-23. A tie check catches
these only by coincidence.

**With the scores repaired there are zero ties across all 1,731 games**, which
is what a sport that abolished them in 1996 should look like.

**This is adjudication, not fabrication**, and the difference is the second
source. Every change is written to `model/cfb_score_repairs.csv` — 315 rows
with the before and after — so the claim is auditable rather than trusted, and
a test asserts the record has not shrunk.

| | original | ties dropped | repaired |
|---|---|---|---|
| `MARGIN_SD` | 17.5401 | 17.8780 | **17.8047** |
| `DVOA_ONLY_MEAN_RESIDUAL` | 2.1644 | 2.2485 | **2.0540** |
| rows used | 1,731 | 1,655 | **1,731** |

The mean moved further than the dispersion did, because a frozen score
understates one side systematically, and that biases a mean far more than a
spread.

## Consequences

**A finding from ADR 0011 reverses.** That record said CFB's labels were the
wrong way round — the number marked `TOTAL_MEAN_PLACEHOLDER` (52.0) was
accurate to seven hundredths of a point, while the one marked merely
`TOTAL_SD_UNVALIDATED` was dangerous. Measured on repaired scores the actual
total mean is **53.66**, so the placeholder is low by **1.66 points at
t = 6.06**. It was never accurate; the corruption, which froze scores low, was
hiding the bias. Neither number is corrected, because the market is withheld
and a right total on an ungraded model is still an ungraded model.

**The whole audit is now green**, 29 frames and zero failures. That is only
meaningful beside the repair record: a green audit and a check that stopped
looking are indistinguishable without it, which is why the test asserts both.

**Ratings are unaffected.** `rating_diff` comes from play-by-play VOA in
`model/cfb_full_walk_forward.py` and never reads a final score, so only the
outcome columns moved. Checked rather than assumed.

**133 rows remain unmatched** and are left alone — games ESPN's regular-season
scoreboard does not carry. They are a smaller unknown than the 315 that were
wrong, and they are stated rather than quietly dropped.

## Recovery

`git revert` restores the corrupt caches. `model/ingest/cfb_espn.py --force`
refetches the reference data; `model/repair_cfb_scores.py` is idempotent and
accumulates its record rather than overwriting it — which it did not do at
first, and a second run erased the first run's 215 repairs before that was
fixed and the full record reconstructed from git.

## Revisit Triggers

- **`model/cfb_score_repairs.csv` shrinks.** The justification for the change
  is the record of it.
- **A tie reappears in the constants cache.** Either the repair was undone or
  a regenerated cache dropped it.
- **The 133 unmatched rows become matchable.** A source that carries them
  would close the last gap.
- **Another cache is found to be frozen rather than missing.** The NFL's
  walk-forward cache has no second source here and has not been cross-checked
  at all.
