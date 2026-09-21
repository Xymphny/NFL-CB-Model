---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: frozen-threshold-grid
---

# 0013. Reconstruct the missing grid without selecting anything

## Context and Problem Statement

`migration/ledger.yaml`'s `frozen-threshold-grid` row has been open since the
rebuild began, and this repository's own documentation calls it the oldest
wound: shipped constants in `model/player_projection.py` cite a grid search
whose output exists in no committed file and whose **specification is also
gone**. Its honesty note names the remedy — "treat the constants as
asserted-not-shown until a `calibrate_player_td_lambda.py` exists."

Nobody had written it, and the reason is that the obvious way to write it is
forbidden. `model/ratings.py` is the worked example: every `calibrate_*`
script printed the held-out column beside each candidate, an operator saw the
test curve while turning a knob, and `half_life=100` shipped on the strength
of it. That took a year and two graded seasons to catch.

## Decision Drivers

- The constants cannot be checked, updated, or defended while the grid is
  missing, and that is true regardless of whether they are any good.
- `2024-2025` are spent on the shipped configuration's grade in
  `model/player_projection_results.json`.
- A reconstruction that spends a holdout to confirm old numbers is a bad
  trade.

## Considered Options

- **Re-run the search properly and re-select.** Costs a holdout to confirm
  values that are probably fine, and any re-selection restarts the leak one
  season later.
- **Use `HoldoutVault`.** The sanctioned tool, and it still requires computing
  the sealed numbers.
- **Sweep on training data only and select nothing.** Chosen. There is nothing
  to leak because the sealed numbers are never computed in the process.

## Decision Outcome

`model/calibrate_player_td_lambda.py` sweeps every constant on 2016-2023 and
commits the surface to `model/player_td_grid.json`. No held-out metric is
computed anywhere in the run.

**The shipped constants hold up. Seven of ten sit exactly on the training
argmin** — `TD_POWER`, `OPP_SHRINK_TD`, `USAGE_W`, `RZ_OPP_SHRINK`,
`TIER_CUTS`, `TIER_K` and `TD_MIN_LAMBDA`.

**The three that do not are the three flattest knobs in the sweep.**

| constant | shipped | train argmin | spread of train log-loss |
|---|---|---|---|
| `ENV_CLAMP` | (0.6, 1.5) | (0.8, 1.25) | **0.000022** |
| `ENV_DAMP` | 0.5 | 0.0 | 0.000468 |
| `LONG_CRED` | 150.0 | 100.0 | 0.000534 |

Against load-bearing spreads of 0.018 for `TIER_K` and 0.009 for `TIER_CUTS`,
these move nothing. `ENV_CLAMP` shifts log-loss by four thousandths of a
percent across its entire range.

**Nothing was re-selected.** Adopting a train argmin is a new selection the
next holdout pays for, and a spread of 0.00002 is not a reason to move a live
constant.

## Consequences

**The confound nearly reversed the conclusion, and the guard that caught it
was aimed at something smaller.** Eligibility for the `anytime_td` sample is
`max(lambda, base) >= TD_MIN_LAMBDA`, and *every* constant in the sweep moves
lambda. The first version of this script guarded only `TD_MIN_LAMBDA` on those
grounds; the guard fired on all ten.

That matters because admitting more low-lambda players **lowers** log-loss for
free — they are easy negatives — so a configuration looks better by being more
permissive. The first run reported `TIER_K = {0: 12, 1: 8, 2: 4}` beating the
shipped value by 0.011 while admitting 1,124 more player-weeks, and concluded
that eight of ten constants were off their argmin. Scored on a **fixed
population** — the player-weeks eligible under the shipped constants — that
same configuration is *worse* than shipped, and seven of ten constants are on
their argmin. The conclusion reversed completely.

**The row stays open.** What exists now is a reconstruction, not a recovery.
The original grid's output and specification are still gone, and this sweep is
a different search asking a narrower question.

**The fixed evaluation set cannot judge a more permissive floor.** Lowering
`TD_MIN_LAMBDA` below 0.15 only adds player-weeks outside the fixed set, so
its log-loss is unchanged to six decimals by construction. That is a real
blind spot, not a pass.

## Recovery

`model/calibrate_player_td_lambda.py --cache <pickle>` re-runs the whole sweep
in about ten minutes from the nflverse caches. Deleting
`model/player_td_grid.json` returns the repository to the state this record
found it in.

## Revisit Triggers

- **A constant that misses its argmin stops being flat.** The decision not to
  move it rests entirely on the spread, and
  `test_the_constants_that_miss_the_argmin_are_the_flat_ones` is where that
  fails.
- **Fewer than seven constants sit on their argmin.** Something moved and the
  reconstruction no longer describes what ships.
- **A new holdout season becomes available for props.** Then, and only then,
  re-selecting is affordable.
- **The original grid output turns up.** The row closes properly instead of
  being superseded by a reconstruction.
