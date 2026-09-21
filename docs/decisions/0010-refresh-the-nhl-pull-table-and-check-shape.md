---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-nhl-nba-structure
---

# 0010. Refresh the NHL pull table, and check shape rather than only likelihood

## Context and Problem Statement

ADR 0008 shipped the rules layer at t = 39.99 and named its own revisit
trigger: the pull table is 2016-2021 behaviour applied to a league that pulls
more every year. ADR 0009 named a different gap — MLB was graded on likelihood
but never on calibration, because interval coverage assumes a continuous
symmetric distribution and does not apply to counts.

Both were closed by the same tool, and the first one turned out to be already
costing something.

## Decision Drivers

- A likelihood ratio says one model beats another. It cannot say whether the
  winner is the **right shape**, and the puck line is priced off shape.
- The drift was documented as a future problem. Nobody had checked whether it
  was a present one.
- 2024 and 2025 were reserved for exactly this question and nothing else.

## Considered Options

- **Randomised PIT.** Standard, and the answer depends on a seed.
- **Non-randomised PIT (Czado, Gneiting and Held).** Averages the conditional
  uniform across each atom instead of drawing inside it. Chosen.
- **Leave the table alone and record the drift.** What ADR 0008 did. Defensible
  until the drift was measured to be costing 0.105 goals a game.

## Decision Outcome

**The PIT found a defect the likelihood ratio could not see.** The shipped
layer was well calibrated on the margin — maximum deviation 0.0103 against a
Kolmogorov scale of 0.0265 — and **sloping upward on the total**: 0.85 in the
lowest tenth against 1.11 in the ninth. A slope means bias. The total came out
at 6.219 goals against an actual 6.324, a shortfall of **0.105 at t = +2.35**,
which is small until you remember NHL totals are priced in half-goal steps
around six.

**The cause was the drift ADR 0008 wrote down.** The layer adds 0.617 goals a
game; 2022-2023 delivered 0.727. The 0.110 gap is the total bias to three
decimals.

**The refresh was declared in the file before it was run.** Tune on 2022-2023
— spent as a holdout and therefore usable for tuning — and grade once on
2024-2025, which nothing had touched. Zero trials, as before.

| | first table | refreshed table |
|---|---|---|
| tune | 2016-2021 | 2022-2023 |
| graded once on | 2022-2023 | 2024-2025 |
| overall t | +39.99 | **+34.95** |
| goalie-pull component | +6.75 | **+5.82** |
| total bias | +0.105, t = +2.35 | **−0.006, t = −0.13** |
| total PIT max deviation | 0.0300 | **0.0055** |

The bias is gone, the layer replicates on two seasons it has never seen, and
the total PIT is flat to half a percent. **The refreshed table ships.**
Shipping the stale one because it was graded first would be preferring the
order things happened to what they measured. The superseded grade stays in the
artifact — it is the evidence that the drift is real and that re-measuring is
what fixes it.

**MLB is now calibrated on its own terms.** The away side passes; the home
side exceeds the Kolmogorov scale by a hair, 0.0133 against 0.0123 on 12,148
games. Small, real, recorded, and in sample for `r_home` and `r_away`, so it
is a floor on the error rather than a measure of it.

**The baseline's defect is now visible rather than argued.** Its margin PIT
histogram is starved in the centre — 0.55 in the middle tenth — and piles up
at both ends, which is exactly what assigning a sixth of the mass to an
impossible tie looks like.

## Consequences

**NHL has nothing clean left again.** 2016-2021 tuned the first table,
2022-2023 graded it and then tuned this one, 2024-2025 graded this one. The
next unspent season is 2026. That is a real cost of refreshing, it is recorded
in the artifact, and it means the next refresh has to wait for a season rather
than being run whenever the table looks old.

**A refresh cadence now has a price attached.** Re-measuring every season is
correct and each re-measure costs a holdout. This is the tension to manage,
not a problem to solve.

## Recovery

`model/fit_nhl_rules.py --refresh` re-runs the graded refresh;
`model/export_fitted.py` rebuilds `data/nhl_rules.json` from it; reverting the
`REFRESH_*` seasons in the exporter ships the first table again, with its bias.

## Revisit Triggers

- **`test_the_rules_layer_is_calibrated_on_the_total` fails.** The pull table
  has drifted again, and this is the cheapest signal of it.
- **The total PIT histogram starts sloping.** Same thing, seen earlier.
- **2026 completes.** The next unspent season, and the next refresh.
- **MLB's home-side deviation grows.** Small was the whole defence.
- **The margin PIT slope grows.** It sits at 0.90 to 1.05 across the shipped
  holdout, which is within scale and is not nothing.
