---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-mlb
---

# 0017. MLB has hockey's two-rule problem, and it lands on the other market

## Context and Problem Statement

`model/calibration_pit.json` flagged the MLB **home** score as marginally
miscalibrated — a maximum PIT deviation of 0.0133 against a Kolmogorov scale
of 0.0123 — while the away side passed at 0.0076. ADR 0010 recorded that as
"small, real, recorded", which it is. What it also is, in a model that treats
the two sides symmetrically, is a **one-sided** defect, and a one-sided defect
is a structural hint rather than noise.

Following it lands on the same shape ADR 0007 found in hockey.

## Decision Drivers

- The MLB moneyline is in `primary_markets` and Thursday brings live prices.
- A symmetric model with a one-sided calibration failure has a reason, and the
  reason is usually a rule.

## Considered Options

- **Treat 0.0133 against 0.0123 as noise.** Defensible on its own, and it
  leaves the question of *why only the home side* unasked.
- **Follow it.** Chosen.

## Decision Outcome

**Two league rules, neither expressible by a smooth distribution.**

**Extra innings resolve every game.** Zero of 12,148 games finished tied. The
shipped model assigns **10.07%** of its mass to a tied final score. This is
precisely hockey's overtime defect in a different sport.

**The home team stops batting when it leads.** It never gets a ninth inning
while ahead, and a walk-off ends play the instant it takes the lead. The
fingerprints are unambiguous:

| | home | away |
|---|---|---|
| runs scored *by the winner* | 6.024 | 6.443 |
| variance of score | 9.63 | 10.49 |
| P(wins by exactly 1) | **0.1725** | **0.1111** |

The winning home team scores 0.42 fewer runs than the winning away team, its
score distribution is narrower, and one-run home wins are over-represented by
6.1 points. All three are truncation from above, in exactly the games the home
team wins.

**Where it hurts is not where it looks.**

**The runline is fine.** The model puts 0.6420 below the 1.5 line against an
actual 0.6409 — an error of 0.0011. The fictitious tie mass and the missing
one-run wins sit on the **same side** of 1.5, so the errors cancel exactly
where that market is priced. That is a measurement, not an argument: it is
right for a reason that would change if either component moved, and the ADR
records the reason so a future change is visible rather than surprising.

**The moneyline is withheld.** It is the market that looked safest, because
`recommend.py` already conditions the tie out and its docstring cites MLB as
the case that motivated doing so. Conditioning redistributes that 10%
**proportionally**, and the walk-off rule gives it overwhelmingly to the home
side. Model P(home | no tie) averages **0.5063** against an actual home win
rate of **0.5315** — a 2.5 point understatement, systematic, in one direction,
on every game. That is several times a typical edge and it would manufacture a
false edge on the away side of every card.

## Consequences

`MLBModel.primary_markets` returns `("runline", "total")`. The moneyline comes
back when a redistribution layer exists and has been graded, which is the same
path the NHL puck line took through ADRs 0007 and 0008.

**The repair is specified.** The tie mass must be redistributed to the two
sides in a measured, score-dependent proportion rather than in proportion to
their existing mass — a walk-off layer conditional on the pre-ninth state, the
baseball analogue of the goalie-pull layer.

**Nothing here spends a holdout.** `measure_mlb_rules.py` fits nothing. It
also cannot be graded yet without care: `R_HOME` and `R_AWAY` were measured on
all five seasons, so no MLB season is pristine, and the layer's grade will
have to say so.

## Recovery

Deleting `model/mlb_rules_structure.json` and restoring `"moneyline"` to
`primary_markets` returns the previous behaviour, which offers a market with a
known 2.5 point bias.

## Revisit Triggers

- **The measured moneyline error drops below 1.5 points.** The market can come
  back, and it comes back with a grade rather than with this measurement.
- **The runline error leaves ±0.01.** The cancellation has stopped holding and
  that market needs the same treatment.
- **`R_HOME` or `R_AWAY` are refitted.** Both errors above are conditional on
  them, and the runline's cancellation particularly so.
- **A clean MLB season becomes available.** Everything here is in sample for
  the dispersion constants, which is why this record measures and does not
  grade.
