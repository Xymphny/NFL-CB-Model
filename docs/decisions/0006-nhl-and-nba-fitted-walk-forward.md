---
status: accepted
date: 2026-09-21
supersedes: 0005-nhl-and-nba-structure-without-coefficients.md
superseded_by: null
ledger_id: league-nhl-nba-structure
---

# 0006. Fit NHL and NBA within season, and ship the graded parameters

## Context and Problem Statement

ADR 0005 shipped both leagues as structure with no coefficients, on the
grounds that a plausible constant is indistinguishable from a measured one
once the session that produced it is over. That held until the numbers could
be earned.

The first attempt earned nothing. Static ratings fitted across seasons were
graded once and **failed**: NBA **t = −6.96**, decisively worse than
predicting the league-average margin, and NHL **t = −1.33**. Three-year-old
NBA ratings are not merely stale, they are actively misleading in a sport with
that much roster turnover.

Both failures pointed at the same repair: rate **within** season and walk
forward, which is what the NFL and MLB models here already do.

## Considered Options

- **Stay structure-only.** Defensible, and ignores that the repair is obvious
  and testable.
- **Refit and reuse the same holdout.** Fastest. Test-set reuse, and each
  reuse raises the chance of a spurious pass whatever the code says.
- **Refit and grade on seasons never graded.** Costs the remaining clean data.

## Decision Outcome

Chosen: **refit walking forward, grade once on clean seasons, ship what
clears.**

| | tuned on | graded once on | result |
|---|---|---|---|
| NBA | 2021–2022 | 2023 | **t = +5.96** |
| NHL | — (0 trials) | 2024 | **t = +3.02** |

Both clear. The sign flip on NBA, from −6.96 to +5.96, is the whole finding.

**Holdout accounting, because it is easy to lose track of.** NBA 2024–2025
were spent on the static question, so 2023 — never graded — was used instead,
and 2024–2025 remain spent. NHL had only 2024 and 2025 usable at all
(2021–2023 are corrupt), and 2025 was spent, leaving 2024. **There is now
nothing clean left in either league for a third question.** That is a real
cost of having asked the static one first.

**NHL hyperparameters were fixed a priori, with zero trials**, because
searching on the only ungraded season and then grading on it is how a result
gets manufactured. Weaker than NBA's design; also the only honest option.

**NBA's 18 hyperparameter combinations were checked against their own noise
ceiling.** Bailey and López de Prado's bound puts the expected best t from 18
pure-noise trials at 1.85; the observed 5.96 clears it more than threefold.

**The parameters ship as graded artifacts, not module constants.**
`data/{nhl,nba}_fitted.json` each carry their own holdout grade, and the
loader raises `NotFitted` if that grade does not clear. So ADR 0005's actual
property survives: a number in these packages has earned its place and travels
with the evidence for it. The no-module-constants test still passes unchanged.

## Consequences

Better: five leagues implemented, none carrying an unmeasured constant.

Worse, and worth stating plainly:

**NHL's rates are graded; its JOINT distribution is not.** Independent Poisson
under-predicts one-goal games by about ten points — 28.4% against an actual
38.1% — because the two scores are negatively correlated at −0.14 and
`BivariatePoissonDistribution`'s shared component can only express *positive*
correlation. Walking forward does not touch this: it is a property of the
joint, not the rates. The puck line stays out of `primary_markets`, and
`joint_distribution_warning` sits in the artifact so a reader seeing
`supported: true` does not assume the whole model is sound.

**NBA sigma is constant because varying was rejected** (t = −6.48, fitted
slope −0.111). ADR 0005 predicted the opposite. That prediction is *not*
refuted: these ratings separate games across 0.9–8.6 points where market
spreads reach the high teens, so the effect is ruled out only where measured.
`NBAModel` still takes sigma as a function, keeping the question open.

## Recovery

ADR 0005 is superseded, not deleted, and its reasoning stands for the period
it governed. Reverting means deleting the two artifacts; the loaders raise
`NotFitted` and the packages return to refusing, which is the state 0005
describes.

## Revisit Triggers

- **When market spreads are captured**, test ADR 0005's sigma claim properly.
  That is the measurement this one could not make.
- **NHL needs a joint distribution admitting negative correlation** before the
  puck line can be priced. Until then hockey's closeness is unmodelled in a
  measured, one-directional way.
- **Neither league has clean data left.** A third question in either needs new
  seasons, not a re-cut of these.
- **If sportsdataverse fixes the 2021–2023 NHL files**, three usable seasons
  appear and the NHL fit should be redone with a real tuning set.
