---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-nhl-nba-structure
---

# 0005. Ship NHL and NBA as structure, with no fitted coefficients

## Context and Problem Statement

The project is expanding from three leagues to five. NFL, CFB and MLB all had
something to port: a shipped coefficient vector, a committed walk-forward
cache, a held-out grade, or all three. NHL and NBA have none of it — no legacy
model, no cache, no grading.

Both could be fitted in an afternoon. Hockey goal rates and basketball
efficiency ratings are not hard to estimate, and the resulting numbers would
look entirely reasonable sitting next to the ones that were earned.

## Considered Options

- **Fit and ship.** Five working leagues today. The coefficients would be
  unvalidated, and in six months nothing in the file would say so.
- **Ship nothing.** Honest, and leaves the seam unproven for two of the five
  leagues it claims to serve — including both new distribution shapes.
- **Ship the structure, refuse to ship numbers.** The packages define what a
  model must provide and which family fits, and cannot produce a prediction
  unless a caller supplies fitted parameters.

## Decision Outcome

Chosen: **structure, with no fitted coefficients, enforced by a test.**

A plausible constant is indistinguishable from a measured one once the
session that produced it is over. That is the specific failure this project
already carries a permanent ledger row about — eight shipped constants citing
a grid search that exists in no committed file — and it is not a hypothetical
risk here, it is the same risk, on a bigger surface.

So `NHLModel` and `NBAModel` take their parameters as arguments. There is no
module-level constant to import by accident, and
`tests/core/test_nhl_nba.py::test_neither_package_ships_a_fitted_constant`
fails on any float literal appearing at module scope in either file.

**What the packages ARE for.** Two things. They prove the seam works for all
five leagues — NHL exercises the Poisson family, NBA the continuous-margin
one — so the architecture claim is tested rather than asserted. And they put
the known traps where whoever fits these will actually look.

**The traps, recorded now because they will otherwise be rediscovered
expensively:**

*NHL — empty nets.* Roughly 7% of goals, ~0.42 a game, and not random: they
arrive conditional on a late one-goal deficit, in a league where about 57% of
games are one-goal games. So they inflate the winner's score in exactly the
games that decide the puck line. A homogeneous scoring process over-prices the
underdog at −1.5. This is why `primary_markets` does **not** include the puck
line: it would be pricing the market whose bias is best understood and least
corrected.

*NHL — the fixed line.* The puck line is always 1.5 and never moves; the book
expresses information through price instead. That makes the SHAPE of the goal
distribution do work the mean cannot, so two models agreeing on the moneyline
can disagree on the puck line. A win-probability model is insufficient here in
a way it is not for football.

*NBA — sigma is not constant.* Margin variance correlates about 0.60 with
spread magnitude, against roughly −0.06 in the NFL. Copying NFL's fixed
`MARGIN_SD` makes a model over-confident on small spreads and under-confident
on large ones, in a way that presents as a calibration problem rather than a
modelling one. `NBAModel` therefore takes a sigma **function**, with no
default, so passing a constant has to be deliberate.

*NBA — minutes are the model.* The published work that beats this market is a
player-impact metric weighted by projected minutes with injury designations
applied, not a team rating. `GameFeatures.minutes_projected` records whether
that layer ran, so a team-level approximation is not mistaken for an NBA
model.

**MLB is the cautionary note both of these should read.** Baseball is the
sport everyone reaches for Poisson with, and measuring found runs 2.2x
overdispersed and the two sides independent to three decimal places — so the
obvious family was wrong and the obvious fix for it (a shared component) was
wrong in a second way. Measure hockey before assuming it is Poisson.

## Consequences

Better: five leagues on one seam, and two of them cannot silently ship
unvalidated numbers. The architecture claim is exercised by both distribution
shapes rather than argued for.

Worse: NHL and NBA cannot price anything today. That is the intended cost.

## Recovery

Nothing was dropped; nothing existed to drop. Fitting either league means
supplying parameters to an existing interface and logging the fit as an
attempt in `evidence/attempts.yaml` like any other change. The packages are at
`src/coverline/leagues/{nhl,nba}/model.py`.

## Revisit Triggers

- **When either league is fitted**, move it from `STRUCTURAL_ONLY_LEAGUES` to
  `IMPLEMENTED_LEAGUES` deliberately; a test already fails if the sets stop
  accounting for all five.
- **If a constant appears in either file**, the guard fails — and the right
  response is to ask what measured it, not to relax the guard.
- **If NHL measures overdispersed**, as MLB did, the Poisson family choice in
  this record is wrong and the package needs the negative-binomial one.
