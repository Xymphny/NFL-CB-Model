---
status: superseded
date: 2026-09-21
supersedes: null
superseded_by: 0006-nhl-and-nba-fitted-walk-forward.md
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

## Fitted and graded, 2026-09-21 — and neither ships

Both were fitted on real data with the holdout fixed before any result was
seen. Neither earned its coefficients, so `STRUCTURAL_ONLY_LEAGUES` is
unchanged and this record stands.

**NHL** (`model/fit_nhl.py`, trained 2024, graded once on 2025): Poisson
attack/defence ratings gained **−0.0137 mean log-likelihood, SE 0.0103,
t = −1.33** against league-average rates. Directionally negative, not
significant. Ratings from one season do not carry to the next well enough to
beat knowing nothing.

**NBA** (`model/fit_nba.py`, trained 2021–2023, graded once on 2024–2025):
ridge team ratings gained **−0.0576, SE 0.0083, t = −6.96**. Decisively worse
than predicting the league-average margin. Three-year-old ratings are not
merely stale in this sport, they are actively misleading — which is why real
NBA models rate within season and walk forward.

### What the fits established anyway

**Hockey IS approximately Poisson** — variance/mean 0.96 and 1.04, unlike
MLB's 2.2 — so the family choice in this record holds.

**But the two scores are NEGATIVELY correlated (−0.14),** and
`BivariatePoissonDistribution`'s shared component can only express POSITIVE
correlation. So it is the wrong instrument for hockey, in the opposite
direction from how it was wrong for baseball. The model under-predicts
one-goal games — 28.1% against an actual 38.3% — because real games stay
closer than independent rates allow: a leading team defends, a trailing team
presses. Empty-net goals push the other way, so the underlying closeness
effect is *larger* than that 10-point gap shows.

**The NBA sigma claim in this record is INCONCLUSIVE, not confirmed.** Fitted
sigma slope was −0.111 and held-out correlation between predicted-spread
magnitude and absolute residual was +0.015 — flat. But these ratings separate
games only from 0.9 to 8.6 points on average, where real NBA spreads reach the
high teens, and the 0.60 figure this record cites was measured against
*market* spreads. The effect is ruled out where measured and untested where
claimed. Testing it properly needs market spreads, which odds capture will
supply.

### The data fault that nearly went through

The first NHL fit ran on sportsdataverse's 2021–2023 schedule files, in which
**every game carries an identical score** — 1,312 rows all 6–3. It fitted,
graded, and reported t = −5.12 with a straight face. `model/fit_data_checks.py`
now refuses degenerate data, and it subsequently caught the NBA All-Star Game
(EAST 211, WEST 186) entering the fit twice, because ESPN files it as
`season_type == 2`.

## Revisit Triggers

- ~~**When either league is fitted**~~ — attempted 2026-09-21; both failed
  their gates and neither moved. The next attempt should rate WITHIN season
  and walk forward, which is what both results point at.
- **NHL needs a joint distribution that admits negative correlation.** The
  shared-component Poisson cannot express it. Until then hockey's closeness
  is unmodelled in a measurable, one-directional way.
- **If a constant appears in either file**, the guard fails — and the right
  response is to ask what measured it, not to relax the guard.
- **If NHL measures overdispersed**, as MLB did, the Poisson family choice in
  this record is wrong and the package needs the negative-binomial one.
