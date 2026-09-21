---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-nhl-nba-structure
---

# 0007. The NHL joint distribution is a rules problem, not a correlation

## Context and Problem Statement

ADR 0006 shipped NHL rates and withheld the puck line, giving this reason:

> Independent Poisson under-predicts one-goal games by about ten points —
> 28.4% against an actual 38.1% — because the two scores are negatively
> correlated at −0.14 and `BivariatePoissonDistribution`'s shared component
> can only express positive correlation.

The observation is right. The **explanation is wrong**, and it was written
into the module docstring, the fitted artifact, and that ADR, where it pointed
at a specific next step: find a joint distribution admitting negative
correlation. That step would have produced a fitted parameter that reproduces
a moment and misses the shape.

It is wrong by its own algebra. Negative covariance makes margins **more**
dispersed, not less:

    Var(H − A) = Var(H) + Var(A) − 2·Cov(H, A)

So "real games stay closer than independent rates allow" predicts **fewer**
one-goal games than independence, which is the opposite of the problem it was
offered to explain. Measured over 9,576 games, the margin variance is 6.280
against an independent-Poisson 5.944 — a dispersion ratio of **1.06**, above
one, exactly as the algebra requires.

The correlation itself was not invented: it is real and it is negative. It is
also **smaller than recorded and not stationary**, moving from −0.055 in 2017
to −0.142 in 2023. A single fitted value describes no season.

## Decision Drivers

- A wrong mechanism with a right observation is worse than an open question,
  because it closes the question and aims the repair at the wrong object.
- Only two NHL seasons were ever usable and both are spent. Anything that
  needs a holdout needs new data first.
- The puck line is fixed at 1.5, so the **shape** of the goal distribution
  does work the mean cannot. Two models agreeing on the moneyline can
  disagree here.

## Considered Options

- **Fit a copula with a negative dependence parameter.** General, standard,
  and fits a drifting number to a shape it cannot produce — a copula on
  Poisson marginals still puts mass on a tie, and NHL final scores never tie.
- **Diagonal inflation.** The textbook football fix. Adds tie mass, which is
  the exact opposite of what hockey needs.
- **Measure what an NHL final score actually is, and model the rules.**
  Chosen.

## Decision Outcome

Model the two **league rules** that produce the deviation. Neither is a
correlation and neither is fitted.

**The overtime rule is deterministic.** No NHL game ends tied, and all 2,166
games decided after regulation across 2016-2023 end at a margin of **exactly
one** — no exceptions in 9,576 games. About 22% of the distribution's tie mass
is relocated onto ±1 by fiat. This is the larger part of the ten points, and
no amount of fitting reaches it.

**Empty-net goals are conditional on the score.** 92% arrive with the scoring
team leading by one or two. A team down two pulls its goalie and rarely comes
back; a team down one pulls and often ties, vanishing into overtime and back
out at one. The result is a **non-monotone** margin distribution — more
three-goal games than two-goal games (22.7% against 20.3% in the league's own
data, 25.1% against 20.2% in a second source) — across exactly the 1.5 line
the puck line is priced on. No independent Poisson produces this at any rates.

**Stripping empty-net goals restores monotonicity.** That is what makes this
the mechanism rather than a coincidence, and it is asserted in
`tests/core/test_nhl_joint_structure.py` rather than left in prose.

## Consequences

**The puck line stays withheld**, for a better-understood reason. The repair
is now specified — a regulation distribution, an empty-net layer keyed to
score state and time, and a deterministic overtime layer — rather than
gestured at.

**New data was acquired rather than a holdout reused.** 9,576 games and 56,834
goals from `api-web.nhle.com` for 2016-2023, from a different pipeline than
the corrupt sportsdataverse files, and never graded on. That is what makes a
third NHL question askable at all; ADR 0006 correctly recorded that nothing
clean was left.

**Nothing here spends a season.** `model/measure_nhl_joint.py` is descriptive.
There is no prediction in it, so there is no holdout in it.

**Two figures in ADR 0006 and the shipped docstring are corrected**, not
quietly: empty-net goals are 5.15% of goals and 0.306 a game over 2016-2023,
not the 7% and 0.42 guessed there. And "about 57% of games are one-goal games"
is 42.6% as played — 55.5% once empty-net goals are stripped, which is the
convention that number came from. Both are defensible and they are not the
same quantity, which is the kind of ambiguity that silently changes a price.

**A cross-check reversed its own verdict.** The league's empty-net flag was
checked against the goalie-presence digits in `situationCode`. They agree on
56,821 of 56,834 goals, and the thirteen that disagree are the **code** being
wrong: filed empty-net at 15:04 to 19:53 of the third with a code reading both
goalies on the ice. The check validated the field it was built to doubt and
found errors in the field it was built to trust.

## Recovery

`model/ingest/nhl_api.py` and `model/ingest/nhl_goals.py` refetch everything
from the league API; `data/raw/nhl/` retains it, tracked in git. Delete
`model/nhl_joint_structure.json` and `model/measure_nhl_joint.py` rebuilds it.
Nothing in this ADR depends on state that exists only on one machine.

## Revisit Triggers

- **The dispersion ratio drops below one.** The original "games stay closer"
  story becomes arguable again, and `test_margins_are_over_dispersed_not_under`
  is the place it will surface.
- **The correlation stabilises.** A fitted correlation becomes defensible if
  it stops drifting; `test_the_correlation_is_drifting` fails when it does.
- **The flag and the situation code diverge further.** The empty-net layer
  should not be fitted while it is unknown which field is right.
- **Market puck-line prices are captured.** None of this has been tested
  against a book. The shape argument says a win-probability model is
  insufficient here; that is an argument, not a measurement, until there are
  closing prices to measure against.
