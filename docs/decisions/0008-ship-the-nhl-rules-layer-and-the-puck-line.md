---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: league-nhl-nba-structure
---

# 0008. Ship the NHL rules layer, and release the puck line

## Context and Problem Statement

ADR 0007 established that an NHL final score is a scoring process with two
league rules stacked on top — a goalie pull that fires conditional on the
score, and a tie-break that awards exactly one goal — and that the repair is
to model the rules rather than fit a correlation. It specified the repair and
did not build it. The puck line stayed withheld.

This record is the build, the grade, and one defect found in already-shipped
work along the way.

## Decision Drivers

- The shipped model assigned about a sixth of its probability to a tied final
  score, which the league does not permit. That is not a subtle mispricing.
- The pull layer needed a holdout, and the two NHL seasons that existed were
  both spent. ADR 0007's new data is what made the question askable.
- A large result is a reason to decompose it, not to report it.

## Considered Options

- **Fit a hazard over the closing minutes.** Correct in principle and it
  needs the timing model this data can support but this holdout cannot grade.
  Deferred, and named in the artifact as deferred.
- **A per-game table conditional on the no-pull margin.** Coarser, estimable
  per season, and enough to move mass across 1.5. Chosen.

## Decision Outcome

`GoaliePullLayer`, `OvertimeLayer` and `NHLFinalScoreDistribution` in
`src/coverline/leagues/nhl/rules.py`. Tune on 2016-2021, graded **once** on
2022-2023, both fixed before running. Zero hyperparameter trials: `k` is
inherited from the already-graded rating scheme, the pull table is an
empirical frequency table with no smoothing constant, and the overtime split
is one measured proportion.

| comparison | gain | t |
|---|---|---|
| overtime rule alone | +0.173 | **+39.03** |
| both layers | +0.211 | **+39.99** |
| goalie-pull layer, over and above the overtime rule | +0.038 | **+6.75** |

**82% of the gain is the overtime rule**, which is a fact anyone can look up.
Reporting +39.99 as the layer's achievement would be dishonest; the modelled
part is the +6.75, and it clears on its own. The decomposition is in the
artifact and asserted in `tests/core/test_nhl_rules.py` so the headline cannot
be read alone.

The composed model reproduces the non-monotonicity that motivated all of
this — 0.236 at a three-goal margin against 0.194 at two, holdout actuals
0.242 and 0.187 — and puts exactly zero mass on a tie.

**The puck line is released into `primary_markets`.** That is permission to
price it, not a claim of edge: the layer was graded on the likelihood of
realised scorelines and has never been compared to a book. Moneyline and total
stand on exactly the same footing, which is why they are in the same list.

**Home advantage does not survive the tie-break.** Home teams win 54.94% of
games decided in regulation and 50.65% of those decided after it — 50.28% in
overtime, 51.34% in the shootout. A model carrying its regulation home edge
through the tie-break overprices every home moneyline by roughly the tie
probability times the difference.

## Consequences

**A lookahead defect was found in the already-shipped rates fit, and
corrected.** `walk_forward` computed its league base rate as the mean over the
whole frame — including games not yet played — so every prediction started
from a base that already knew how much the season would score. On the 2024
holdout the grade was **t = 3.02 with it and t = 2.44 without**. The fit still
clears, and it was inflated by about a fifth by a line that looked like
bookkeeping. `data/nhl_fitted.json` now carries the corrected figure.

Recomputing a grade after fixing a defect is not a second look at the holdout;
it is the same look with a corrected estimator. What would be illegitimate is
trying variants until one passes, so both numbers are recorded and only one
correction was made.

**The composition refuses final-score rates.** The rates in
`data/nhl_fitted.json` already contain the empty-net goals the pull layer
adds. Composing with them double-counts, the result looks entirely plausible,
and nothing but an explicit refusal catches it. `GameFeatures` carries no-pull
rates as separate fields and `NHLModel.predict` raises rather than
substituting.

**Two sources now validate each other.** 2024 and 2025 exist in both the
league API and sportsdataverse and agree on all 2,624 games, with zero score
mismatches. The 2021-2023 corruption is therefore a bounded defect in specific
files, not a reason to distrust the vendor wholesale — and it is equally a
check on the API pull, which nothing else here validates.

**2024 and 2025 were deliberately kept out of the grade.** They were pulled
afterwards, to produce current ratings. Folding them into the holdout after
seeing the result would be enlarging a test set because the answer was liked,
and the answer here was liked very much.

## Recovery

`model/ingest/` refetches everything; `data/raw/nhl/` retains 12,200 games and
71,000 goals, tracked. `model/fit_nhl_rules.py` re-grades and
`model/export_fitted.py` rebuilds `data/nhl_rules.json`. Removing that file
returns `primary_markets` to moneyline and total, which is the state before
this record.

## Revisit Triggers

- **The pull layer is drifting and the table should be re-measured per
  season.** Empty-net goals have gone from 0.241 a game in 2017 to 0.399 in
  2025, and the average one arrives about twenty seconds earlier in the third
  period. The table graded here is 2016-2021 behaviour applied to a league
  that has moved.
- **`test_the_margin_distribution_is_non_monotone` fails.** The layer has
  stopped doing the one thing it was built for.
- **The goalie-pull component stops clearing on its own.** The headline grade
  is then the overtime rule and must be reported as that.
- **Closing puck-line prices become available.** Nothing here has been
  measured against a book. Everything above is a better description of NHL
  scorelines and says nothing yet about edge.
- **A timing model becomes gradeable.** The per-game table cannot tell a pull
  with ninety seconds left from one with three minutes left, and it conditions
  on the no-pull margin rather than the score a coach actually sees.
