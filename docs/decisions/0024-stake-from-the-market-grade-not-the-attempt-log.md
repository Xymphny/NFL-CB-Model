---
status: accepted
date: 2026-09-22
supersedes: null
superseded_by: null
ledger_id: market-relative-weight
---

# 0024. Stake from a grade against the market, not from the attempt log

## Context and Problem Statement

Staking blends the model toward the market before sizing a bet:
`p_used = w * p_model + (1 - w) * p_market`. For NFL and CFB the operator
command took `w` from `evidence/attempts.yaml` — the robust empirical-Bayes
weight, 0.9141.

Every attempt in that log is graded **model against model**: does a change
beat the version before it on held-out games. That is the right evidence for
"is the model improving" and the wrong evidence for `w`, which asks a
different question — given the closing price, does the model add anything?
A model can improve steadily and add nothing to a closing line. When that is
so, any `w` above zero turns disagreement with the market into a claimed edge,
and Kelly sizes it.

An earlier audit had already found the symptom: regressed against the spread,
the NFL margin's slope is −0.064 (SE 0.072) over 1,885 games, and ATS at a
gap of 2.5 points or more was 50.7% against a 52.38% break-even.

## Considered Options

- **Keep the attempt-log weight and add a warning.** The warning shipped with
  the five-league runner. A warning next to a 0.91 stake is still a 0.91 stake.
- **Set football to zero by hand.** Right today and unmeasured tomorrow; a
  constant that says 0 is as ungrounded as one that says 0.91.
- **Measure `w` against the market and stake its lower bound.** Chosen.

## Decision Outcome

`core/market_weight.py` estimates `w` by maximum likelihood under exactly the
blend staking applies, over settled games the model priced before they were
played, priced through the production path (`LeagueModel` →
`cover_probability`, closing prices devigged with the runner's method). The
staking weight is the one-sided 95% lower bound, clipped to [0, 1]: zero
unless the data rule zero out. `model/grade_market_weight.py` writes
`data/market_weights.json`; the runner reads it for every league, and a league
with no row gets 0.

Measured 2026-09-22:

| league | bets | seasons | w_hat | SE | staking | side hit rate |
|---|---|---|---|---|---|---|
| NFL | 1,884 | 2014–2023 | +0.039 | 0.082 | 0.000 | 50.74% |
| CFB | 1,704 | 2021–2023 | +0.072 | 0.064 | 0.000 | 51.00% |

Both are **upper bounds**. The NFL rating-only coefficients' fit window is not
recorded, and no contiguous range of the walk-forward cache reproduces them;
the CFB vector was fitted on the graded seasons. In-sample predictions can only
make the model look better against the market, so the truth is at or below
these figures. No NFL season on its own is distinguishable from zero
(−0.34 to +0.47, SE 0.22–0.30).

CFB's closing lines carry no prices, so its market side is taken as 0.5 (a
standard −110/−110 line). That is recorded in the artifact as an assumption.

## Consequences

- **Every league paper-trades.** The runner still prices every game and, with
  `--commit`, records model and market probabilities for every candidate. That
  ledger is how a future positive grade gets earned.
- **The ledger could not have earned one, and now can.** Building this found
  that closes attached only to placed bets, outcomes existed for no one, and
  unplaced rows dropped their book, price and game -- so paper trades could
  never be graded. Paper rows now keep all three plus their side;
  `scripts/settle_ledger.py` closes and settles them; and
  `model/grade_market_weight.py` grades a league from its own ledger once 150
  games have settled, one row per game. That grade is clean by construction:
  every probability was written before its game, against a price that existed.
- `--pooled` is refused with this record's number: it chose between two
  attempt-log weights and there is no longer anything for it to choose.
  `--market-weight` remains the explicit override and says it is a guess.
- The attempt log is untouched and still printed. It measures what it always
  measured.
- MLB, NHL and NBA stay at zero for lack of data rather than by measurement.
  The free MLB archive (`ingest/mlb_odds_archive.py`) now redirects every
  season to its homepage, and there is no free NHL or NBA line history. The
  capture job records closing lines from the Odds API subscription onward;
  once a season of settled bets exists they get a row here.
- A test pins football at zero. A refit that moves it fails the test and
  reopens this record, rather than starting to stake unnoticed.

## Recovery

Nothing was dropped. The previous behaviour is `scripts/recommend_slate.py`
at commit `f503418` (`choose_weight` reading `E.load_attempts()`), and the
attempt log and its weights are unchanged in `src/coverline/core/evidence.py`.
Restoring it is a one-function revert; the reason not to is the table above.

## Revisit Triggers

- **A league gains a season of settled closing prices** from the capture job.
  Grade it; if its lower bound clears zero it stakes, and this record gains a
  row rather than being superseded.
- **The NFL or CFB coefficients are refit** with a recorded fit window. The
  seasons after that window are the first clean grade either league has had,
  and they replace the upper bounds here.
- **`test_football_has_no_measured_edge_over_the_close` fails.** That means a
  refit or new data moved a football weight off zero. It is a decision to make
  on purpose, not a test to update.
- **Staking changes its blend** away from `w * p_model + (1 - w) * p_market`.
  The estimator measures that exact blend and nothing else.
