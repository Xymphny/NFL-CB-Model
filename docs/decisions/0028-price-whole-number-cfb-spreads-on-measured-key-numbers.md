---
status: accepted
date: 2026-09-25
supersedes: null
superseded_by: null
ledger_id: league-cfb-live
---

# 0028. Price whole-number CFB spreads on measured key numbers

## Context and Problem Statement

ADR 0027 kept every whole-number CFB spread withheld, because CFB had no
measured key-number table. The recommender refuses to price an integer
line with a plain rounded normal (ADR 0022). That normal understates the
chance of a push on a key number, which inflates the apparent edge. In
the CFBD closing lines, 45% of spreads are whole numbers, so nearly half
the CFB slate was refused.

## Considered Options

1. Keep refusing whole-number lines.
2. Reuse NFL's table (`data/nfl_key_numbers.json`).
3. Measure a CFB table with NFL's method, and grade it once on a held-out
   season.

## Decision Outcome

Option 3. Option 2 is wrong on its face: CFB margins are wider, and
overtime makes a tie impossible.

The table is `data/cfb_key_numbers.json`, built by
`model/cfb_key_numbers.py`. The method is the NFL one: a multiplicative
weight per margin, applied to a rounded normal centred on each game's
closing spread, then renormalised. It is fitted on 2021–2022 closes
(2,262 games) and graded once on 2023 (1,347 games).

Mean held-out log-likelihood improves in two places:

- By +0.133 per game (t = 9.2) at the market's residual width of 15.36.
- By +0.135 per game (t = 9.3) at the width the model actually prices
  with, 17.80.

The table must pass at both widths to count as supported, so a pass at a
convenient width cannot mis-sell it.

Push probabilities on the 2023 holdout:

| Margin | Plain rounded normal | With the table | Actually observed |
|---|---|---|---|
| 3 | 1.7% | 5.2% | 5.7% |
| 7 | 1.7% | 4.2% | 4.2% |
| 0 | 1.7% | 0 | 0 |

`CFBModel` loads the table only if the artifact says it cleared its gate.
A test pins that an ungraded table is refused.

## Consequences

- Whole-number CFB spreads now price and paper-trade.
- The other CFB refusals still stand: neutral site, unrated team, stale
  ratings, and a snapshot taken after kickoff.
- `margin_mean()` on a CFB distribution is now the reweighted pmf's mean,
  as it already was for NFL. Renormalising pulls it toward zero by up to
  about one point. The board's model line therefore moves slightly toward
  zero. The legacy parity test now compares the linear predictor
  (`mu_margin`).
- CFB cover probabilities changed, so the CFB tier bands were
  re-derived: coin_flip moved from 0.103 to 0.116, lean from 0.205 to
  0.217, play from 0.307 to 0.315. NFL and MLB were unchanged, with the
  MLB cut-off still pinned at 2026-09-21.
- CFB's market-weight grade (`data/market_weights.json`, ADR 0024) was
  re-run on the new probabilities: w_hat 0.072 → 0.077, se 0.064 → 0.062.
  The staking weight stays **0**, and CFB still paper-trades.
- The Record page lists the table under fitted gates.

## Recovery

This is a one-line revert. Set `KEY_NUMBER_WEIGHTS = None` in
`src/coverline/leagues/cfb/model.py`, or delete the artifact. Integer
lines are then refused again, as before. Re-derive the tier bands
afterwards.

## Revisit Triggers

- CFB's own settled paper trades on integer lines push noticeably more
  or less often than the table predicts. After 150 trades, compare the
  observed push rate on 3 and 7 with the table.
- A new holdout season becomes available (2024 or 2025 closes in
  `model/cfb_lines_cache.csv`). A refit spends that season, and the
  refit has to be recorded here.
- A rule change affects scoring, for example on overtime or
  two-point conversions.
