---
status: accepted
date: 2026-09-25
supersedes: null
superseded_by: null
ledger_id: market-relative-weight
---

# 0030. Grade NBA, NHL and MLB against free ESPN closes

## Context and Problem Statement

The stake rule (ADR 0024) sizes a bet from each league's grade against
closing prices. NFL and CFB have historical lines from free sources. MLB,
NHL and NBA had none:

- The Sportsbook Reviews archive now redirects to its homepage.
- The Odds API historical backfill costs about 90,000 credits for one
  season, which is 4.5 months of the current plan.

So those three leagues had no grade. MLB, whose season ends on Sep 27,
would have stayed ungraded until about May 2027.

ESPN's public odds records keep **open and close** blocks per game: each
side's moneyline, spread (points and price), and the total. They start in
late 2023, first with several books, then ESPN BET, then DraftKings in
2026.

## Considered Options

- Pay for the backfill: one season, about 90,000 credits.
- Wait for 150 settled paper trades per league. That takes until November
  for NBA and NHL, and May 2027 for MLB.
- Pull ESPN's records for free. **Chosen.**

## Decision Outcome

`model/ingest/espn_closes.py` pulls regular-season closes into
`data/raw/{nba,nhl,mlb}/espn_closes_{season}.parquet` and records the book
for each game. In-play feeds are never used. When several books have a
close, a record with **both** a closing moneyline and a closing spread is
preferred over the preferred book with only one.

`model/grade_market_weight.py` grades each league through its live
model, one row per game from the home side:

- NBA spread: `NBALiveSource`, replayed one minute before tip. Integer
  lines are left out, as the live system refuses them (ADR 0022); ESPN's
  NBA lines are 99.8% half points.
- NHL moneyline: `NHLLiveSource` with the pull and overtime layers.
- MLB moneyline: the walk-forward with the starters who actually started.

The graded rows are kept in `data/market_grade/`. The test re-grades them
and re-prices a sample through the live paths.

| League | Games | Seasons | w_hat | SE | Stake |
|---|---|---|---|---|---|
| NBA | 3,669 | 2024–26 | −0.019 | 0.080 | 0 |
| NHL | 3,908 | 2024–26 | −0.012 | 0.092 | 0 |
| MLB | 8,326 | 2023–26 | +0.148 | 0.091 | 0 |

NBA and NHL are firm zeros: given the close, their models add nothing.

MLB comes closest to a stake: the one-sided lower bound is −0.002.

- **2026**, the only fully clean season, gives +0.24 ± 0.19.
- **Dropping 2023** would give a stake of 0.031. 2023 is the thin season:
  1,121 of its games have no closing moneyline from any book. The
  missingness depends on which book ESPN attached to each game, not on
  outcomes.
- **The season range was set before any result was seen, and it stands.**
  Choosing seasons after seeing a result is how a grade gets manufactured.

### Tier bands

NBA, NHL and MLB now take their bands from their own history instead of
borrowing another league's. Each band now also records `market_said`: the
mean devigged closing probability of the side the model preferred.

On a moneyline the model's strongest leans are mostly underdogs, so a
hit rate well under 50% says nothing by itself. MLB's Play band won 37.4%
against a price of 37.0%. NHL's won 32.0% against 33.7%.

**One band of twenty beats its price by more than 2 SE**: MLB Coin flip,
48.0% won against 44.9% priced (z = +3.13, n = 2,507). This is recorded,
not acted on, for four reasons:

- It is not monotone: MLB's Lean and Play bands sit at +1.16 and +0.22.
- It is one comparison among twenty.
- It is measured against a devigged price, and about 2 points of hold
  would leave roughly +0.8 points.
- Its seasons are partly contaminated.

ADR 0025's claim that no band is distinguishable from its market is
amended to this. A test pins the exact set of bands above 2 SE.

## Consequences

- Every league now has a market grade, and each is still stake 0.
  Paper trading continues; the ledger replaces each grade at 150 settled
  games.
- The Gates page compares each band's win rate with its market price, not
  with the −110 break-even alone.
- The paid backfill is no longer needed for grading.

## Recovery

Delete `data/market_grade/*` and drop `nba`, `nhl` and `mlb` from
`grade_market_weight.LEAGUES`, then re-run it and
`derive_tier_thresholds.py`. The leagues return to "no grade" (weight 0)
and borrowed bands. The raw closes are inert without the loaders.

## Revisit Triggers

- MLB's lower bound turns positive with more clean seasons: the 2026 close
  and 2027 on. That would be the first earned stake, and it needs its own
  ADR.
- The set of bands above 2 SE changes. The test fails and says so.
- ESPN stops publishing close blocks, or changes provider. Grades name
  their books, so a change shows up in the next rebuild.
