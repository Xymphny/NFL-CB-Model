---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: execution-recommend
---

# 0015. The away side and the total were both mispriced

## Context and Problem Statement

Adding `"puck_line"` to the NHL's `primary_markets` raised a question nobody
had asked: does anything in production *read* `primary_markets`? It does not.
Only tests did. The declared market list was documentation, and the recommender
took a market string, matched it against the vendor key on a quote, and never
asked the league whether it priced that market.

Worse, the two halves speak different languages. Leagues say `"spread"`,
`"moneyline"`, `"total"`, `"runline"`, `"puck_line"`. Quotes arrive as
`"spreads"`, `"h2h"`, `"totals"`. Nothing translated, so wiring the recommender
to a league's own list would have matched zero quotes.

Writing the test that priced a puck line end to end — the first test in this
repository to price **both sides of a market and compare them** — found two
pricing bugs instead.

## Decision Drivers

- Both bugs are in the path that decides what to bet and how much.
- Neither was visible to any existing test, and the reason is structural:
  every test priced one side and checked it against a number computed the same
  way the code computes it.

## Considered Options

- **Fix the puck-line test and move on.** The complementarity failure looked
  at first like a bad assertion.
- **Treat the failing property as the finding.** Chosen.

## Decision Outcome

**Bug 1: the away side was priced with the home team's answer.**
`cover_probability` always answers for the home team at the line it is given,
and `price_candidate` handed it whichever line sat on the quote being priced.
On a home favourite at −6.5 the away side returned **0.5314 against a truth of
0.8214**, and the two sides of a no-push market summed to **1.0849**.

The error flips sign with the line, so it is not a constant bias that a losing
record would eventually expose — it inflated the away side on some games and
deflated it on others. **Every away-side handicap and moneyline price in every
league was wrong.**

The fix reads which side is being priced off the quote's own team names,
converts that side's line to a home line, and complements. Guessing the side —
by position, or by assuming the first quote is home — would reintroduce it
silently when a team is renamed, so an unrecognised outcome name now raises.

**Bug 2: a total was priced off the margin.** Every market went through
`cover_probability`, which reads `margin_cdf`. An Over 44.5 was priced as
`P(margin > −44.5)` and came back **0.9982 against a truth of 0.4801**. MLB is
the only league that currently offers totals, and it would have bet every Over
at maximum stake. `total_probability` now exists and is used for Over/Under.

**Bug 3, found while fixing them: the push guard refused legitimate markets.**
`_can_price_push` demanded a key-number correction at any integer line. That
correction exists because a *rounded normal* understates `P(margin = 3)` by
nearly threefold — a count distribution has no such problem, because its pmf
*is* the atom. The guard was withholding every integer total in baseball and
hockey. `pmf_is_exact` now distinguishes the two, and the same function was
also negating a total's line, asking about −44 instead of 44.

**And `core/markets.py` now bridges the two vocabularies**, with
`require_offered` refusing a market a league does not offer. `recommend()`
takes `primary_markets` optionally — a required parameter is how a safety
feature gets reverted — and enforces it when given. The optionality is stated
in the docstring rather than left to be discovered.

**`push_probability` is now carried into the ledger.** It was computed, it
decided the price, and it was dropped on the way to the row. The quantity
responsible for the largest pricing error this project has found was not in
the record of the bets it priced.

## Consequences

Nothing has been bet through this code, so no money was lost. That is luck
about timing, not a mitigation: the Odds API subscription is days away and
these prices would have gone live with it.

**The lesson is about the shape of the tests, not the bugs.** Both errors
survived a suite that covers this module heavily, because every test asserted
agreement between the code and a hand-computed number derived the same way.
The properties that caught them — the two sides of a no-push market sum to
one; a total priced near the mean is not 0.998 — relate two outputs to each
other and need no oracle.

## Recovery

`git revert` restores the previous behaviour, which is worse. The three
property tests in `tests/core/test_recommend.py` fail immediately on any
reintroduction.

## Revisit Triggers

- **A market arrives whose two sides are not complementary.** Three-way
  markets exist; the complementarity property is specific to two-way markets
  after voiding outcomes are conditioned out, and a draw-inclusive market
  needs its own handling rather than this one.
- **An outcome name stops matching a team name.** The side determination
  raises, which is correct, and the fix is a normalisation layer rather than
  a fallback guess.
- **`primary_markets` gains a production consumer beyond `recommend()`.** The
  optionality of that parameter is a transitional compromise and should end.
