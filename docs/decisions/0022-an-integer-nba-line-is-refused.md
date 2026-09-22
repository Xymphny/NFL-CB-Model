---
status: accepted
date: 2026-09-22
supersedes: null
superseded_by: null
ledger_id: league-nhl-nba-structure
---

# 0022. An integer NBA line is refused, not answered with a push of zero

## Context and Problem Statement

`NormalMarginDistribution`'s docstring said NBA margins are integers "but the
modelling convention there treats them as continuous because the support is
wide enough that atom mass at any single value is small."

Nobody had measured "small". And `discrete=False` does not encode *small* — it
makes `margin_pmf` return **zero**, which encodes *impossible*.
`_can_price_push` waved every continuous distribution through, so an integer
NBA spread was priced with `p_push = 0.0` and a cover probability that
conditioned nothing out. NFL and CFB **refuse** an integer line without a
measured key-number correction. NBA answered one.

## Decision Drivers

- The same failure shape as ADR 0011: a confident wrong number where a
  refusal belongs.
- It is a live pricing path, and market prices arrive this week.

## Considered Options

- **Measure a key-number table and ship it, as the NFL has.** Needs a holdout,
  and the ratios are mostly noise (below).
- **Make NBA `discrete=True` with a rounded normal.** Worse: it would put
  2.585% on a tie that overtime forbids.
- **Refuse integer lines for want of a measured correction.** Chosen, and it is
  exactly what CFB already does.

## Decision Outcome

`NormalMarginDistribution` gains `integral_margin`, default False. When a
distribution is continuous *and* admits its margin is integral,
`_can_price_push` requires a key-number correction before pricing an integer
line. NBA passes `integral_margin=True`.

**Measured over 3,540 walk-forward games** (`model/nba_atoms.json`):

| | |
|---|---|
| largest empirical atom (modal spread) | **3.3%** |
| NFL's atom at a margin of 3, for scale | 8% |
| empirical rate of a tied final score | **0.000** |
| what the rounded normal would say | **2.585%** |
| ratio at +1 / −1 | **0.75 / 0.81** |

A tie is impossible — overtime resolves every game — which is the third league
to be caught pricing one, after ADR 0007 in hockey and ADR 0017 in baseball,
and the first where the margin is modelled continuously.

The plus-and-minus-one **depletion** is the opposite of hockey. There the
tie-break awards exactly one goal and piles mass onto ±1; an NBA overtime is a
full five minutes and scatters it.

**No key-number table is claimed.** About 90 games per margin value gives a
ratio standard error near 0.10, so the values between 1.0 and 1.2 are noise and
only the tie and the ±1 depletion sit outside it. Fitting a table to that would
be fitting noise, and no NBA season is pristine to grade one on — 2021-2022
tuned the ratings, 2023 graded them, 2024-2025 were spent on the static
question.

**Half-point lines are unaffected**, which is most of the board, and a test
asserts it so the refusal is not collateral damage.

## Consequences

`model/measure_nba_atoms.py` records the measurement and the reason no table
follows from it. The `core` docstring no longer asserts that atom mass is small
without saying how small.

**A ledger row that contradicted its own ADRs is corrected in the same change.**
`league-nhl-nba-structure` still read "STRUCTURE ONLY, NO FITTED COEFFICIENTS…
no held-out grading" after ADRs 0006, 0008 and 0010 had shipped graded
parameters for both leagues. `migration/ledger.yaml` is the reverse index that
keeps this directory honest; on that row it had stopped, and a stale index is
worse than none because it reads as current.

## Recovery

Removing `integral_margin` from the NBA constructor restores the previous
behaviour, which prices integer spreads with a push of zero.

## Revisit Triggers

- **Most margins move outside sampling noise.** A measured table becomes
  defensible, and it needs a holdout NBA does not currently have.
- **A pristine NBA season appears.** Then the table can be graded rather than
  guessed, and integer lines can come back.
- **Half-point pricing breaks.** The refusal is meant to be narrow and a test
  holds it there.
- **Another league models an integral margin continuously.** The flag is
  opt-in, so it would be silently unprotected.
