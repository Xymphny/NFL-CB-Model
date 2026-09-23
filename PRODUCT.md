# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **The owner (primary).** Builds and bets the model. Keeps the site open all
  day on a laptop and refreshes as news lands. Wants the reasoning, the
  evidence behind every number, and how the market is moving.
- **A small group of friends (secondary).** Want the picks, not the method.
  They use the same site; cards lead with the pick and the reasoning is one
  step away (confirmed: no separate mode, no owner-only gating).

## Product Purpose

Coinflip is an emotionless second opinion on sports betting markets across
five leagues (NFL, CFB, MLB, NHL, NBA). For each game it shows what the model
thinks against what the market thinks, so the owner can stand behind a pick or
see what else to weigh. It suggests; it never instructs.

Message: **"Every number on this screen was earned by evidence."**

Success: the owner trusts every figure because each one traces to a graded
artifact, and the site never claims more than the evidence supports.

## Positioning

The site renders the model's own numbers and computes none of its own, so it
can never disagree with the model. Stakes are sized only from a grade of the
model against closing prices; until a league earns one, every pick is shown
**unsized**, and the site says so rather than showing a dollar figure. Tiers
(Play / Lean / Coin flip / No edge) express conviction relative to the league's
own history, not claimed edge.

## Operating Context

- Static site on Render, built from `data/` files committed by cron jobs:
  `data/site/board_{league}.json` (per game: teams, start, market, line, best
  price and book, p_model, p_market devigged, edge, tier, the core's stake,
  line at open, per-sport context, refusals), `data/site/record.json`
  (settled paper trades vs a 150-game floor, CLV), `data/site/gates.json`
  (fitted-artifact grades, validation verdicts, market grades, decision
  records). Produced by `scripts/export_board.py` and `scripts/export_record.py`.
- Refreshed after every odds capture and twice daily; the owner reloads to see
  what changed since their last visit.
- Existing account features stay: Discord login, bankroll settings, Kelly
  sizing driven by the core's stake, bet log, exposure caps, personal CLV.

## Capabilities and Constraints

- Leagues and their slates: NFL weekly (spreads, key numbers 3 and 7, QB,
  weather); CFB weekly and large (conference/ranked filters, early-season Lean
  cap) -- currently has no live source and shows its refusal; MLB daily
  (probable starters are the headline, scratch alerts; moneyline, runline,
  total); NHL daily (moneyline, puck line; ratings reset each season, early
  cards are low-information); NBA daily (spread; rest and back-to-backs; the
  model is team-level with no injury or minutes layer).
- Views hide what does not apply to a sport rather than rendering empty.
- The frontend must contain no pricing, probability, tier or staking logic.
- Every data guard and test in the repository is preserved.
- Terminology: Play, Lean, Coin flip, No edge (tiers); "Unsized -- not yet
  graded against the market"; market grade; paper trades; CLV; refusal.
- The product name Coinflip shares a word with the Coin flip tier; the two
  must stay visually and verbally distinct.

## Brand Commitments

- Name: **Coinflip** (replaces Coverline and the football/chalkboard identity).
- Voice: calm, clinical, trustworthy -- an instrument panel, not a sportsbook.
  No hype, no urgency, no "lock of the day".
- Each sport has a restrained accent used for identification only.

## Evidence on Hand

- Market grades: NFL and CFB show no measured edge over closing prices
  (ADR 0024); MLB, NHL and NBA are ungraded. Every league is unsized today.
- Tier bands are provisional; none has been distinguishable from break-even
  (ADR 0025).
- No testimonials, performance claims or profit figures exist and none may be
  shown beyond what `record.json` reports.

## Product Principles

1. Every number traces to an artifact; the site renders, it never computes.
2. Suggest, never instruct: no urgency, no certainty the evidence does not carry.
3. A refusal is information: say what could not be priced and why, in plain words.
4. Conviction and money are different things, shown differently.
5. Show what changed since the last look.

## Accessibility & Inclusion

Status is never conveyed by colour alone. Tabular numerals for all figures.
Keyboard-friendly. Dark mode primary; responsive down to phone as a fallback.
