---
status: accepted
date: 2026-09-23
supersedes: null
superseded_by: null
ledger_id: dashboard-board-export
---

# 0025. Tiers are conviction, set by each league's own history, not edge

## Context and Problem Statement

The dashboard revamp gives every game one of four states: Play, Lean, Coin
flip, No edge. The old board set these in the frontend from a points gap
(`PLAY_GAP`, `LEAN_GAP` in `frontend/src/staking.js`) and priced the game
itself with its own sigma. The brief requires the core to define tiers, the
site to compute nothing, and the thresholds to be derived from the existing
backtests, labelled provisional, and replaced by ledger-calibrated values at
150 settled games.

Deriving them forced the question of what a tier can honestly mean. Measured
against closing lines, the side the model prefers wins about 51% of the time
**in every band of disagreement**:

| league | games | no edge | coin flip | lean | play |
|---|---|---|---|---|---|
| NFL | 1,884 | 50.8% | 49.4% | 51.6% | 52.9% |
| CFB | 1,704 | 50.7% | 51.6% | 49.9% | 52.6% |
| MLB | 87 | 34.3% | 42.3% | 41.2% | 44.4% |

No band is distinguishable from the 52.4% break-even. The top band sits
nominally above it in both football leagues -- NFL 52.9% on 189 games, CFB
52.6% on 171 -- by 0.15 and 0.07 standard errors, which is no evidence at all;
every other band is at or below it, and there is no gradient worth the name.
Bigger disagreement has not meant a better result.

## Considered Options

- **Fixed probability-point cutoffs across leagues.** The leagues disagree
  with their markets by very different amounts (median |edge| ~10 points in
  the NFL, ~13 in CFB, ~3 in MLB), so one cutoff calls half of one league a
  Play and almost none of another.
- **Cutoffs chosen where the hit rate clears break-even.** There is no such
  place; choosing one anyway would be fitting noise and calling it a Play.
- **Quantiles of each league's own disagreement.** Chosen.

## Decision Outcome

`model/derive_tier_thresholds.py` writes `data/tier_thresholds.json`: per
league, the 40th, 70th and 90th percentiles of |p_model − p_market|.
`core/tiers.py` maps a game's edge to a tier. Play is the league's strongest
tenth of disagreements, Lean the next fifth, Coin flip the next three tenths,
No edge the bottom four tenths.

- NFL and CFB bands come from the walk-forward backtests against closing
  spreads.
- MLB's bands come from about 87 games of odds-watch snapshots against the
  walk-forward, and are rough.
- NHL borrows MLB's moneyline bands and NBA borrows NFL's spread bands,
  because neither has any market history. Both are marked `borrowed`.

Every league's bands are replaced by the same quantiles over its own settled
paper trades once it has 150 (`source: ledger`).

**A tier never implies a stake.** Stake comes from the market grade (ADR 0024),
which is 0 everywhere today. The site shows the tier and "Unsized — not yet
graded against the market".

## Consequences

- The site loses its pricing code entirely; the core ships the tier.
- "Play" now means the same fraction of every league's games, which is what a
  conviction scale should mean, and says nothing about profit, which is right.
- The per-band hit rates travel in the artifact, so the claim that tiers are
  not edge stays checkable as the ledger grows.

## Recovery

The old gap thresholds are `PLAY_GAP` and `LEAN_GAP` in `frontend/src/staking.js`
at commit `5973d7c`. They are superseded rather than lost.

## Revisit Triggers

- **A league reaches 150 settled paper trades.** Its bands switch to its own
  ledger. If the ledger shows a real gradient, which none of the backtests do,
  the tiers may become worth more than conviction, and that is a new record.
- **NHL or NBA gains market history.** Its borrowed bands are replaced.
- **The market grade for a league clears zero.** Tier and stake then start to
  mean related things, and the site's copy should say so.
