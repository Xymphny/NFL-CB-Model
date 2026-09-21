---
status: accepted
date: 2026-09-21
supersedes: null
superseded_by: null
ledger_id: ngs-team-code-mismatch
---

# 0004. Record the NGS team-code mismatch; do not fix it in the same breath

## Context and Problem Statement

While porting the NGS feature fetch, the new core's output disagreed with the
published board: the board ran the full ensemble on 6 of 16 week-2 games, but
recomputing the NGS frame showed all 32 teams present. Chasing that difference
found the cause.

The ratings snapshot and the nflverse schedule call the Rams **`LA`**. The
nflverse Next Gen Stats release calls them **`LAR`**. `build_week_predictions`
decides, per game:

    ngs_present = home in ngs_features.index and away in ngs_features.index

So every Rams game silently falls back to `MARGIN_COEFFICIENTS_V1_RATING_ONLY`.
That vector carries no `elo_diff` term, so those games lose Elo as well as the
NGS block.

Measured: **17 games per season in every season from 2022 to 2026 — 85 games
across the committed range.** It presents as "NGS is not available for this
game", which is indistinguishable from the feed being down, and nothing
recorded the difference.

## Considered Options

- **Fix the mapping now.** One line. Obviously correct on its face.
- **Record and guard, fix as a measured change.** Slower; keeps the model
  change separate from the discovery.
- **Normalise only inside the new core.** Makes the new core correct and
  deliberately diverges from the board, which breaks the parity property the
  migration depends on.

## Decision Outcome

Chosen: **record and guard now; fix as its own gated change.**

The mapping fix looks unambiguous, and the reason to slow down is not doubt
about which code means the Rams. It is that the backtest which produced the
validated NGS improvement — straight-up 58.22% to 64.04% — ran against this
same mismatch. Every Rams game was in the rating-only bucket for that
measurement too. So the full ensemble's validated support does not include
Rams games, and moving 85 of them onto it is a model change, not a typo
correction.

This project's own rule is that a change ships on a held-out measurement, not
on looking obviously right. A fix that is obviously right is exactly the kind
that gets waved through without one.

`tests/core/test_team_codes.py` guards the class rather than the instance: it
compares source vocabularies directly, runs against committed artifacts with
no network, and fails loudly on the next join mismatch — a relocation, a
rebrand, a vendor changing convention mid-season.

## Consequences

Better: the mismatch is now visible, measured, and named, instead of looking
like intermittent feed unavailability. Any future one is caught in CI.

Worse: 17 games a season keep pricing on the rating-only vector until the
measured fix lands. That is a real ongoing cost, accepted deliberately and for
a stated reason rather than by oversight.

Also newly visible: the published NGS improvement figure was measured on a
sample that excluded one team's games entirely. That does not invalidate it,
but it means the figure describes 31 teams, and nothing said so.

## Measured, 2026-09-21 — and the fix is NOT an improvement

The port landed (`src/coverline/leagues/nfl/ngs.py`, which normalises) and the
fix was graded on the 26 Rams games in 2022-2023, seasons neither coefficient
vector was fit on. Paired per game: same game, same ratings, two vectors, one
actual margin.

| | MAE |
|---|---|
| rating-only (shipped, the bug) | 10.8295 |
| full ensemble (fixed) | 10.7384 |

Paired gain **+0.0911, SE 0.7570, t = +0.12**. Indistinguishable from nothing.

A first version of this measurement returned **t = −1.68** and appeared to show
the fix making predictions materially worse. That version held `elo_diff` at
0.0 on both sides, which compares the rating-only vector against a full
ensemble stripped of one of its own features. With the real walk-forward Elo
difference passed to both — the rating-only vector simply has no term for it,
which is the shipped behaviour — the sign flips and the effect vanishes. The
unfair version was the more dramatic result and would have been the easier one
to believe.

**So this is a correctness question, not an edge question, and the measurement
says so.** Fixing it buys consistency across 32 teams, not accuracy. Anyone
expecting better predictions from it will be disappointed, and that is now
written down rather than discovered later.

The new core normalises. The legacy path is deliberately untouched: changing a
live board on a t = +0.12 measurement needs a reason, and "no detectable
difference" is not one. A test fails if someone patches the legacy without
updating this record.

## Recovery

Nothing was removed. `model/ngs_team_code_fix.py` reproduces the measurement,
and its result artifact records every graded game.

## Revisit Triggers

- ~~**Immediately, when the NGS port lands.**~~ RESOLVED 2026-09-21: the port
  normalises, and the fix measured at t = +0.12.
- **If the new core takes over the live board**, the normalisation comes with
  it and Rams games change coefficient vector. That is a publishing change
  worth announcing even though it is not an accuracy change.
- **If the sample grows** -- more seasons, or the same question asked across
  all teams rather than one -- 26 games is thin, and a real effect of the size
  seen here would need roughly 700 games to distinguish from zero.
- **If nflverse switches NGS to `LA`**, the alias becomes a no-op and should
  be retired deliberately — a test already fails if `LAR` starts appearing in
  the canonical vocabulary.
- **If a second code mismatch appears**, one-off handling stops being
  adequate and a shared normalisation layer is warranted.
