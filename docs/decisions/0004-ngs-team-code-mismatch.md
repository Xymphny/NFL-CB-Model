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

## Recovery

Nothing was removed. The fix is a team-code normalisation applied wherever the
NGS frame is joined, plus a held-out measurement of its effect logged as an
attempt in `evidence/attempts.yaml`. The scope to re-measure is the 85 affected
games.

## Revisit Triggers

- **Immediately, when the NGS port lands.** The port must normalise codes or
  reproduce the bug; either way it needs this decision resolved first.
- **If nflverse switches NGS to `LA`**, the alias becomes a no-op and should
  be retired deliberately — a test already fails if `LAR` starts appearing in
  the canonical vocabulary.
- **If a second code mismatch appears**, one-off handling stops being
  adequate and a shared normalisation layer is warranted.
