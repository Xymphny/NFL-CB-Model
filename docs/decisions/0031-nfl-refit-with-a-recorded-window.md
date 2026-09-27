---
status: proposed
date: 2026-09-27
supersedes: null
superseded_by: null
ledger_id: league-nfl-pricing-leaf
---

# 0031. Refit both NFL vectors on 2016-2022, and grade them on 2023-2025

## Context and Problem Statement

The NFL market grade is labelled UPPER BOUND because no contiguous season
range reproduces the rating-only coefficients, so no graded season is provably
unseen. The full ensemble's docstring says it was fit on 2016-2021; neither
vector records its window in an artifact. The model file also flags its own
defect for "the next refit": `home_field` and `intercept` are collinear in the
full ensemble (equal-team home edge -1.13 against a market near +2.5).

Accuracy plan 2a asks for a clean refit on 2016-2022 with 2023-2025 held out,
the window recorded, a pre-registered test, and a ship rule: the refit
replaces the current vectors only if its held-out `w_hat` is not worse.

## What was done

`model/nfl_refit_2016_2022.py` -> `data/nfl_refit_2016_2022.json`.

- **Data.** Walk-forward `rating_diff` from `model/expanded_walk_forward_cache.csv`
  (2014-2023) plus `model/nfl_walk_forward_2024_2025_cache.csv`, built by the
  same `process_season` (checked: rebuilding 2023 reproduces the committed
  cache exactly, 208 of 208 games, max difference 0.0). Weeks 4-17. NGS through
  the core's alias table. Closing spreads and prices from nflverse.
- **Neutral sites kept.** `home_field` is 0 at a neutral site, so 23 neutral
  games in 2016-2022 (17 with NGS) separate it from the intercept. Weakly: the
  full ensemble's `home_field` SE is 3.2 points.
- **Elo as served.** The shipped ensemble was fit on the per-game Elo column,
  which includes Elo's +65 home advantage; the board differences final
  ratings without it. That is a constant 65 x 0.0348 = 2.26 points that
  training put in `elo_diff` and serving puts nowhere. The slate de-bias
  absorbs constants, so it never showed. The refit uses the served definition.

### Pre-registered test

Primary: held-out `w_hat` against the close (ADR 0024's estimator), the system
as the live board runs it: full ensemble where NGS and Elo exist, else
rating-only; that week's slate de-bias; `p_model` through the production
distribution and `cover_probability`; power devig; pushes dropped.

**Amended after the first run, stricter.** Weeks 4-17 always have NGS, so the
"system" grade priced all 609 graded games on the full ensemble and never
tested the rating-only vector. It is now graded separately with every held-out
game priced rating-only, and each vector must pass on its own.

## Result (2023-2025, 609 graded games)

| | Current | Refit |
|---|---|---|
| Full ensemble, with de-bias (primary) | w_hat **0.124** (SE 0.210) | w_hat **0.140** (SE 0.218) |
| Rating-only vector, with de-bias | -0.010 | -0.001 |
| Full ensemble, no de-bias | -0.054 (SE 0.158) | -0.012 (SE 0.208) |
| ATS, all games, with de-bias | 52.4% | 50.4% |
| Slope of model margin on market | 0.89 | 0.91 |
| Equal-team home edge, full ensemble | -1.13 | +1.14 |
| Equal-team home edge, rating-only | +1.65 | +1.63 |

ATS by gap band (with de-bias), current -> refit: 0-1 pts 55.1% -> 51.7%,
1-2 52.1% -> 42.0%, 2-3 51.5% -> 57.7%, 3-5 47.4% -> 44.3%, 5+ 54.7% -> 57.1%.
Bands hold 75-205 games each; none of these differences is distinguishable
from noise.

Both vectors **pass the ship rule** (not worse). Neither is better: every
difference is well inside one standard error.

## Considered Options

- **Ship the refit.** Satisfies the plan's rule, records the window, removes
  the collinearity defect and the Elo train/serve mismatch.
- **Keep the current vectors.** Nothing measured says the refit is better, and
  ATS moved the wrong way (a secondary, not gated).
- **Ship the refit alongside the legacy vectors, selected by label.** Old
  boards stay re-derivable with the vectors that priced them.

## Decision Outcome

**Proposed: ship the refit, alongside the legacy vectors and selected by
label.** It passes the pre-registered rule. Following that rule is the point:
overruling it on ATS, a secondary measure, would be choosing the metric after
seeing the results. It also records a fit window, which is the reason 2a
exists.

**Not applied.** Swapping the live coefficients was blocked by the session's
permission policy (2026-09-27), so this record stays proposed until someone
authorises that change. Nothing live has changed. The implementation, prepared
and not committed:

- `nfl/model.py` gains `MARGIN_COEFFICIENTS_2016_2022` and
  `MARGIN_COEFFICIENTS_RATING_ONLY_2016_2022` (4 dp, from the artifact), a
  `VECTORS` map keyed by the label a board records in `coefficient_set`, and
  `LIVE_VECTORS` / `LEGACY_VECTORS`.
- `predict_margin(f, vectors=LEGACY_VECTORS)` keeps its legacy default, so
  the board-provenance and parity harnesses keep re-deriving legacy numbers.
  `NFLModel(..., vectors=LIVE_VECTORS)` prices with the refit, and the board
  records the refit's label per game.

## Consequences

If applied: the fit window is recorded, and 2023 onward becomes genuinely
out-of-sample for the NFL market grade. The full ensemble stops carrying a
-1.13 home edge and a 2.26-point Elo constant, both of which the de-bias was
hiding. The board no longer equals the legacy odds-watch number for any game.
That is intended, but `data/nfl_ensemble_parity.json` then describes the port,
not the published number. The paper ledger (NFLModel through
`recommend_slate`) moves to the refit rating-only vector too.

Not better, and nobody should expect it to be: held-out `w_hat` moved +0.015
with SE 0.21.

## Recovery

Nothing is removed. The legacy vectors stay in `nfl/model.py` under their
legacy labels. Reverting means setting `LIVE_VECTORS` back to
`LEGACY_VECTORS`: one line. The refit is reproduced by
`python3 model/nfl_refit_2016_2022.py`, and its 2024-2025 rating cache is
committed training data.

## Revisit Triggers

- **2026 finishes.** Add it to the held-out window. It is the first season
  neither vector could have seen.
- **Weeks 1-3.** Neither grade covers them: the cache starts at week 4, and
  weeks 1-3 are when the rating-only vector and the preseason prior do their
  work (plan phase 3, candidate 1).
- **Neutral sites.** If a held-out season adds enough neutral games to tighten
  `home_field`'s SE of about 3 points, re-check the split.
