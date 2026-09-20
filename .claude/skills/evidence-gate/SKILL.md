---
name: evidence-gate
description: Use before shipping ANY model change to Coverline — a new signal, layer, market, or recalibration. Enforces the project's rule that nothing reaches the board without held-out evidence, and that whatever fails its test is withheld in code and said out loud.
disable-model-invocation: true
allowed-tools: Bash Read Write Edit Grep Glob
arguments: [change]
---

Run the evidence gate on: **$change** (if empty, ask what is being
gated before doing anything else).

This repo's entire claim is that its numbers were earned. A change
that skips this gate breaks that claim even if the change is good, so
run every step, in order, and report honestly.

## 1. Establish the baseline first

Before touching anything, record what the current code scores on the
same held-out data the change will be judged on. A change with no
baseline cannot be said to beat anything.

## 2. Split, and keep the split honest

- Fit and tune on TRAIN seasons only (currently `<= 2023`).
- Grade on HELD-OUT seasons (currently `2024-2025`), once.
- Every free parameter — blend weights, shrink constants, powers,
  thresholds — is chosen on train. If a constant was picked by
  looking at held-out scores, it is not held out any more; say so and
  refit.
- Exclude cold-start seasons from any frozen fit (see
  `BURN_IN_SEASONS`, `MLB_BURN_IN_GAMES`). The engine's first season
  has no prior states and is a distributional outlier — it
  contaminates shapes fit on it.

## 3. Compare on a common pool

Score old vs new on the SAME rows. A change that also changes which
rows qualify has moved the pool, and a pool change can manufacture an
improvement on its own. Report:

- log-loss and Brier (probabilities), or MAE vs the trailing-N
  baseline (point estimates)
- bucket calibration: claimed vs actual, with n per bucket
- which direction the misses run — conservative beats hot

## 4. Decide, and be willing to say no

- **Beats baseline and calibrates** → ship, in watch mode: the output
  is a labeled second opinion, never a verdict input, until live
  graded results exist.
- **Fails** → withhold it IN CODE, with a dated comment naming the
  numbers that failed and what would let it back in. Precedent:
  `CALIBRATED_MARKETS` in `model/player_projection.py` — pass yards
  is the biggest market and it sits out. That is the point, not an
  embarrassment.
- **Ambiguous** → it fails. Ties go to the market.

## 5. Guard the live path

Anything shipped must degrade gracefully:

- Refuse to answer outside the pool the gate was verified on (see
  `TD_MIN_LAMBDA`, `MIN_PROJ_OPP`) — silence beats extrapolation.
- Soft-fail when an upstream feed is missing, and say so in the log
  rather than crashing the job or silently changing behaviour.
- Add a regression test that pins the new guard, not just the happy
  path.

## 6. Write it into the ledger

Add a dated entry to the README's findings ledger with the verdict
and the numbers behind it — including what failed. The ledger is the
product; a change that is not in it did not happen.

## 7. Ship it the project's way

- `python3 tests/test_parsers.py` must pass in full.
- Commit with a message that states the held-out numbers.
- Tag the commit so the work survives a tree reset.
- Never `git add .` — stage named paths only; cron jobs commit data
  files into this repo and a blanket add erases them.
- Never force-push.
- Deliver changed files individually with a placement table (which
  folder, replace or new).
