---
status: accepted
date: 2026-09-23
supersedes: null
superseded_by: null
ledger_id: dashboard-board-export
---

# 0026. Retire the legacy board surfaces the revamp did not carry over

## Context and Problem Statement

The dashboard revamp (Coinflip) renders `data/site/board_*.json`,
`record.json` and `gates.json` from the core and computes nothing of its own.
The legacy site priced games in the browser from the legacy odds-watch files
and carried a dozen surfaces built on that path. Some had successors, some had
none, and the owner asked that anything not carried over be written down so
it can be revisited.

## Considered Options

- Port every legacy surface onto the new data.
- Keep the legacy surfaces alongside the new board.
- Carry over what the brief requires, record the rest here with a way back.
  Chosen.

## Decision Outcome

Carried over, rebuilt on core data: league switcher, board, reasoning panel,
per-sport context (NFL QB, venue, first-year staff; MLB probables and scratches;
NBA rest and back-to-backs; NHL early season), the regime cap (now applied in
the core export), line at open, the record (now paper trades), gates (now
generated), Teams, the NFL props-engine ledger, Discord login, bankroll, bet
log, exposure cap, and personal CLV.

Not carried over:

| Legacy surface | Why not | Data it read (retained) |
|---|---|---|
| Browser pricing: `coverProb`, `sizeStake`, `PLAY_GAP`/`LEAN_GAP`, `ATS_SIGMA` | The brief forbids site-side pricing; the core ships every number | `data/margin_dist.json`, `data/cfb_edge_calibration.json` |
| Confidence meter (five drivers) | Its drivers were computed in the browser and several were unmeasured | `data/divergence/*`, `data/performance.json` |
| Alt-line fair prices | Computed in the browser from a residual pmf | `data/margin_dist.json` |
| Track record and KPI strip (`TrackRecord`, `useClvReport`) | Graded the legacy board's picks; the record is now the paper ledger | `data/performance.json`, `data/cfb_performance.json`, `data/divergence/*` |
| Hand-written gates ledger (`GATES` in JSX) | Replaced by `gates.json`, generated from the artifacts | — |
| Division standings, schedule, team profile pages, rating trend chart | Not in the brief; the Teams view shows what each model rates | `data/site/teams.json`, `data/ratings/*`, `data/cfb_ratings/*` |
| NFL players tab (leaders, weekly props board, player grades) | The brief keeps only what the player model uses; the engine is in watch mode, so its graded ledger is shown | `data/site/player_leaders.json`, `data/props/*`, `data/player_grades/*` |
| ESPN live scores, team logos | External calls and branding outside the neutral five-sport identity | `data/site/cfb_logos.json` |

Every data file above is still produced by its job and still copied into the
build by `deploy/generate_manifest.py`; only the surfaces are gone.

## Consequences

- The site cannot disagree with the model: it has no model code.
- Legacy performance history is no longer visible on the site, though it stays
  in the repo and the jobs still write it.
- `tests/test_model_guards.py` guards that pinned legacy strings were rewritten
  to pin the same properties in the new architecture. Each docstring says what
  moved.

## Recovery

- **Git commit**: `8b619ce` (the last commit with the legacy site).
- **Paths at that commit**: `frontend/src/App.jsx`, `frontend/src/staking.js`,
  `frontend/src/MyBook.jsx`, `frontend/src/index.css`.
- **Data artifacts**: listed in the table; all retained and still refreshed.
- **To bring one back**: a surface that reads a retained file is about half a
  day to rebuild as a view on the new shell. Any figure it computed in the
  browser must first move into an export, which is the brief's one rule.
- **What has changed since**: tiers and stakes are now the core's, so no legacy
  surface's thresholds or Kelly sizing can return as they were.

## Revisit Triggers

- The owner asks for any row in the table.
- A league's market grade clears zero: the legacy track-record idea (P&L at
  the shown stakes) becomes meaningful again, built on the paper ledger.
