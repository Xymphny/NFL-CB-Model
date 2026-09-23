---
status: accepted
date: 2026-09-23
supersedes: null
superseded_by: null
ledger_id: league-cfb-live
---

# 0027. Paper-trade CFB from the weekly ratings, on the vector that was graded

## Context and Problem Statement

Every league paper-trades until its ledger reaches the 150 settled games that
replace its historical market grade (ADR 0024). CFB could not start. Its only
feature source, `CachedWalkForwardSource`, reads the 2021-2023 walk-forward
cache, so the operator command could not price a current game, the capture
job skipped CFB, and settle.py had no CFB finals. CFB would never have
reached the floor.

The inputs already existed. The legacy CFB board prices every week from
`data/cfb_ratings/{season}-week-NN.json`, written by the cfb-weekly-job.
ESPN's scoreboard, already used to repair the CFB scores (ADR 0019/0020),
lists the current season with event ids, kickoffs, neutral-site flags and a
completion flag, and it spells schools exactly as the ratings do.

## Considered Options

- Leave CFB unpriced until a full live pipeline exists.
- Reuse the legacy board's numbers: the ensemble vector with the snapshot's
  Elo, then its slate de-bias and carryover alignment.
- Add a live source (`leagues/cfb/live.py`) that reads the weekly ratings
  against ESPN's schedule and feeds the core `CFBModel` unchanged, on the
  DVOA-only vector. Chosen.

## Decision Outcome

Chosen: the live source on the DVOA-only vector. There are two reasons.

1. **Elo.** The snapshots' `elo_rating` comes from the 2021-2025 schedule
   cache and does not move with this season's games. The ensemble was fit on
   in-season Elo, so feeding it a preseason number would run a model that was
   never fit.
2. **What gets graded.** The market grade (CFB w_hat +0.072, SE 0.064) and
   the CFB tier bands were both measured on the DVOA-only vector, because the
   cache has no Elo. Paper-trading the same vector means the ledger grade
   continues the historical one. Paper-trading a different vector would start
   a new grade.

The legacy de-bias and alignment are not carried over. Both change the
model's output, which ADR 0001 routes through the gate. The DVOA-only
path's +2.05-point mean residual (`cfb/model.py`) stays recorded and
uncorrected. The ledger grade will measure it along with everything else.

How it runs:

- CFB becomes a **date** slate in the operator command, like MLB, NHL and
  NBA. A CFB week spans Thursday to Saturday, and a paper run prices the
  games that start near its capture. The board still shows the whole week.
- Games are keyed by ESPN event id. Odds API names map to ESPN names through
  `leagues/cfb/teams.py`. That table is exact and built from names seen in
  live payloads; an unknown name is refused and named.
- Refused, with the reason on the board:
  - a neutral-site game (the vector has no home term to remove)
  - a team the snapshot does not rate
  - ratings more than one week behind the game's week
  - any snapshot computed after kickoff
- Whole-number spreads stay withheld, because CFB has no measured key-number
  table.
- settle.py reads CFB finals from `data/raw/cfb/espn_{season}.parquet`
  (completed games only). live-inputs-job refreshes that file daily.

## Consequences

- CFB paper-trades automatically from the first capture on, and settles and
  grades like every other league.
- A stalled cfb-weekly-job now **stops** CFB prices within a week instead of
  aging them silently. On the day this was built, the newest committed
  snapshot was week 2 (computed 2026-09-09). Week 4 is therefore refused
  until that job produces week 3 or 4. The job runs locally in 6 seconds
  with a peak of about 970 MB, which suggests a memory limit on its Render
  instance. That is unconfirmed until its logs are checked.
- Games against FCS opponents are never priced; the ratings cover FBS
  play-by-play only.
- The legacy CFB board (`deploy/cfb_odds_watch.py`) keeps running on its own
  numbers until ADR 0026's retirement reaches it. The two will disagree:
  that board applies the de-bias, and this source does not.

## Recovery

- **Git commit**: the parent of this ADR's commit has CFB as a week league
  priced from `CachedWalkForwardSource`.
- **Paths**: `scripts/recommend_slate.py` (`load_cfb`, the `LEAGUES` row),
  `scripts/paper_trade.py` (the CFB skip), `scripts/export_board.py` (the CFB
  refusal), `src/coverline/execution/settle.py` (no CFB finals).
- **Data artifacts**: `data/raw/cfb/espn_2026.parquet` (raw; regenerable with
  `model/ingest/cfb_espn.py --seasons 2026-2026 --force`). `data/cfb_ratings/*`
  is unchanged and still owned by the cfb-weekly-job.
- **To walk it back**: restore those four hunks. About an hour. CFB rows
  already in the ledger stay valid; they record what was priced.

## Revisit Triggers

- The cfb-weekly-job starts publishing in-season Elo. The ensemble becomes
  runnable as fit, and it needs its own market grade first.
- The CFB ledger reaches 150 settled games: its grade replaces the
  historical one, and this vector choice is judged on live evidence.
- More than about a fifth of a week's FBS-vs-FBS games are refused for an
  unknown team name. If so, the table needs another pass against a live
  payload.
