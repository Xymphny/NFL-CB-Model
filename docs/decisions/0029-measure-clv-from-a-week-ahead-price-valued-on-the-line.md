---
status: accepted
date: 2026-09-25
supersedes: null
superseded_by: null
ledger_id: dashboard-board-export
---

# 0029. Measure CLV from a week-ahead price, valued on the line

## Context and Problem Statement

Every league paper-trades until its ledger has 150 settled games (ADR
0024). Closing-line value (CLV) is the paper trades' earliest verdict: it
is available the moment a game closes, long before 150 results exist. A
rehearsal of the first real capture, run end to end on 2026-09-25, found
the measurement broken in five places, all before any real trade had been
written:

1. **Early prices came too late.** The daily early poll fired only when a
   game was within 24 hours. So the NFL was never polled on Tuesday or
   Wednesday, and its "early" price was taken on game day.
2. **CLV ignored line moves.** It compared the close's fair probability
   at the close's line with the bet's price at its own line. Those are
   different bets whenever the line moved. A trade that beat the close by
   a point read −2.4 probability points, which is exactly the vig.
3. **Closes were attached before kickoff.** Every snapshot holds every
   upcoming game. A settle run before kickoff therefore took the latest
   snapshot so far as the close, permanently, because closes are
   append-only.
4. **Games already in play were paper-traded.** The odds feed lists
   games in progress, and the runner priced them.
5. **An out-of-memory kill could lose the snapshot.** Paper trading and
   the export ran inside the capture process, with a combined peak close
   to the 512 MB limit. A kill arrived before the commit, and Render's
   ephemeral disk lost the snapshot with it.

## Considered Options

- Keep game-day early prices and a price-only CLV. That measures little
  on spreads, where lines move and prices do not.
- A second paid poll midweek. It isn't needed: the daily poll already
  returns the whole week.
- Price the whole football week from each daily poll, once per game, and
  value moved lines on the league's margin distribution. **Chosen.**

## Decision Outcome

**Timing (NFL, CFB).** An early poll runs on every in-season day with a
game inside the next 7 days. It prices through the end of the week its
inputs are good for:

- NFL: the NFL week of the next game. Ratings update on Tuesday.
- CFB: until the next ratings run, Sunday 10:00 UTC.

A game already in the ledger is skipped. So each game gets its first
early trade plus its close. Daily leagues keep a 24-hour window. The cost
is about 100 credits a month.

**Valuation.** When the line moved, the close's devigged probability is
moved to the bet's line (`execution/line_clv.py`). The margin mean is
solved from the close, and only the distribution's shape comes from the
model: NFL and CFB use their measured key numbers, NBA its fitted sigma.
The NFL half point off 3 is worth +4.4 points and off 5 is worth +1.1. A
CFB half point across 0 is worth nothing, because a tie cannot happen.
Moneylines and unmoved lines keep the price-only figure.

**Integrity.**

- A close attaches only after kickoff.
- A quote for a game that had started by the snapshot's time is dropped.
- Paper trading and the export run in child processes. The snapshot is
  always committed. The ledger is committed unless a paper child was
  killed. `data/site` is committed only after a clean export.

## Consequences

- CLV now measures the week's movement, which is where football lines
  move, and reports it in the unit that is aggregated.
- The ledger holds two trades per game. Grades take one row per game:
  the latest for results (ADR 0024), the earliest for CLV.
- The CLV unit changed before any real CLV existed, so there is nothing
  to restate.

## Recovery

Each piece reverts on its own:

| Piece | How to revert |
|---|---|
| Lookahead | `EARLY_LOOKAHEAD_HOURS` in `scripts/capture.py` |
| Week window | `WEEKLY_EARLY` in `scripts/paper_trade.py` |
| Valuation | Drop `line_aware_clv` in `grade.clv_summary` and `export_record` |

The integrity fixes should not be reverted: each one corrects data.

## Revisit Triggers

- Paper CLV on moved lines disagrees in sign with the price-only CLV on
  unmoved lines across more than 100 games. That would mean the shape is
  wrong.
- Credits: the site shows the remaining quota. If month-end headroom
  falls under 10%, reduce the lookahead first.
- A league gains a mid-week input, for example NFL injury reports as a
  model input. The week window then stops meaning "the inputs are fixed".
