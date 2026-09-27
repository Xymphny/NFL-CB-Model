# Model accuracy plan

**Date:** 2026-09-27  
**Owner:** Pedro

"Accuracy" here means **beating the closing line**, graded as in ADR 0024 (`w_hat` vs close, CLV on paper
trades). Win rate on a few games is not the measure. Every league currently has market weight 0.

The plan has three phases:
1. Fix what's broken. Parity-gated, no new inputs.
2. Make the grades honest.
3. Test new inputs through the evidence gate, a few at a time.

The attempt log shrinks every shipped gain (b ≈ 0.6, a ceiling), so each attempt in phase 3 is chosen
deliberately.

---

## Phase 1: fix what's broken (build now)

### 1a. NFL: restore the full ensemble on the live board

**Problem.**
- `NFLLiveSource` reports `ngs_present=False` for every game, so the live board prices all of them with
  `MARGIN_COEFFICIENTS_V1_RATING_ONLY`. The docstring says so: "honest about covering the rating-only path
  and nothing else".
- The legacy odds-watch prices the same games with the full ensemble (NGS + Elo, 15 of 16 games in week 3),
  plus an in-season slate de-bias.

**Evidence:** walk-forward 2016–2025, closing `spread_line`. These are upper bounds; see 2a.

| | Rating-only (live now) | Full ensemble (legacy) |
|---|---|---|
| Slope of model margin on market spread | 0.43–0.50 | 0.81–0.83 |
| Takes the underdog | 80–86% | 60% |
| ATS | 49.0% (n=471) | 52.9% (n=1,492) |
| ATS on underdog picks | 47.0% (n=387) | 53.6% (n=921) |

**Live, week 3 board (2026-09-27):**
- 13 of 15 preferred sides are underdogs.
- Model spreads average 1.9 points against the market's 4.3 (slope 0.27).
- The model leans home against the market by +1.8.

**Build:**
- Port the NGS feature fetch (cpoe, separation, yac_oe, ryoe) and Elo into a live source, the ledger row
  named in `leagues/nfl/live.py`. Set `ngs_present` per game from what was actually fetched.
- Port the in-season intercept-only slate de-bias (`inseason_offsets`, `load_prior_debias` in
  `deploy/odds_watch_job.py`) into the board assembler, not the feature source; `live.py` explains why.
  Record the offset on the board.
- **Parity gate.** For every week-3 game in `data/divergence/2026-week-03-*.json`, the new core's margin
  matches the legacy margin to within 0.05 points: same `feature_values`, same coefficient set, same
  de-bias. Also run it on the two newest snapshots. If parity fails, say which term differs and stop.
- **No new inputs and no coefficient changes.** This restores what was already live.

**Until the parity gate passes,** add a board banner on NFL:

> "The live NFL board is running the ratings-only model: it takes the underdog about 80% of the time, and
> historically went 47% on those picks. The full model returns when its data feed is restored."

Holding NFL tiers until then (as NHL does early in the season) is **Pedro's call**. Build it behind a flag
and leave the flag off.

### 1b. NFL: catch feed disagreements about who started

- nflverse `games.csv` listed Tua Tagovailoa as ATL's week-2 starter. It was Cooper Rush.
- In `deploy/qb_status.get_qb_alerts`, when the last-start name and the depth chart disagree with a second
  source, mark the alert `source_conflict: true` and show "starter unconfirmed" rather than a name.
- Test it with the ATL week-2 case.

---

## Phase 2: make the grades honest (measure before improving)

### 2a. NFL: clean refit with a recorded fit window

- Today the grade is labelled "UPPER BOUND": no contiguous season range reproduces the rating-only
  coefficients.
- Refit both vectors on 2016–2022. Hold out 2023–2025. Record the window in the artifact.
- **Pre-registered test:** held-out `w_hat` vs close, and ATS by gap band.
- Also fix what the model file itself flags for its next refit: `home_field` and `intercept` are collinear.
  Include neutral-site games so the two can be separated.
- **Ship rule:** the refit replaces the current vectors only if held-out `w_hat` ≥ the current one's, since a
  refit is not allowed to look worse. Write the result as an ADR either way.

### 2b. Grades by point in the season, every league

The same model is not equally good all season. In the full-ensemble walk-forward (upper bound), ATS by
week was:

| Weeks | ATS |
|---|---|
| 1–4 | 52.3% |
| 5–8 | 54.3% |
| 9–12 | 55.3% |
| 13+ | 50.4% |

For CFB, the README ledger records the rating-divergence signal as validated **for weeks 5+ only** (54.6% at
5+ points, 574 games).

**Build:**
- `model/grade_by_week.py`: per league, `w_hat` and ATS by season segment. Write it to
  `data/grade_by_week.json`, shown on Record.
- **Proposal for Pedro:** hold CFB tiers in weeks 1–4, as NHL does in its first 10 games. The only
  validated CFB signal starts in week 5. This changes the display only, not prices. Build it behind a flag,
  off until he says so.

---

## Phase 3: candidate inputs (evidence gate: pre-registered, held out, graded vs close, ADR)

These are ranked by expected value. **Pedro picks which to register.** Don't start any without his go.

| # | League | Candidate | Why it's worth a test | Data |
|---|---|---|---|---|
| 1 | NFL | Backtest the preseason prior blend (k=2.0, win-total weight) | It drives weeks 1–4 and is labelled "weights not backtested" in `weekly_job.py`. It's the likely source of the early-season squeeze. | ratings history, win totals |
| 2 | NHL | Starting-goalie adjustment | It's the largest known factor the model misses, and the market prices it. | NHL API boxscores (goalie of record per game) |
| 3 | NBA | Availability adjustment: minutes-weighted absence of rotation players | The NBA model file names it as "the real model". Players data is now ingested. | `nba_players` ingest, 2024–26 |
| 4 | NHL | Partial carryover of last season's rates for the first ~10 games | Ratings reset to average, so opening weeks are blind. The static cross-season fit failed (t = −1.33), but a carryover like the NBA's 0.5 is untested. | nhl 2016–2026 |
| 5 | CFB | Extra shrinkage for teams with thin FBS history (new FBS members) | The week-4 Missouri State +34.5 at SMU gap was 33 points. | CFBD membership, ratings |
| 6 | MLB | Bullpen workload (relief outs, last 3 days) | Shown now, not priced. Offseason, so it's the lowest priority. | pitching cache |

**Already tested and declined; don't redo:**
- the quantified QB adjustment (NFL)
- the pbp pressure proxy
- the QB-continuity prior
- CFB roster priors
- schedule-spot residuals

---

## Order

1. Phase 1a and 1b, now.
2. Phase 2a and 2b next. They change no prices without an ADR, and they make every later grade trustworthy.
3. Phase 3 only for the candidates Pedro registers.
