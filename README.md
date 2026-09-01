# NFL/CFB Efficiency Model — Build Status

Implements the full spec (`football-efficiency-model-spec-v0.1.md`) as far as it can go without live API keys, a real GitHub remote, or network access this sandbox doesn't have. This README is the ground truth on what's actually been run vs. what's structurally written but unverified — read it before bug-fixing anything.

## Tested and working (real data, real output, checked by hand)

- **`ingest/nfl_pbp.py`** — real NFL play-by-play from nflverse (GitHub-hosted, no key needed)
- **`ingest/nfl_schedules.py`** — real schedule data: home/away, rest days, closing lines, weather, neutral-site flags (272 games/season, 5 correctly flagged neutral-site in 2023)
- **`model/play_value.py`** — play scoring, now league-aware (NFL/CFB threshold tables both implemented, though CFB path is only exercised once `ingest/cfb_pbp.py` is verified — see below)
- **`model/ratings.py`** — bucketing, baselines, opponent adjustment, garbage-time filtering (uses nflverse's own precomputed `wp` field — no win-probability model had to be built from scratch), home-field/rest joins, recency weighting, aggregation
- **`model/calibrate_points_model.py`** — points-prediction layer calibrated against 5 real seasons. Home-field coefficient (+2.83 pts) independently matches the NFL's real ~2-3 point home-field advantage — a genuine sanity check, not just "it ran." R² is low (0.03-0.06) and *should be*, since this version has no in-season updating yet (see "Known integration gaps" below)
- **`model/injuries_and_var.py`** — real 2023 injury reports and snap-count-based QB start tracking. Correctly identified the actual 2023 Cardinals QB carousel (Dobbs → Tune → Murray) from snap-share data alone
- **`model/market_comparison.py`** — de-vig math (verified to sum to exactly 1.0) and divergence-flagging logic
- **`model/season_simulation.py`** + **`demo/run_season_simulation.py`** — Monte Carlo season simulation, validated against what actually happened in the second half of 2023. Several teams landed within half a win of their real final record. One instructive miss: Minnesota was projected at 10.0 wins, actually finished at 7 — Kirk Cousins tore his Achilles right around the simulation's cutoff week, and the persistent-QB-adjustment module isn't wired into the simulation yet. Not a bug — a clear illustration of exactly why that integration (see below) matters
- **`deploy/validate.py`** — data validation checks, tested against synthetic bad data (wrong team count, implausible rating value) — both correctly caught. One real bug found and fixed during testing: the original play-count bounds assumed single-week data but the job passes season-to-date data
- **`deploy/notify.py`** — layered alerting logic, confirmed to fail gracefully (not crash) when no webhook/heartbeat URL is configured
- **`deploy/weekly_job.py`** — full orchestration tested end-to-end (pipeline → validate → write JSON) with git push correctly skipped when no remote is configured, rather than failing
- **Memory usage** — fixed a real production bug: the initial version peaked at ~493.5MB RSS for the whole pipeline, dangerously close to Render's 512Mi cron job limit (and this is exactly what caused the "Out of memory (used over 512Mi)" failure). Root cause, found by measuring rather than guessing: decompressing this file's ~100MB of raw CSV text happens *before* pandas can apply any column or row filtering — a single `read_csv` call has to decompress the whole gzip stream regardless of `usecols`. Fixed by reading in chunks and filtering each chunk before accumulating, so the full decompressed text is never held in memory at once. Measured result: **155.6MB peak**, ~381MB of margin under the limit instead of ~18MB.
- **`deploy/git_utils.py`** — shared git commit/push logic, extracted from `weekly_job.py` so `odds_watch_job.py` doesn't duplicate it. Fixes two real production failures found by reproducing them locally (not guessed): Render's checkout doesn't leave a named `origin` remote configured (fixed with `git remote add` falling back to `set-url`), and Render's checkout leaves the repo in detached HEAD state, which breaks `-u origin HEAD` (fixed by pushing to an explicit branch, `GIT_BRANCH` env var, default `main`). Both reproduced against a real local bare repo in detached-HEAD state before being marked fixed.
- **`odds_watch_job.py`'s git push** — this was a real gap: the original version wrote `divergence.json` locally but never committed it, so nothing would have reached the repo or triggered the static site's auto-deploy. Now wired to the same shared, hardened `git_utils.git_commit_and_push`. Verified with a mock that the control flow calls it with the correct file path and commit message.
- **Game-day gating for `odds_watch_job.py`** — a real production measurement confirmed 6 credits per API call, revealing the fixed every-4-hours/every-day schedule would burn through The Odds API's 500-credit free tier in ~2 weeks, not a month. `ingest.nfl_schedules.is_game_day()` checks real schedule data rather than guessing a day-of-week pattern (verified: NFL game days are mostly Sunday but meaningfully include Monday/Thursday and occasionally Friday/Saturday). Tested against four known real 2023 dates, all four correct. Wired into `odds_watch_job.py` to skip the API call entirely on non-game days — tested both branches directly.
- **`model/prediction.py`** — the points-prediction layer (Section 11.4), finally wired into `odds_watch_job.py`. Loads the ratings `weekly_job.py` already committed, matches them to the current week's real schedule, and produces spread/total/win-probability per game using the actual calibrated coefficients from `calibrate_points_model.py`. Verified: neutral-site handling isolates and removes exactly the 2.83-point home-field coefficient (tested with identical-rated teams), and predictions were generated for all 16 real Week 1 2026 games and all 16 real Week 18 2023 games.
- **Two real bugs found and fixed while wiring this in:**
  - `compute_divergences()` was pulling `market_spread`/`market_total` from the model's own prediction dict (a placeholder oversight) instead of parsing the actual spreads/totals markets from the odds API response. Fixed to parse both markets properly — tested against a realistic synthetic API response matching The Odds API's documented format.
  - `flag_divergence()`'s boolean flags broke JSON serialization (`Object of type bool is not JSON serializable`) because comparisons on numpy floats (which flow through from pandas upstream) produce `numpy.bool_`, not a native Python `bool`. Fixed with explicit `bool(...)` casts, caught by an actual end-to-end test run, not by inspection.
- **Full end-to-end verification**: ran `odds_watch_job.py`'s real `main()` control flow (API call mocked, everything else real) against real committed ratings and a realistic odds payload — produced a correct, complete `divergence.json` with sensible output (Detroit favored by both model and market, small gap, correctly not flagged as divergent).
- **`SeasonNotStartedError` handling in `weekly_job.py`** — a real production failure: with `SEASON=2026` (the actual current season), nflverse hasn't published any play-by-play data yet since no 2026 games have been played (confirmed directly — the URL returns a real 404). This isn't a bug, but it was surfacing as a generic, alarming failure. Now caught specifically and treated as an expected, temporary soft-skip (matching how `odds_watch_job.py` treats "not a game day"), rather than firing a failure alert for something that isn't actually wrong. Will resolve on its own once Week 1 happens and nflverse publishes real data.
- **Webhook retry-on-429**: the Discord alert webhook hit its own rate limit twice during heavy manual testing today. Added a single retry with the `Retry-After` backoff Discord returns — tested that the graceful no-webhook-configured path still works correctly after this change.
- **Extensive git push diagnostics added to `git_utils.py`** — when the missing-`ratings.json` mystery first appeared, added verbose logging (computed repo path vs. git's own reported root, configured remotes, commit/push exit codes, and the resulting remote ref) to `git_commit_and_push()` to make the *next* failure immediately diagnosable rather than requiring another round of guessing. Verified against a fresh detached-HEAD reproduction that the diagnostics themselves are accurate before relying on them. As it turned out, the actual root cause was upstream of git entirely (see `SeasonNotStartedError` above) — but the diagnostics remain in place for the next time something git-related actually does go wrong.
- **`get_next_upcoming_week()` — a real bug caught before it could manifest**: `odds_watch_job.py` was using `get_current_week()` (the *last completed* week — correct for `weekly_job.py`'s own rating computation) to decide which week's games to predict, instead of the *upcoming* week. Pre-season, both happen to return week 1 by coincidence, which would have hidden this bug until Week 2 actually arrived — at which point it would have silently kept comparing against already-finished Week 1 games. Fixed with a dedicated function (earliest week with any unplayed game, not a "+1" offset, since a fixed offset breaks around bye weeks and the season boundary), and verified against a simulated real mid-season point (2023 data with weeks 2+ artificially blanked out) to directly confirm the two functions diverge exactly where they need to.

## Written but NOT testable in this sandbox (need real credentials/network outside it)

- **`ingest/cfb_pbp.py`** — CFBD integration; `api.collegefootballdata.com` isn't reachable here. Verify schema mapping against a live response before trusting it (see prior detailed notes in this file's git history / the spec's Section on CFB)
- **`model/external_tracking.py`** — ESPN FPI scraping; `espn.com` isn't reachable here, and the endpoint/response shape is a best guess, not verified. Also carries the ToS consideration flagged in the spec
- **`model/player_props.py`** — needs a live Odds API key. Two things need verification before building further: whether player props require a plan tier above the $30/mo 20K tier, and actual CFB coverage depth
- **`deploy/odds_watch_job.py`** — needs a live Odds API key; confirmed to fail gracefully without one
- **`deploy/weekly_job.py`**'s git commit/push path specifically — the pipeline portion is tested, the actual push to a remote is not, since there's no real repo to push to here

## Known integration gaps (built separately, not yet wired together)

- **QB persistence isn't fed into the season simulation** — the Minnesota/Cousins case above is the concrete example. `model/injuries_and_var.py`'s `apply_persistent_qb_adjustment()` exists but `demo/run_season_simulation.py` doesn't call it yet
- ~~The points-prediction layer isn't fed into `deploy/odds_watch_job.py`~~ — **fixed**, see `model/prediction.py` above
- ~~Preseason prior / credibility weighting (Section 11.1)~~ — **fixed**, see new section below
- **Bootstrap uncertainty isn't fed into `deploy/weekly_job.py`'s output** — it's computed and tested in `model/market_comparison.py` but the weekly job doesn't call it or include it in `ratings.json`
- **Walk-forward backtesting harness (Section 11's protocol)** — not built. `calibrate_points_model.py`'s prior-season-only approach is a simpler stand-in, explicitly caveated in its own output

## Preseason prior / credibility weighting (Section 11.1) — built, calibrated, live

`k=2` (how many games the prior is "worth") found via a real backtest against 2021-2023 data (`model/calibrate_credibility_k.py`), not guessed — checked how close early-season blended ratings got to each team's true final-season rating vs. raw in-season-only ratings. Result: blending reduced error by 11.1% overall, and — confirming the theory, not just the number — helped most early (+16.2% at Week 2) and faded to near-nothing by Week 6 (+2.8%), with larger k values actively hurting once real data existed to trust instead. Wired into `weekly_job.py`: the prior (last season's final rating) is computed once and cached (`data/priors/{season}.json`, committed like the ratings snapshots), then reused on every subsequent week rather than recomputed — verified directly (first run computed and cached a fresh prior, second run correctly skipped recomputation).

**Honest scope limit**: the full Section 11.1 design also calls for blending in Vegas win-total-implied strength, especially for CFB. That needs historical preseason betting lines, which require a paid Odds API tier not available here. `vegas_win_total` is a supported optional input in `model/preseason_prior.py`, but nothing fabricates that data — what's live in production is the prior-season-rating blend alone, which is what's actually been calibrated against real results.

Offense/defense components are blended separately (same `k`, extended by reasonable assumption) and `total_rating` is derived from them, keeping `offense - defense = total` internally consistent — worth noting that `k=2` was calibrated specifically against `total_rating` error, so applying it to the components individually is an extension, not independently validated on its own.

## Known gaps vs. the full spec (not started)

- Special teams sub-model (spec 3.7) — special-teams plays are filtered out entirely, not scored
- CFB-specific deltas beyond the threshold table (conference-strength substitution, FBS-only baseline pool)
- Player-level target/carry data ingestion (needed for a real player-props projection, not the placeholder currently in `player_props.py`)

## The dashboard (`frontend/`)

A Vite/React static site — scoreboard/tote-board visual theme (deep board-green and near-black, amber LED-style rating numbers, "Big Shoulders Display" for headlines), matching the subject matter rather than a generic dashboard look. Reads whichever ratings/divergence snapshot the build-time manifest says is latest (see below) — not a fixed `ratings.json` path anymore.

- **Built and verified**: compiles cleanly (`npm run build`), served locally and checked against real sample data (32 real teams from an actual pipeline run, 2 sample divergence entries including one flagged game) — both data endpoints returned correct content, the served index page returned HTTP 200 with the expected React root element.
- **NOT verified**: actual visual rendering in a real browser. This sandbox couldn't get a reliable headless-browser session running, so "compiles and serves the right data" is confirmed, but "looks right and the click-to-sort table interaction actually works" has only been confirmed by code review, not by seeing it rendered. Worth a visual check once deployed.
- `render.yaml`'s static site build command copies the top-level `data/*.json` files into `frontend/public/data/` before running `vite build` — this is how the committed ratings/divergence data actually reaches the deployed site. Without this copy step the site would build successfully but show empty states for everything.
- Empty states are intentional, not missing content: if `ratings.json`/`divergence.json` don't exist yet (before the cron jobs' first real run) or a game day has no posted lines yet, the site explains why rather than showing a blank page or crashing.
- **Per-team profile page** (added after reviewing a reference sports-betting dashboard): click any team in the ratings table to see a dedicated page with EPA/play, success rate, DVOA, red zone points-per-trip, and turnover margin — the five metrics requested — styled as stat tiles matching the existing scoreboard theme, plus a 0-100 composite "grade" badge (a display-only scaling of `total_rating`, not a new statistic). All five metrics are computed in `model/team_profile.py` and verified against real 2023 data (sanity-checked: SF/Miami's strong 2023 offenses show up correctly, Philadelphia's real -10 turnover margin matches their known late-season collapse, and every value falls in a realistic real-NFL range).
- **Stopped overwriting `ratings.json`/`divergence.json` — immutable snapshot files instead.** The repeated overwrites were causing real git friction (a bot repeatedly modifying the same lines is what produces divergent-branch pain when you also work on the repo locally). Fixed by writing `data/ratings/{season}-week-{week}.json` and `data/divergence/{season}-week-{week}-{timestamp}.json` — every write is now a pure git addition, never a modification, verified directly (`git show` on a test commit confirmed "1 file changed, 1 insertion(+)", not a diff on an existing file). Zero impact on the actual rating computation — this only changes where output is written, not how it's computed.
  - This also directly unlocks the time-window/trend feature flagged as missing earlier, and gives `odds_watch_job.py`'s repeated-checks-per-game-day a real history of line movement for free — exactly what Section 9.3's closing-line-value tracking needs (verified: two checks in the same test run produced two separate snapshots showing the line moving from -7.5 to -9.0, not one overwritten value).
  - **`deploy/generate_manifest.py`** — since a static site can't list a directory itself, this runs at *build time* (not commit time) to list whatever snapshots currently exist and copy them into the site's build. Critically, this script is never committed to git and produces no git changes of its own — which is exactly why this design doesn't reintroduce the overwrite problem it was built to solve.
  - Fixed a real bug this change surfaced: `git_utils.py`'s `repo_dir` computation assumed a fixed folder depth (`data/ratings.json`, 2 levels from root) and would have silently broken once snapshot files went a level deeper (`data/ratings/2023-week-18.json`, 3 levels). Fixed to use the process's actual working directory instead of counting path segments — tested against the exact deeper-nested path to confirm.
  - Fixed `odds_watch_job.py`'s ratings lookup, which was still checking for the old fixed `ratings.json` path — added `find_latest_ratings_snapshot()` and verified it correctly picks the highest week number when multiple snapshots exist (tested with week 17 and 18 present, correctly selected 18).

## Running it

```bash
pip install pandas requests numpy scipy
cd football_model

# Layer 1 pipeline on real 2023 data
python3 demo/run_nfl_2023.py

# Season simulation, validated against real second-half 2023 results
python3 demo/run_season_simulation.py

# Points-prediction calibration across 5 real seasons (takes a few minutes — pulls multiple full seasons)
python3 model/calibrate_points_model.py

# Injury/QB tracking on real data
python3 model/injuries_and_var.py

# De-vig and divergence math (self-contained, synthetic example)
python3 model/market_comparison.py

# Validation logic (self-contained, synthetic example)
python3 deploy/validate.py

# Weekly job dry run (no git remote = local-only)
SEASON=2023 CURRENT_WEEK=10 REPO_DATA_PATH=/tmp/test_output python3 deploy/weekly_job.py

# Frontend dashboard (from the frontend/ directory)
npm install
npm run build      # outputs to frontend/dist
npm run dev        # local dev server with hot reload
```
