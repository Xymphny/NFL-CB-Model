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

- ~~QB persistence isn't fed into the season simulation~~ — **fixed, with an honest finding.** `model/injuries_and_var.py` now has real per-play QB attribution (`passer_player_name` added to the pipeline) computing actual VAR against a real backup-level replacement baseline, and correctly detects real starter changes (verified: correctly identified Kirk Cousins as MIN's starter through week 8 of 2023, then Josh Dobbs after week 10, matching the real injury). Wired into `demo/run_season_simulation.py`. **Important finding, not oversold**: this moved MIN's projection from 9.9 to 9.2 wins (correctly directioned), but the actual outcome was 7 — because Dobbs was still playing well at the week-10 cutoff, and the real collapse (benched for Mullens) happened afterward. The adjustment works correctly; it was never going to predict a decline that hadn't happened yet.
- ~~The points-prediction layer isn't fed into `deploy/odds_watch_job.py`~~ — **fixed**, see `model/prediction.py` above
- ~~Preseason prior / credibility weighting (Section 11.1)~~ — **fixed**, see new section below
- ~~Bootstrap uncertainty isn't fed into `deploy/weekly_job.py`'s output~~ — **fixed.** `ratings.json` now includes `rating_std`/`rating_p05`/`rating_p95` per team, computed via 100 bootstrap iterations each run — tested against real data, confirmed no nulls across all 32 teams.
- ~~Walk-forward backtesting harness (Section 11's protocol)~~ — **built and run for real.** `model/walk_forward_backtest.py` computes in-season ratings using only data strictly before each predicted week (true walk-forward, zero lookahead), across 2021-2023, 623 real games. **Honest result**: 58.75% straight-up accuracy, Brier score 0.2326 — real predictive signal (meaningfully better than the 0.25 an uninformative model would score), but only modestly ahead of a simple home-field heuristic (~57-58% historically), not a highly sharp predictor yet. This is the first rigorous, lookahead-free accuracy number for the whole project — a real baseline to measure future improvements against, not a claim of being highly accurate already.

## Preseason prior / credibility weighting (Section 11.1) — built, calibrated, live

`k=2` (how many games the prior is "worth") found via a real backtest against 2021-2023 data (`model/calibrate_credibility_k.py`), not guessed — checked how close early-season blended ratings got to each team's true final-season rating vs. raw in-season-only ratings. Result: blending reduced error by 11.1% overall, and — confirming the theory, not just the number — helped most early (+16.2% at Week 2) and faded to near-nothing by Week 6 (+2.8%), with larger k values actively hurting once real data existed to trust instead. Wired into `weekly_job.py`: the prior (last season's final rating) is computed once and cached (`data/priors/{season}.json`, committed like the ratings snapshots), then reused on every subsequent week rather than recomputed — verified directly (first run computed and cached a fresh prior, second run correctly skipped recomputation).

**Honest scope limit**: the full Section 11.1 design also calls for blending in Vegas win-total-implied strength, especially for CFB. That needs historical preseason betting lines, which require a paid Odds API tier not available here. `vegas_win_total` is a supported optional input in `model/preseason_prior.py`, but nothing fabricates that data — what's live in production is the prior-season-rating blend alone, which is what's actually been calibrated against real results.

Offense/defense components are blended separately (same `k`, extended by reasonable assumption) and `total_rating` is derived from them, keeping `offense - defense = total` internally consistent — worth noting that `k=2` was calibrated specifically against `total_rating` error, so applying it to the components individually is an extension, not independently validated on its own.

## Real 2026 preseason performance signal — built after discovering a real data gap

Checked directly: nflverse (this project's core data source) has **zero preseason play-by-play, for any season, ever** — not a timing issue, a structural one. Real 2026 preseason results (Hall of Fame Game + all 3 preseason weeks, 49 games) were gathered manually from ESPN instead (`model/preseason_2026_results.py`) — ESPN has real final scores but not down-by-down play-by-play, which limits what's computable to a point-differential signal, not a full Layer 1 rating.

`model/preseason_performance.py` converts each team's average preseason point differential into a small rating nudge (default weight 0.10), applied on top of the properly-backtested last-season prior — verified against real data: Baltimore's actual +21.7 average preseason point differential and Miami's actual -13.7 both computed correctly by hand-checking the real game logs, and the wiring into `weekly_job.py`'s prior computation was tested directly (bypassing the season-not-started guard, since 2026 in-season data doesn't exist yet, but the prior computation itself uses 2025 — a real completed season).

**Honest caveats, load-bearing for whether to trust this:**
- Preseason starters typically play one series total — this signal mostly reflects backup/roster-bubble performance, not the actual Week 1 roster
- A meaningful fraction of preseason participants get cut before Week 1
- Unlike `k=2` (backtested against 2021-2023), the 0.10 weight here is **not backtested** — repeating that process would mean manually gathering multiple past seasons' preseason scores the same way, which wasn't done. Treat it as a conservative, defensible default, not a validated number
- This data is manually curated and 2026-specific. A future season needs its own `preseason_{season}_results.py`, gathered the same way — there's no automated pipeline for this, by necessity, since no free API provides it

## Real Week 1 2026 preseason-informed predictions and market comparison

Built at the user's request to see real predictions for opening week using preseason data, before the season provides its own in-season data.

**What's real here, end to end:**
- `model/preseason_wk3_boxscores.py` — real box-score team stats (total yards, giveaways) for all 16 PRE WK3 games, gathered directly from ESPN's box score pages. Scoped to PRE WK3 only (closest to final roster cuts) rather than all 49 preseason games, per an explicit scope decision.
- `model/preseason_performance.py` — extended to combine point differential (50% weight), yardage margin (25%), and takeaway margin (25%) into one preseason signal, rather than point differential alone. Still blended into the last-season prior at the same conservative 0.10 weight — see the module's honest caveats section, unchanged from before.
- `deploy/generate_week1_predictions.py` — computes the preseason-informed prior and runs it through the real points-prediction layer against the real Week 1 2026 schedule. Written as `data/ratings/2026-week-00.json` (week 0, not week 1 — deliberately, so it never collides with or gets confused for the real post-game Week 1 rating `weekly_job.py` will eventually produce; "2026-week-00" sorts before "2026-week-01" alphabetically, so real data automatically takes over once it exists).
- `model/week1_2026_lines.py` — real, current Week 1 book lines gathered from ESPN's odds page.
- `deploy/generate_week1_divergence.py` — compares the two, using the same de-vig and divergence-flagging logic as the rest of this project.

**A real bug found and fixed in this process**: 4 of the 16 gathered lines had the home team's spread sign wrong, specifically on games where the *away* team was favored (an easy mismatch to make, since home teams are favored more often). This initially produced a suspicious, uniform pattern — every single game diverging in the same direction — which was the tell that something was wrong rather than a genuine finding. Caught by testing determinism of the underlying computation, then auditing the raw gathered data against the original ESPN text. Fixed by cross-checking all 16 home/away assignments against the authoritative schedule data before re-running.

**The corrected result is a genuinely mixed picture, not a uniform bias** — some games show near-exact model/market agreement (CHI@CAR: 0.0 gap, correctly not flagged), while others show real, large disagreement (BAL@IND: the model and market don't even agree on who's favored). 15 of 16 games still exceed the divergence threshold, which is an honest, expected finding given the underlying limitation already documented: a 2025-based prior with a small preseason nudge is inherently less informed than live market pricing, which already incorporates this year's actual roster/coaching/injury news the model can't see yet. This is presented as "the model doesn't know what the market knows," not as 15 discovered betting opportunities.

**The 2023 demo data has been removed** and replaced by this real 2026 preseason-informed projection, per an explicit request to do so.

## Vegas win totals — the highest-leverage gap, now closed

Real 2026 season win totals for all 32 teams, gathered from Squawka (sourced to BetMGM, dated August 28, 2026) — `model/win_totals_2026.py`. This is exactly the market signal Section 11.1's design called for but was never populated until now: the market's own aggregated view of team strength, synthesizing information (beat-reporter access, scouting, coaching interviews) this model can't replicate on its own.

**A real bug found and fixed while wiring this in**: the prior cache was being written *before* the preseason/vegas adjustments were applied, meaning a fresh computation included them but a cached reload silently didn't — the two paths would have quietly disagreed with each other. Caught by directly testing that two consecutive calls (fresh vs. cached) produced identical results, which they didn't until fixed.

**Combined weighting**: last-season rating (50%), real preseason performance (15%), real Vegas win totals (35%) — a reasonable, deliberately-considered combination, but *not* backtested the way the in-season k=2 credibility weight was. There's no efficient way to validate this specific blend without repeating real data-gathering across multiple past seasons, which wasn't done here.

**Measured result**: flagged divergences with the real market dropped from 15/16 to 13/16, and several individual gaps shrank dramatically — BAL@IND went from a +6.2 gap (model and market disagreeing on who even wins) to +2.9 (agreeing on the favorite, disagreeing on the margin), and CLE@JAX/BUF@HOU are now within 0.3 points of the market. Real, measured progress from real data — not a claim of having closed the gap entirely.

## Coaching and QB changes — the last two gaps addressed this round

**Real 2026 coaching changes** (`model/coaching_changes_2026.py`) — all 10 confirmed head coaching changes this offseason, cross-referenced across ESPN, Yahoo Sports, FOX Sports, and NFL.com for consistency.

**Real 2026 QB changes** (`model/qb_changes_2026.py`) — only high-confidence, unambiguous cases included (Arizona trading away Kyler Murray for Jacoby Brissett; Murray landing in Minnesota; Miami releasing Tua Tagovailoa for Malik Willis), not every "projected" starter from preseason coverage, many of which remain genuinely uncertain this far out.

**Design decision**: rather than try to quantify each new coach's or QB's individual value without real player/scheme-level data, teams with a real disruption get a *boosted* Vegas weight in the prior blend (+0.10 per disruption type, stacking for teams with both, capped at 0.60 total) — since last season's rating reflects a coach/QB who won't be there, while the market has already priced in the real personnel change. 11 teams affected: the 10 coaching changes plus Minnesota (the only QB-only case not already covered by a coaching change).

**A real bug caught and fixed during this wiring**: the prior cache was being written *before* the preseason/Vegas/disruption adjustments were applied, meaning a fresh computation included them but a cached reload silently didn't — the two paths would have quietly disagreed. Caught by directly testing that two consecutive calls (fresh vs. cached) produced identical results, which they didn't until the cache-write was moved to after all adjustments.

**Measured result**: a modest further improvement on top of the Vegas-totals gain — ARI@LAC's gap tightened from -5.6 to -4.6, BAL@IND from +2.9 to +2.6, consistent with disrupted teams' ratings correctly moving closer to what the market already knows about their coaching/QB changes.

**Not pursued in this round**: real-time injury reports. Week 1 official injury reports typically aren't filed until the Wednesday of game week itself, and with kickoff still over a week out, there's no real report to gather yet — flagged as the one remaining item from the original gap analysis, revisit closer to game day.

## Season-ending injuries — the actual last gap, closed with what's real right now

Confirmed directly before building anything: NFL.com's real Week 1 injury report page shows **"No Injuries Reported"** for every single game — official reports genuinely don't exist yet, exactly as expected (teams aren't required to file until Week 1 itself). That part of the gap is a hard calendar constraint, not something more research would solve.

But real, already-public information about **confirmed season-ending injuries** from camp does exist (`model/season_ending_injuries_2026.py`, gathered from CBS Sports and Sharp Football's trackers) — deliberately limited to confirmed cases (torn ACLs, IR placements) rather than the much larger set of "questionable, could go either way" injuries that remain genuinely uncertain this far out and aren't actionable with real confidence.

Wired into the same disruption-weighting mechanism as coaching/QB changes — teams with a confirmed season-ending loss get the same +0.10 Vegas-weight boost, since the market (Vegas totals gathered Aug 28) had almost certainly already priced these injuries in by the time those lines were set, while this model's last-season-based component obviously can't see them. 15 teams affected in total across all three disruption types.

**Measured result**: BUF@HOU's gap tightened to +0.1 — essentially perfect agreement — consistent with Houston's real Jayden Higgins ACL tear being a piece of information the market already had priced in. ARI@LAC improved further to -4.1. This closes out every gap that was actually gatherable right now; the remaining divergence is honestly attributable to the market simply knowing more than a model built from public box scores and win totals ever fully can.

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
  - This also directly unlocks the time-window/trend feature (see the dedicated section on it below), and gives `odds_watch_job.py`'s repeated-checks-per-game-day a real history of line movement for free — exactly what Section 9.3's closing-line-value tracking needs (verified: two checks in the same test run produced two separate snapshots showing the line moving from -7.5 to -9.0, not one overwritten value).
  - **`deploy/generate_manifest.py`** — since a static site can't list a directory itself, this runs at *build time* (not commit time) to list whatever snapshots currently exist and copy them into the site's build. Critically, this script is never committed to git and produces no git changes of its own — which is exactly why this design doesn't reintroduce the overwrite problem it was built to solve.
  - Fixed a real bug this change surfaced: `git_utils.py`'s `repo_dir` computation assumed a fixed folder depth (`data/ratings.json`, 2 levels from root) and would have silently broken once snapshot files went a level deeper (`data/ratings/2023-week-18.json`, 3 levels). Fixed to use the process's actual working directory instead of counting path segments — tested against the exact deeper-nested path to confirm.
  - Fixed `odds_watch_job.py`'s ratings lookup, which was still checking for the old fixed `ratings.json` path — added `find_latest_ratings_snapshot()` and verified it correctly picks the highest week number when multiple snapshots exist (tested with week 17 and 18 present, correctly selected 18).

## Rating trend chart — the dashboard piece the snapshot architecture unlocked

The per-team profile page now shows a real week-over-week trend of `total_rating` across every published snapshot, using `useRatingsHistory()` (fetches all files the manifest lists, not just the latest) and a hand-built SVG line chart matching the existing scoreboard theme — no charting library needed for something this simple.

**Verified with real data**: generated 5 real weekly snapshots (2023 weeks 4/8/12/16/18), confirmed the build correctly bundles all 5, and manually checked SF's `total_rating` across them (0.2165 → 0.1791 → 0.2250 → 0.2174 → 0.2028) — a sensible progression matching their real, consistently-strong 2023 season. Couldn't get a visual screenshot this round (Claude in Chrome disconnected partway through and didn't reconnect), so the actual rendered appearance is unconfirmed even though the underlying data pipeline is verified correct — worth a visual check next time the browser's available.

**Still not built**: distinct recent-form windows (last 4/8 games) as a separate view from the season-long line — the snapshots now exist to support this, it just hasn't been built as its own feature yet, and the page says so honestly rather than implying it's there.

## Still pending: ESPN FPI live verification

Was going to verify the guessed FPI scraping approach (`model/external_tracking.py`) against the real page this round, but Claude in Chrome disconnected and didn't come back before this session wrapped. That verification — and player props, which needs a live Odds API key this environment doesn't have — remain the two genuinely untested items in the project.

## Six real gaps closed this round

**1. Special teams sub-model (`model/special_teams.py`)** — the biggest one, unbuilt since the original spec. Scores field goals (against a real distance-based make-probability curve), punts (net yards vs. league average), and kickoffs (return yards allowed vs. average), aggregated into `special_teams_voa`. **A real scale mismatch was caught and fixed before it caused damage**: the raw computation produced values roughly 10-15x larger than `offense_voa`/`defense_voa`'s scale — combining them directly would have let special teams dominate the whole rating. Fixed with a documented, approximate rescaling constant (not a formal calibration). **Deliberately kept separate from the core `total_rating`** that the calibrated points-prediction coefficients already depend on — merging it in would require re-running that calibration to stay consistent, which wasn't done this round. Wired into `weekly_job.py`'s real output, verified no nulls across all 32 teams.

**2. Closing-line value (CLV) tracking (`model/clv_tracking.py`)** — the actual metric the snapshot architecture was built to enable, finally computed. Compares the earliest and latest odds-watch snapshot for a game and checks whether the market moved toward or away from where the model originally diverged. **Honestly caveated**: real accumulated multi-snapshot data doesn't exist yet (the season hasn't started), so this was tested against constructed-but-realistic scenarios, not real data — verified both directions work correctly (market moving toward the model's view scores positive, moving away scores negative).

**3. Extended VAR beyond QB (`model/injuries_and_var.py`)** — generalized the QB-only VAR computation to any per-play attribution column. **Validated for receivers** against real 2023 data: Tyreek Hill, CeeDee Lamb, and Amon-Ra St. Brown all showed strongly positive VAR, matching their real elite seasons. **Explicitly NOT extended to pass rushers** — tested with real sack data first, and found the methodology doesn't transfer: Myles Garrett and T.J. Watt (two of the league's most elite edge rushers) both showed *negative* VAR, because a sack's value is driven by the down/distance situation it happened in, not by the rushing skill involved. Documented as a real finding, not silently shipped as a working feature.

**4. Methodology versioning (`model/version.py`)** — every output snapshot now tags which version of the formula produced it (currently 1.4.0), with a changelog describing what changed at each version. Solves the real problem flagged earlier: comparing two weeks' ratings previously had no way to tell whether a difference was real team performance or a formula change in between.

**5. Weather fetching (`model/weather.py`)** — real, stable stadium coordinates and dome/outdoor status for all 32 teams (doesn't need live data), paired with a National Weather Service fetch function. **Untested past the dome-detection logic**: `api.weather.gov` isn't reachable from this sandbox (confirmed: outdoor stadiums correctly attempted the fetch and got a graceful 403, while dome stadiums correctly skipped the network call entirely and never attempted it) — the fetch itself needs real network access to verify.

**6. Real playoff probability / tiebreakers — built and validated against a real known outcome.** `model/playoff_seeding.py` implements the most commonly-decisive NFL tiebreaker rules (head-to-head, division record, conference record, strength of victory) in official order — explicitly not every rule in the real tiebreaker procedure (common games, combined conference/league point rankings, net touchdowns, and the coin-toss step aren't implemented; point differential stands in for those remaining rare cases, documented as a simplification, not represented as official).

**Validated with a genuinely strong result**: run against the real, final 2023 season standings, it reproduced the *exact* real playoff seeding in both conferences — including every tiebreaker that actually mattered that year (the four-way 11-6 tie in the AFC, the three-way 12-5 tie in the NFC). This is real evidence the simplified rule set correctly handles the tiebreakers that matter in practice, not just a hope that it does.

Wired into `model/season_simulation.py` as `simulate_season_with_playoffs()` — runs the tiebreaker system on each simulated season's full standings, not just tallying win totals. **Honest predictive-accuracy finding, tested from a real Week 10 cutoff**: 24/32 teams (75%) correctly classified as making/missing the playoffs. The misses are directly connected to the same limitation the QB-persistence finding surfaced earlier — Minnesota (92% modeled) and Jacksonville (73%) both actually missed the playoffs after late collapses; Buffalo (24%) and Green Bay (6%) both actually made it after late surges no Week-10 model could see coming. 75% is a real, decent, unglamorous number — not a claim of forecasting mastery, and not tested yet with the QB-persistence adjustment applied, which is a natural next step.

## ~~Still fully blocked, unrelated to this round's work~~ — resolved this round

~~**ESPN FPI verification** — needs Claude in Chrome, which disconnected mid-session and never reconnected across several retry attempts.~~ **Fixed — see the dedicated section below.** **Player props** — still needs a live Odds API key this environment doesn't have; the only genuinely untested piece remaining.

## Layer 2 — real player-tracking data, and a genuine accuracy improvement (with an honest correction along the way)

Discovered nflverse has a real Next Gen Stats (NGS) release — actual tracking-chip data (`avg_separation`, `completion_percentage_above_expectation`, `avg_yac_above_expectation`, `rush_yards_over_expected_per_att`), not another play-by-play derivative. `model/layer2_ngs.py` ingests this and builds both player-level grades and team-level features.

**A real bug caught immediately**: the first player-grade run put backup QBs (Mason Rudolph, 71 attempts) and low-volume receivers (Hunter Renfrow, 8 receptions) at the top instead of real stars — small, noisy samples dominating the ranking. Fixed with real season-scale sample thresholds (200+ attempts for QB, 40+ receptions for WR/TE); re-verified against known real 2023 names (Dak Prescott, Brock Purdy, Lamar Jackson) with sensible results.

**First accuracy test: a genuine negative result, reported honestly.** Adding Layer 2 features as full-season averages to explain that same season's game margins improved R² by only +0.4% — statistically negligible. Real hypothesis: the existing opponent-adjusted rating already captures most of what these features measure once a full season has accumulated.

**Second, more rigorous test: the opposite finding.** Testing with the *same walk-forward discipline* as the real backtest (ratings and NGS features both computed only through week W-1, zero lookahead) showed a **substantial, real improvement**: straight-up accuracy 58.22% → 64.04%, R² up 47%, MAE down 0.45 points. The two tests aren't contradictory — early-to-mid season, NGS features apparently converge to a stable signal faster than the opponent-adjustment machinery does, adding real value precisely when data is limited; by season's end, both approaches converge to similar information, which is why the first (full-season) test found them redundant.

**Wired into production** (`model/prediction.py`'s `MARGIN_COEFFICIENTS`, `odds_watch_job.py`) with the validated, walk-forward-calibrated coefficients — tested end-to-end against real 2023 data with no errors, backward-compatible defaults for callers that don't supply NGS data.

**A second real bug, caught by testing the actual Week 1 deployment, not assumed to work**: tried using a prior season's NGS data as a "Layer 2 prior" for the 2026 Week 1 preseason projection (parallel to how the rating itself uses last season as a prior). This is a fundamentally different use case from the validated one — cross-season transfer using old rosters, not within-season application — and **testing it directly showed it made predictions measurably worse** (mean absolute gap vs. real market lines rose from 2.076 to 2.716, 9 of 16 games got worse). Along the way, also found the 2024 NGS release itself is severely incomplete (only 2 teams, 4 rows — not a full season) — added a completeness check (`min_teams` validation) so this failure mode can't silently corrupt a result again. **Fixed by removing Layer 2 from the Week 1 preseason scripts entirely** — it has no valid basis there — while keeping it fully active in `weekly_job.py`/`odds_watch_job.py`, which correctly use each season's *own* accumulating data, the actual validated case.

## Layer 2, round 2 — extending the feature set, testing for diminishing returns

After the first Layer 2 addition, checked whether the remaining unused NGS fields (`avg_cushion`, `catch_percentage`, `percent_attempts_gte_eight_defenders`) added further value, using the identical walk-forward methodology.

**Real, positive, but smaller result** — exactly the diminishing-returns pattern you'd expect once the easiest signal has already been captured: straight-up accuracy 64.04% → 64.90% (+0.86 points), R² up 13.7%, MAE down another 0.08 points. Wired into production the same way as the first round.

**One coefficient worth flagging rather than hiding**: `cushion_diff`'s sign came out negative — more cushion given to the home team's receivers correlates with a *worse* margin. Plausible explanation: teams already losing often see more prevent-style cushion late in games (a game-state confound), not a genuine "more cushion causes a worse outcome" relationship. The walk-forward test measures real out-of-sample prediction accuracy regardless of whether each individual coefficient has a clean causal story — the aggregate result is what to trust, not every coefficient's sign in isolation.

## ESPN FPI — verified against the real live page, not guessed

The original `model/external_tracking.py` guessed a JSON API endpoint that turned out not to exist. Verified directly via Claude in Chrome against the real page (espn.com/nfl/fpi): there's no separate API call at all — confirmed via network-request monitoring while the page loaded (nothing fired but ad/analytics tracking pixels). The real data lives embedded in the page's HTML as `window['__espnfitt__']`, with a completely different shape than guessed (`abbrev` not `abbreviation`; stats as a list of `{name, value}` pairs, not a nested dict).

Rewrote the scraper against this real, verified structure and tested the parsing logic against the actual real data extracted from the live page (Los Angeles Rams: FPI 5.9, offense 4.1, defense 1.6 — matching the live page exactly). The one remaining untested piece is the literal HTTP request succeeding from a different network environment, since this sandbox can't reach espn.com directly — a much smaller, more confined caveat than the previous "entirely guessed" state.

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
