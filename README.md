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

## Written but NOT testable in this sandbox (need real credentials/network outside it)

- **`ingest/cfb_pbp.py`** — CFBD integration; `api.collegefootballdata.com` isn't reachable here. Verify schema mapping against a live response before trusting it (see prior detailed notes in this file's git history / the spec's Section on CFB)
- **`model/external_tracking.py`** — ESPN FPI scraping; `espn.com` isn't reachable here, and the endpoint/response shape is a best guess, not verified. Also carries the ToS consideration flagged in the spec
- **`model/player_props.py`** — needs a live Odds API key. Two things need verification before building further: whether player props require a plan tier above the $30/mo 20K tier, and actual CFB coverage depth
- **`deploy/odds_watch_job.py`** — needs a live Odds API key; confirmed to fail gracefully without one
- **`deploy/weekly_job.py`**'s git commit/push path specifically — the pipeline portion is tested, the actual push to a remote is not, since there's no real repo to push to here

## Known integration gaps (built separately, not yet wired together)

- **QB persistence isn't fed into the season simulation** — the Minnesota/Cousins case above is the concrete example. `model/injuries_and_var.py`'s `apply_persistent_qb_adjustment()` exists but `demo/run_season_simulation.py` doesn't call it yet
- **The points-prediction layer isn't fed into `deploy/odds_watch_job.py`** — `compute_divergences()` expects a `model_predictions` dict that nothing currently populates; it needs the current week's calibrated prediction, not the historical calibration exercise
- **Preseason prior / credibility weighting (Section 11.1)** — not built. The calibration script's prior-season-only approach is related but isn't the same as the full credibility-weighted blend described in the spec
- **Bootstrap uncertainty isn't fed into `deploy/weekly_job.py`'s output** — it's computed and tested in `model/market_comparison.py` but the weekly job doesn't call it or include it in `ratings.json`
- **Walk-forward backtesting harness (Section 11's protocol)** — not built. `calibrate_points_model.py`'s prior-season-only approach is a simpler stand-in, explicitly caveated in its own output

## Known gaps vs. the full spec (not started)

- Special teams sub-model (spec 3.7) — special-teams plays are filtered out entirely, not scored
- CFB-specific deltas beyond the threshold table (conference-strength substitution, FBS-only baseline pool)
- Player-level target/carry data ingestion (needed for a real player-props projection, not the placeholder currently in `player_props.py`)

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
```
