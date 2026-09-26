# Dashboard v2 — build brief

**Date:** 2026-09-26  
**Owner:** Pedro  
**Design reference:** `design/coinflip-v2/` (see its README)  
**Scope:** display and context only. No price, coefficient, tier band or stake changes.

## Ground rules

These carry over from the existing repo contracts.

1. **The site computes nothing.** Every number, rank, label and flag arrives in `data/site/*.json` from an
   export script. JSX formats and lays out. That includes ranks, "in the price" labels and the choice of
   which injuries to show.
2. **Every context item says whether the model uses it.** Carry `in_price: true|false` per item, and set it
   from the model's actual inputs (`GameFeatures`), not from what seems likely. A display field is never an
   input.
3. **Nothing here becomes a model input.** Anything that would move a number goes through the evidence gate
   separately (preregistered test, held-out grade vs the close, ADR).
4. **Soft-fail every new source.** A missing feed shows "no report" or "unknown", never "healthy" or zero. It
   never blocks a board export.
5. **Terminology** follows 9e4211a: "Gap", not "Edge". Tiers are Play / Lean / Coin flip / No edge. The board
   offers suggestions; it doesn't give instructions.
6. **Repo hygiene.**
   - Stage named paths only.
   - Never commit regenerated `data/site/*.json` by hand.
   - Don't touch `.github/workflows/*`.
   - No force push.
   - `bash scripts/check.sh` must pass before each commit.
   - Regenerate `tests/MANIFEST.txt` when tests are added or renamed.
7. **Bundled cleanup.** Tests currently rewrite `data/site/*.json` in the working tree. Point them at a temp
   dir as part of this build.

## The 14 items

Each item gives what to build, where the data comes from, and how to know it's done. File paths are as of
origin/main a5ce240.

### 1–2. Key-player flag ("Not in the price")

**What.** A strip on any game card, in any tier, when a key player is out, doubtful, returning, or (for
goalies) unconfirmed. It is labelled `NOT IN THE PRICE · <TEAM> <ROLE>` with one sentence of explanation.
The key player by league:
- NFL and CFB: QB1
- NHL: starting goalie
- NBA: starters

**Data.**
- **NFL:**
  - Alerts come from `deploy/qb_status.get_qb_alerts`, which already produces alerts for the legacy
    odds-watch. It never reached the new board, which is the bug to fix.
  - Call it from `scripts/export_board.py` (`_nfl_context`) and emit
    `context.key_player: [{team, role, text, source}]`.
- **CFB:** the same shape, from the logic in `deploy/cfb_odds_watch.py`.
- **Manual overrides:** extend `data/qb_overrides.json` with `"nhl": {TEAM: {"alert": "..."}}` and
  `"nba": {TEAM: {"alert": "..."}}`. For v2, NHL and NBA flags come from overrides plus the NBA players
  feed (`data/site/players_nba.json`, starters Out/Doubtful/Questionable). There's no automated goalie
  feed yet; say so in the NHL side panel.
- **Known data issue.** nflverse `games.csv` lists Tua Tagovailoa as ATL's week-2 starter, but team reports
  say Cooper Rush started weeks 1–2. The "started last game" alert text inherits feed errors.
  - Prefer depth chart plus override, and show the source.
  - Test: an override beats every feed.

**Done when:**
- A week with a QB1 Out renders the strip on that game's card.
- Removing the alert removes the strip.
- The price is unchanged, verified by a test that compares `model` before and after.

### 3. NBA Players tab

This has already landed (`players_nba.json`, `NbaPlayers.jsx`). Check that it matches
`NBAPlayers.dc.html`, and that the key-player strip on NBA cards links to it.

### 4. MLB

No key-player flag. The probable starter is already in the price, and a game with no probable is refused.
Scratches already show.

### 5. Venue and weather on every card

| League | Venue | Weather | `in_price` |
|---|---|---|---|
| NFL | `games.csv` `stadium`, `roof`, `surface` | `deploy/game_context.py` Open-Meteo fetch at kickoff hour (temp, wind, precip) | weather **false**: wind feeds only the totals model, which is withheld |
| CFB | venue needs a source (CFBD or ESPN); soft-fail to none | Open-Meteo by venue coordinates when known | false |
| MLB | park name: add `venue.name` to `model/ingest/mlb_slate.py` (MLB Stats API) | Stats API game feed `weather` or Open-Meteo | park **true** (park factor); weather **false** |
| NBA | ESPN venue | none, indoors | — |
| NHL | NHL API venue | only for outdoor games | false |

Domes and closed roofs say "no weather". **Done when** every card shows venue plus weather or "indoors".

### 6. Injury report on every card

**Sources:**
- NFL: `deploy/game_context.fetch_espn_injuries` (already in teams.json).
- CFB: ESPN (partial).
- NBA: `players_nba.json`.
- NHL and MLB: ESPN injuries for those leagues, with soft-fail. Check availability first.

**Sort order:** Out, then Doubtful, then Questionable (NBA adds Probable). Within a status, key positions
first; for the NFL: QB, WR, RB, TE, OT, G/C, DE, DT, LB, CB, S.

**Limits:**
- Play cards show 4 per team, other cards 3, plus "+N more".
- Coin flip cards show counts only.
- MLB shows the IL and day-to-day; NHL shows IR and day-to-day.

**Labels:** always "Not in the price". CFB also says "a missing report means unknown, not healthy".

### 7. Teams tabs: every team clickable

Clicking a team opens a detail panel. Compute every rank in the export (a new `teams_<league>.json`, or
extend `board.teams`).

- **NFL.** Source: the ratings snapshot plus teams.json.
  - Header: model rating with its rank of 32 and the 90% range (`rating_p05`–`rating_p95`).
  - Model rating: offense, defense and special-teams VOA.
  - Efficiency: EPA per play and success rate, for and against.
  - Ball security and schedule: turnover margin, red-zone points per trip, schedule played and remaining.
  - Also: QB1, injury counts, results and the next 3 games.
  - Rank direction:
    - Lower is better for defense VOA, EPA allowed and success allowed.
    - Schedule rank 1 means the hardest.
    - Null stats show "—" with no rank.
  - Division place comes from win % and then point differential.
  - Include a footnote: this early, rating and efficiency can disagree, and only the rating moves the number.
- **NBA.** Source: the 2025-26 season file the model already reads (`data/raw/sportsdataverse/nba_2026.parquet`,
  STD games).
  - Ranked of 30: record, net per game, points scored, points allowed, final model rating.
  - Also: conference place, home and road records, last 10, games within 5 points.
  - Opening rating and first game.

### 8. CFB board (new view)

**Content.**
- Tiers from `board_cfb.json`.
- A banner: "EARLY SEASON" in weeks 1–4, with the CFB band explanation.
- Play, Lean and Coin flip cards.
- Full slate grouped by kickoff window, in Eastern time:

  | Window | Start time |
  |---|---|
  | Friday night | any Friday kickoff |
  | Saturday noon | before 3 pm |
  | Saturday afternoon | 3–6:59 pm |
  | Saturday prime time | 7–9:59 pm |
  | Saturday late | 10 pm or later |

- **Side panel: not priced, with the reason.** Each count comes from `status.game_refusals` plus the games
  with no market:
  - no line captured yet
  - FCS opponent with no rating
  - whole-number spread
  - already kicked off
  - neutral site

**"Check before you trust it" flag.**
- Show it when |model margin − market margin| exceeds the league's 99th percentile of historical |gap| in
  points.
- Derive that threshold in `model/derive_tier_thresholds.py` from the same backtest, and write it to
  `data/tier_thresholds.json` as `outlier_points` per league. Don't hardcode it.
- Display only. The tier still follows the rule.
- Real example, week 4: Missouri State +34.5 at SMU. The model has SMU by 1.3 and the market SMU by 34.5, a
  33-point gap. It's likely a thin rating (Missouri State is in its second FBS season), not an edge.

**Header:** the CFB tab links to the new view.

### 9. Matchup page (game detail)

**Entry and route.** "Open matchup" on every card. Route `/<league>/game/<game_id>`.

**Content, all exported by the core:**
- The model's margin distribution. Emit `model.margin_pmf` over a window (NFL/CFB −28..+28; NBA wider; NHL
  and MLB their natural support) from `ScoreDistribution.margin_pmf`. The site draws bars and must not
  compute a pmf.
- Key numbers highlighted where the league has them.
- Every market with market vs model probability. Totals show "Not priced: the totals model failed its grade"
  where `total_validated` is false.
- Both teams side by side, using the ranks from item 7.
- Line history (item 10), context (items 5 and 11), both injury reports (item 6).

### 10. Line since open, on every card

- **Source:** `BronzeStore` capture history (`src/coverline/execution/bronze.py`, `.snapshots(sport)`).
- **Export:** per game and headline market, `line_history: [{captured_at, line, price}]`, plus `fair_line`
  (the model's).
- **Drawing:** the site draws a sparkline with the fair line dashed. The close is marked once frozen.
- **Honest gaps:** use `store.gaps()`. Never interpolate. "1 capture so far" is a valid state.

### 11. Rest and travel, on every card

| League | Rest | Travel | `in_price` |
|---|---|---|---|
| NFL | `games.csv` `away_rest`, `home_rest` | great-circle distance between home venues, from a static `data/venues.json` | rest **true** (`rest_diff` is in the live rating-only vector, coefficient 0.1310); travel false |
| NBA / NHL | from the schedule (NBA `_nba_context` already computes rest days and back-to-backs) | distance | false |
| MLB | day game after a night game, from schedule start times | — | false |

### 12. Season outlook on Teams

- **NFL.** After each ratings snapshot, the weekly job runs `model/season_simulation.simulate_season_with_playoffs`
  (played games from `games.csv`, remaining schedule, `total_rating`, `rating_std`,
  `MARGIN_COEFFICIENTS_V1_RATING_ONLY`, n=2000, fixed seed).
  - Output: `data/site/outlook_nfl.json` with `{team: playoff_pct}`, the ratings version and n.
  - Runtime is about 3.5 minutes. Run it in the weekly job, not on page load.
  - Show playoff odds only. The simulator's division tally is a proxy (the #1 seed's division), so don't show
    division odds.
  - **Flag for Pedro:** the simulator uses a residual SD of 13.0, where the model's `MARGIN_SD` is 13.2979.
    Align it or document it in an ADR. Don't change it silently.
  - Label: "2,000 simulated seasons from week-N ratings. QB changes aren't in it."
- **NBA.** "Not simulated yet." An NBA version with the play-in needs its own check before any number shows.

### 13. The model on this team

- `scripts/export_record.py` adds per-team `{settled, wins, mean_clv}` for teams on either side of settled
  paper trades.
- The UI hides the numbers until `settled >= 10` and shows "N of 10 settled" instead.

### 14. MLB "Pitchers & parks" and NHL "Attack & defence"

- **MLB.**
  - Replay with `coverline.leagues.mlb.live.state_before(sched, pit, day)`.
  - Clickable list of 30 teams, with a panel showing:
    - `offense(team)`, runs per game
    - `pen_ra27(team)`, with credibility `outs/(outs+900)`
    - home `park_factor`
    - ranks for each of the above
    - bullpen workload: relief outs over the last 3 days from the pitching cache, marked **not in the price**
    - a "How a game is priced" note (starter 58% of outs, bullpen 42%, times park, shrunk to league average)
  - Today's probables table: `sp_ra27(pid)` plus own-record share `outs/(outs+380)`. TBD starters are
    marked refused. Team names come from slates.
- **NHL.**
  - Clickable list of 32 teams, using 2025-26 from `model/fit_nhl_rules.load((2026,))`:
    - no-pull GF/GA per game (the rates the model is built on)
    - pulled-goalie goals for and against
    - points
    - games past regulation
  - Banner: "NOT USED BY THE MODEL", because ratings reset each season.
  - Switch to current-season rates once every team has 10 games.

### Header

- CFB links to its board.
- The live/sample pill takes the as-of date from the data.

## Suggested commits

Each commit passes `scripts/check.sh`.

1. **Context in the core export.** Items 1, 5, 6, 11: `key_player`, `venue`, `weather`, `injuries` and `rest`,
   each with `in_price`. Includes tests (price unchanged; soft-fail; override wins) and the temp-dir fix for
   `data/site` in tests.
2. **Distribution and line history.** Items 9 and 10: `margin_pmf`, `line_history`, `fair_line`.
3. **Teams, outlook, record and league pages.** Items 7, 12, 13, 14: new exports and the weekly-job hook.
4. **Frontend.** Card upgrades, matchup route, Teams panels, MLB and NHL tabs.
5. **CFB board and header.** Item 8, plus the `outlier_points` derivation.

## Deploy and verify

- **Deploy.** Push to main. The Render static site rebuilds, and the cron jobs pick up export changes on their
  next run.
- **Verify** after deploy:
  - Each `board_*.json` carries the new fields.
  - The site renders all five leagues with no console errors.
  - A card with no context shows empty states, not zeros.
  - No `data/site` file was committed by hand.
  - `CAPTURE_ENABLED` is still "1".
- **Report back:** commit hashes, what shipped, what soft-failed and why, and any item deferred with its
  reason.
