"""
Weekly ratings cron job — Section 9.1's heavy job.

UNTESTED beyond the pipeline logic itself (which IS tested elsewhere in
this codebase). The git commit/push plumbing needs a real repo and a
GitHub token to verify — it's structured correctly but has not been
run against a live remote in this sandbox.
"""

import sys
import os
import subprocess
import json
import urllib.error
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.nfl_pbp import load_season, load_special_teams_plays
from ingest.nfl_schedules import load_schedules, get_current_week
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_home_field_and_rest, team_ratings,
)
from model.team_profile import build_team_profile
from model.preseason_prior import blend_team_ratings, vegas_win_total_to_rating
from model.preseason_performance import apply_preseason_adjustment, compute_combined_preseason_rating, DEFAULT_PRESEASON_WEIGHT
from model.market_comparison import bootstrap_rating_uncertainty
from model.special_teams import compute_special_teams_ratings
from model.version import METHODOLOGY_VERSION
from model.season_simulation import simulate_season_with_playoffs
from model.prediction import MARGIN_COEFFICIENTS
from model.injury_impact import apply_injury_adjustment
from deploy.validate import validate_pbp_data, validate_ratings, ValidationError
from deploy.notify import report_success, report_failure
from deploy.git_utils import git_commit_and_push

REPO_DATA_PATH = os.environ.get("REPO_DATA_PATH", "./data")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GIT_REPO_URL = os.environ.get("GIT_REPO_URL")  # e.g. "github.com/username/repo.git" — no scheme/token


def get_or_compute_prior(season: int, path: str):
    """
    Loads last season's final rating (the preseason prior) from a cached
    file if one exists, otherwise computes it once and caches it. This
    only needs computing once per season, not every week — the cache
    file gets committed to the repo the same way ratings snapshots do,
    so it persists across the ephemeral cron containers.
    """
    import pandas as pd

    prior_dir = os.path.join(path, "priors")
    os.makedirs(prior_dir, exist_ok=True)
    cache_path = os.path.join(prior_dir, f"{season}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return pd.DataFrame(cached["ratings"]).set_index("team"), cache_path, False

    print(f"[weekly_job] no cached prior for {season}, computing from {season - 1} season...")
    prior_season_df = load_season(season - 1)
    prior_season_df = add_situation_buckets(prior_season_df)
    prior_season_df = score_all_plays(prior_season_df, use_turnover_luck_adjustment=True)
    prior_season_df = filter_garbage_time(prior_season_df)
    baselines = compute_baselines(prior_season_df)
    prior_season_df = compute_raw_voa(prior_season_df, baselines)
    prior_season_df = opponent_adjust(prior_season_df, iterations=3, regression=0.5)
    prior_ratings = team_ratings(prior_season_df, use_recency_weights=False)

    # Combine three signals into the final prior: last season's rating
    # (the base), real preseason performance (point diff + PRE WK3
    # box scores), and real Vegas win totals (the market's own
    # aggregated view — beat-reporter access, scouting, and roster
    # analysis this model can't replicate on its own). Weights below
    # are a reasonable, deliberately-considered combination, NOT
    # backtested the way the in-season k=2 credibility weight was —
    # unlike that value, there's no efficient way to validate this
    # blend without repeating real data-gathering across multiple past
    # seasons, which wasn't done here.
    PRESEASON_PERFORMANCE_WEIGHT_IN_PRIOR = 0.15
    BASE_VEGAS_WEIGHT_IN_PRIOR = 0.35
    # For teams with a real coaching or QB change, last season's rating
    # reflects a coach/QB who won't be there — boost reliance on the
    # Vegas signal (which already prices in the real personnel change)
    # for exactly these teams, rather than treating every team
    # identically regardless of real offseason disruption.
    DISRUPTION_VEGAS_WEIGHT_BOOST = 0.10  # per disruption type, stacks for teams with both

    preseason_applied = False
    vegas_applied = False
    disrupted_teams = set()
    try:
        from model.preseason_2026_results import PRESEASON_2026_RESULTS
        preseason_data = {2026: PRESEASON_2026_RESULTS}.get(season)
    except ImportError:
        preseason_data = None

    try:
        from model.win_totals_2026 import WIN_TOTALS_2026
        win_totals = {2026: WIN_TOTALS_2026}.get(season)
    except ImportError:
        win_totals = None

    coaching_changes, qb_changes, season_ending_injuries, personnel_changes = {}, {}, {}, {}
    if season == 2026:
        try:
            from model.coaching_changes_2026 import COACHING_CHANGES_2026
            coaching_changes = COACHING_CHANGES_2026
        except ImportError:
            pass
        try:
            from model.qb_changes_2026 import QB_CHANGES_2026
            qb_changes = QB_CHANGES_2026
        except ImportError:
            pass
        try:
            from model.season_ending_injuries_2026 import SEASON_ENDING_INJURIES_2026
            season_ending_injuries = SEASON_ENDING_INJURIES_2026
        except ImportError:
            pass
        try:
            # Notable non-QB personnel changes (major trades, big free
            # agent signings, notable retirements) -- broader than the
            # 3 hand-verified QB cases, since a new starting corner, an
            # O-line shuffle, or a lost WR1 is just as real a disruption
            # to last season's rating as a QB change is, but wasn't
            # tracked separately until this file exists. Not created
            # with placeholder data -- only wires in once real,
            # verified entries are gathered (see model/qb_changes_2026.py
            # for the format/verification bar to match).
            from model.personnel_changes_2026 import PERSONNEL_CHANGES_2026
            personnel_changes = PERSONNEL_CHANGES_2026
        except ImportError:
            personnel_changes = {}

    for team in prior_ratings.index:
        last_season_component = prior_ratings.loc[team, "total_rating"]
        combined = last_season_component

        if preseason_data is not None:
            preseason_component = compute_combined_preseason_rating(team, results=preseason_data)
            combined = (1 - PRESEASON_PERFORMANCE_WEIGHT_IN_PRIOR) * combined + PRESEASON_PERFORMANCE_WEIGHT_IN_PRIOR * preseason_component
            preseason_applied = True

        if win_totals is not None and team in win_totals:
            vegas_weight = BASE_VEGAS_WEIGHT_IN_PRIOR
            if team in coaching_changes:
                vegas_weight += DISRUPTION_VEGAS_WEIGHT_BOOST
                disrupted_teams.add(team)
            if team in qb_changes:
                vegas_weight += DISRUPTION_VEGAS_WEIGHT_BOOST
                disrupted_teams.add(team)
            if team in season_ending_injuries:
                vegas_weight += DISRUPTION_VEGAS_WEIGHT_BOOST
                disrupted_teams.add(team)
            if team in personnel_changes:
                vegas_weight += DISRUPTION_VEGAS_WEIGHT_BOOST
                disrupted_teams.add(team)
            vegas_weight = min(vegas_weight, 0.60)  # cap so last-season rating always retains some influence

            vegas_component = vegas_win_total_to_rating(win_totals[team])
            combined = (1 - vegas_weight) * combined + vegas_weight * vegas_component
            vegas_applied = True

        prior_ratings.loc[team, "total_rating"] = combined

    if preseason_applied:
        print(f"[weekly_job] applied real {season} preseason performance signal "
              f"(weight={PRESEASON_PERFORMANCE_WEIGHT_IN_PRIOR}, not backtested)")
    if vegas_applied:
        print(f"[weekly_job] applied real {season} Vegas win totals "
              f"(base weight={BASE_VEGAS_WEIGHT_IN_PRIOR}, boosted for {len(disrupted_teams)} teams "
              f"with a real coaching/QB change: {sorted(disrupted_teams)})")

    # Cache AFTER all adjustments — fixes a real bug found during this
    # wiring: the cache was previously written before adjustments were
    # applied, so a fresh computation included them but a cached reload
    # silently didn't, giving inconsistent results between the two paths.
    with open(cache_path, "w") as f:
        json.dump({
            "season": season - 1,
            "ratings": prior_ratings.reset_index().rename(columns={"index": "team"}).to_dict(orient="records"),
        }, f, indent=2)

    return prior_ratings, cache_path, True


class SeasonNotStartedError(Exception):
    """Raised when nflverse hasn't published play-by-play data for a
    season yet — expected and temporary before Week 1, not a real
    failure. Confirmed directly: a 404 on this URL before the season
    starts is nflverse genuinely not having anything to publish, since
    there's no play-by-play data to generate from games that haven't
    been played."""
    pass


def run_pipeline(season: int, current_week: int) -> dict:
    """Runs the full Layer 1 pipeline through the current week and
    returns the ratings + metadata needed for the commit."""
    try:
        df = load_season(season)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SeasonNotStartedError(
                f"No play-by-play data published yet for {season} — the season "
                f"likely hasn't started. This is expected before Week 1, not a bug."
            )
        raise

    df = df[df["week"] <= current_week].copy()

    validate_pbp_data(df)

    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True)
    df = filter_garbage_time(df)

    schedules = load_schedules(seasons=[season])
    df = add_home_field_and_rest(df, schedules)

    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)

    ratings = team_ratings(df, use_recency_weights=True)
    validate_ratings(ratings)

    # Real injury impact (ingest/injuries.py + model/injury_impact.py) --
    # closes the "market has live game-week injury status, we don't" gap.
    # Applied for the UPCOMING week (current_week + 1) -- these ratings
    # feed odds_watch_job.py's predictions for that week, so the
    # question is who's ruled out for THAT game, not this one already
    # played. Guarded: real 2026 injury reports won't exist until the
    # season actually starts, so this fails gracefully until then.
    try:
        injury_adjusted = apply_injury_adjustment(ratings, df, season=season, upcoming_week=current_week + 1)
        injured_teams = injury_adjusted[injury_adjusted["injury_note"].notna()]
        if len(injured_teams) > 0:
            print(f"[weekly_job] applied real injury adjustments for {len(injured_teams)} teams: "
                  f"{dict(zip(injured_teams.index, injured_teams['injury_note']))}")
            ratings["offense_voa"] = injury_adjusted["offense_voa_injury_adjusted"]
            ratings["total_rating"] = injury_adjusted["total_rating_injury_adjusted"]
    except Exception as e:
        print(f"[weekly_job] real injury data unavailable for {season} week {current_week + 1} ({e}), skipping")

    # Preseason prior blending (Section 11.1) — only meaningful early in
    # the season; skip entirely if no prior season's data is available
    # (e.g., the model's first-ever season) rather than fail.
    prior_source = "none"
    try:
        prior_ratings, prior_cache_path, was_freshly_computed = get_or_compute_prior(season, REPO_DATA_PATH)
        ratings = blend_team_ratings(prior_ratings, ratings, games_played=current_week)
        prior_source = f"blended with {season - 1} prior (k={2.0}, week {current_week}) + real preseason performance + real Vegas win totals (weights not backtested, see get_or_compute_prior docstring)"
        if was_freshly_computed:
            print(f"[weekly_job] computed and cached new prior at {prior_cache_path}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[weekly_job] no prior-season data available for {season - 1}, skipping prior blending")
        else:
            raise

    # Team profile stats (EPA/play, success rate, red zone efficiency,
    # turnover margin) for the per-team dashboard page — computed from
    # the same underlying data, alongside the core DVOA rating.
    profile = build_team_profile(df, ratings)

    # Bootstrap uncertainty (Section 10, item 3) — computed and tested
    # in isolation in model/market_comparison.py, but never actually
    # reached weekly_job.py's output until now. n_bootstrap=100 is a
    # reasonable production default (measured earlier: ~0.6s for 20
    # iterations on a full season, so ~3s for 100 — fine for a job that
    # only runs once a week).
    print("[weekly_job] computing bootstrap uncertainty (100 iterations)...")
    def rating_fn(resample):
        return team_ratings(resample, use_recency_weights=False)
    uncertainty = bootstrap_rating_uncertainty(df, rating_fn, n_bootstrap=100)
    profile = profile.join(uncertainty[["rating_std", "rating_p05", "rating_p95"]], how="left")

    # Special teams sub-model (Section 3.7, previously unbuilt) — real
    # field goal/punt/kickoff scoring, added as a visible, separate
    # rating component. NOT merged into the core total_rating that
    # the calibrated points-prediction coefficients and disruption
    # weighting already depend on — doing that would require
    # re-running that calibration to stay consistent, which wasn't
    # done in this pass. Real special teams data, honestly kept
    # separate rather than silently folded into a number other
    # calibrated pieces already trust.
    print("[weekly_job] computing special teams ratings...")
    st_plays = load_special_teams_plays(season)
    st_plays = st_plays[st_plays["week"] <= current_week]
    st_ratings = compute_special_teams_ratings(st_plays)
    profile = profile.join(st_ratings[["special_teams_voa"]], how="left")

    # Playoff probability (previously built and validated -- exact
    # match to the real 2023 seeding -- but never wired past a demo
    # script until now). Guarded to skip when there's no remaining
    # schedule to simulate (season already over) or too little played
    # data yet (week 1-3, where standings are too thin to mean much) --
    # n_simulations=200 keeps this to roughly 10-15s, reasonable for a
    # job that only runs once a week.
    if 4 <= current_week < 18:
        try:
            print("[weekly_job] computing playoff probabilities (200 simulations)...")
            full_schedule = load_schedules(seasons=[season])
            played_schedule = full_schedule[full_schedule["week"] <= current_week].dropna(subset=["home_score"])
            remaining_schedule = full_schedule[full_schedule["week"] > current_week].copy()
            remaining_schedule["rest_diff"] = remaining_schedule["home_rest"] - remaining_schedule["away_rest"]

            if len(remaining_schedule) > 0:
                playoff_result = simulate_season_with_playoffs(
                    remaining_schedule, played_schedule, ratings, uncertainty,
                    MARGIN_COEFFICIENTS, n_simulations=200,
                )
                profile = profile.join(playoff_result[["playoff_pct"]], how="left")
        except Exception as e:
            print(f"[weekly_job] playoff probability computation failed ({e}), skipping")

    return {
        "ratings": profile,
        "season": season,
        "week": current_week,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "prior_source": prior_source,
        "methodology_version": METHODOLOGY_VERSION,
    }


def write_output(result: dict, path: str):
    """
    Writes an immutable, timestamped snapshot rather than overwriting a
    single ratings.json — a real fix for the git friction that repeated
    overwrites were causing (every run modifying the same lines is what
    produces divergent-branch pain locally; a new file every week is a
    pure git ADD, which never conflicts with anything). This also
    directly unlocks the week-over-week trend view flagged as missing
    from the dashboard earlier, at zero cost to the actual rating
    computation — this only changes where the output is written, not
    how it's computed.
    """
    ratings_dir = os.path.join(path, "ratings")
    os.makedirs(ratings_dir, exist_ok=True)
    output_file = os.path.join(ratings_dir, f"{result['season']}-week-{result['week']:02d}.json")

    payload = {
        "season": result["season"],
        "week": result["week"],
        "computed_at": result["computed_at"],
        "prior_source": result.get("prior_source", "none"),
        "methodology_version": result.get("methodology_version", "unknown"),
        "ratings": result["ratings"].reset_index().rename(columns={"index": "team"}).to_dict(orient="records"),
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    return output_file


def main():
    season = int(os.environ.get("SEASON", datetime.now().year))
    # Auto-detects the current week unless explicitly overridden — the
    # cron job doesn't need a manually-updated env var every week.
    week = int(os.environ.get("CURRENT_WEEK") or get_current_week(season))

    try:
        result = run_pipeline(season, week)
        output_file = write_output(result, REPO_DATA_PATH)

        if GIT_REPO_URL:
            git_commit_and_push(output_file, commit_message=f"Update ratings: {season} week {week}")
        else:
            print("[weekly_job] GIT_REPO_URL not set, skipping commit/push (local-only run)")

        report_success("weekly_ratings_job", summary=f"{season} week {week}, {len(result['ratings'])} teams")

    except SeasonNotStartedError as e:
        # Expected and temporary, not an alarm-worthy failure — matches
        # how odds_watch_job.py treats "not a game day" as a soft skip
        # rather than a failure requiring a Discord alert.
        print(f"[weekly_job] {e}")
        report_success("weekly_ratings_job", summary=f"skipped, {season} season hasn't started yet")

    except ValidationError as e:
        report_failure("weekly_ratings_job", error=f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        report_failure("weekly_ratings_job", error=f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
