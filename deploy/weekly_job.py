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

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules, get_current_week
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_home_field_and_rest, team_ratings,
)
from model.team_profile import build_team_profile
from model.preseason_prior import blend_team_ratings
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

    # Preseason prior blending (Section 11.1) — only meaningful early in
    # the season; skip entirely if no prior season's data is available
    # (e.g., the model's first-ever season) rather than fail.
    prior_source = "none"
    try:
        prior_ratings, prior_cache_path, was_freshly_computed = get_or_compute_prior(season, REPO_DATA_PATH)
        ratings = blend_team_ratings(prior_ratings, ratings, games_played=current_week)
        prior_source = f"blended with {season - 1} prior (k={2.0}, week {current_week})"
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

    return {
        "ratings": profile,
        "season": season,
        "week": current_week,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "prior_source": prior_source,
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
