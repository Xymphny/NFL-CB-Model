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
from deploy.validate import validate_pbp_data, validate_ratings, ValidationError
from deploy.notify import report_success, report_failure
from deploy.git_utils import git_commit_and_push

REPO_DATA_PATH = os.environ.get("REPO_DATA_PATH", "./data")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GIT_REPO_URL = os.environ.get("GIT_REPO_URL")  # e.g. "github.com/username/repo.git" — no scheme/token


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

    # Team profile stats (EPA/play, success rate, red zone efficiency,
    # turnover margin) for the per-team dashboard page — computed from
    # the same underlying data, alongside the core DVOA rating.
    profile = build_team_profile(df, ratings)

    return {
        "ratings": profile,
        "season": season,
        "week": current_week,
        "computed_at": datetime.now(timezone.utc).isoformat(),
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
