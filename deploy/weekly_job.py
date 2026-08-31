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
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.nfl_pbp import load_season
from ingest.nfl_schedules import load_schedules, get_current_week
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time,
    add_home_field_and_rest, team_ratings,
)
from deploy.validate import validate_pbp_data, validate_ratings, validate_git_push_succeeded, ValidationError
from deploy.notify import report_success, report_failure

REPO_DATA_PATH = os.environ.get("REPO_DATA_PATH", "./data")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GIT_REPO_URL = os.environ.get("GIT_REPO_URL")  # e.g. "github.com/username/repo.git" — no scheme/token


def run_pipeline(season: int, current_week: int) -> dict:
    """Runs the full Layer 1 pipeline through the current week and
    returns the ratings + metadata needed for the commit."""
    df = load_season(season)
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

    return {
        "ratings": ratings,
        "season": season,
        "week": current_week,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def write_output(result: dict, path: str):
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, "ratings.json")

    payload = {
        "season": result["season"],
        "week": result["week"],
        "computed_at": result["computed_at"],
        "ratings": result["ratings"].reset_index().rename(columns={"index": "team"}).to_dict(orient="records"),
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    return output_file


def _normalize_repo_url(repo_url: str) -> str:
    """
    Strip any scheme prefix the user may have included. GIT_REPO_URL is
    documented as bare host+path (e.g. "github.com/user/repo.git"), but
    entering the full URL (e.g. "https://github.com/user/repo") is an
    easy, understandable mistake — normalize either input rather than
    silently producing a malformed "https://TOKEN@https://..." URL,
    which is a real failure this hit in production.
    """
    for prefix in ("https://", "http://"):
        if repo_url.startswith(prefix):
            return repo_url[len(prefix):]
    return repo_url


def git_commit_and_push(file_path: str, season: int, week: int) -> None:
    """
    Section 9.1's handoff mechanism — commit the generated data file
    and push, which triggers Render's static site auto-deploy.

    IMPORTANT: Render's automatic git checkout does NOT leave a named
    'origin' remote configured the way a normal `git clone` would —
    confirmed directly by a real "No such remote 'origin'" failure in
    production. This adds the remote if missing, falling back to
    updating it if it turns out to already exist.
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))

    subprocess.run(["git", "config", "user.name", "football-model-bot"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "bot@football-model.local"], cwd=repo_dir, check=True)

    repo_url = os.environ.get("GIT_REPO_URL")  # e.g. "github.com/username/repo.git" — no scheme/token
    token = GITHUB_TOKEN
    if repo_url and token:
        repo_url = _normalize_repo_url(repo_url)
        authenticated_url = f"https://{token}@{repo_url}"

        add_result = subprocess.run(
            ["git", "remote", "add", "origin", authenticated_url],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if add_result.returncode != 0:
            # Most likely cause: origin already exists in this environment
            # (e.g. local testing) — fall back to updating it instead.
            subprocess.run(
                ["git", "remote", "set-url", "origin", authenticated_url],
                cwd=repo_dir, check=True,
            )
    else:
        print("[weekly_job] warning: GIT_REPO_URL or GITHUB_TOKEN not set — push will likely fail "
              "against Render's default read-only clone credential")

    subprocess.run(["git", "add", file_path], cwd=repo_dir, check=True)

    commit_result = subprocess.run(
        ["git", "commit", "-m", f"Update ratings: {season} week {week}"],
        cwd=repo_dir, capture_output=True, text=True,
    )
    # A "nothing to commit" exit is fine (data unchanged) — anything else isn't.
    if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
        raise ValidationError(f"git commit failed: {commit_result.stderr}")

    # -u origin HEAD rather than a bare `git push`, since a freshly-added
    # remote has no upstream tracking branch configured yet — this pushes
    # the current branch to a same-named branch on origin and sets
    # tracking, without needing to know the branch name in advance.
    push_result = subprocess.run(
        ["git", "push", "-u", "origin", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
    )
    validate_git_push_succeeded(push_result.returncode, push_result.stderr)


def main():
    season = int(os.environ.get("SEASON", datetime.now().year))
    # Auto-detects the current week unless explicitly overridden — the
    # cron job doesn't need a manually-updated env var every week.
    week = int(os.environ.get("CURRENT_WEEK") or get_current_week(season))

    try:
        result = run_pipeline(season, week)
        output_file = write_output(result, REPO_DATA_PATH)

        if GIT_REPO_URL:
            git_commit_and_push(output_file, season, week)
        else:
            print("[weekly_job] GIT_REPO_URL not set, skipping commit/push (local-only run)")

        report_success("weekly_ratings_job", summary=f"{season} week {week}, {len(result['ratings'])} teams")

    except ValidationError as e:
        report_failure("weekly_ratings_job", error=f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        report_failure("weekly_ratings_job", error=f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
