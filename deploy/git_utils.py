"""
Shared git commit/push logic — used by both weekly_job.py and
odds_watch_job.py. Extracted here to avoid duplicating the same
hard-won fixes in two places (missing origin remote, detached HEAD,
URL normalization — all confirmed as real production failures and
fixed here, see git history / conversation for the specific errors).
"""

import os
import subprocess

from deploy.validate import validate_git_push_succeeded, ValidationError


def normalize_repo_url(repo_url: str) -> str:
    """
    Strip any scheme prefix the user may have included. GIT_REPO_URL is
    documented as bare host+path (e.g. "github.com/user/repo.git"), but
    entering the full URL (e.g. "https://github.com/user/repo") is an
    easy, understandable mistake — normalize either input rather than
    producing a malformed "https://TOKEN@https://..." URL, which is a
    real failure this hit in production.
    """
    for prefix in ("https://", "http://"):
        if repo_url.startswith(prefix):
            return repo_url[len(prefix):]
    return repo_url


def git_commit_and_push(file_path: str, commit_message: str) -> None:
    """
    Commit a generated data file and push, triggering Render's static
    site auto-deploy on push.

    Two things confirmed as real production failures and fixed here:
    - Render's checkout doesn't leave a named 'origin' remote configured
      the way a normal `git clone` would — this adds it if missing,
      falling back to updating it if it turns out to already exist.
    - Render's checkout leaves the repo in detached HEAD state (checked
      out at a specific commit, not a branch) — pushing to an explicit
      destination branch (GIT_BRANCH, default "main") sidesteps the
      "not a full refname" failure that `-u origin HEAD` hits when HEAD
      isn't attached to a branch.
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))

    subprocess.run(["git", "config", "user.name", "football-model-bot"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "bot@football-model.local"], cwd=repo_dir, check=True)

    repo_url = os.environ.get("GIT_REPO_URL")
    token = os.environ.get("GITHUB_TOKEN")
    if repo_url and token:
        repo_url = normalize_repo_url(repo_url)
        authenticated_url = f"https://{token}@{repo_url}"

        add_result = subprocess.run(
            ["git", "remote", "add", "origin", authenticated_url],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if add_result.returncode != 0:
            subprocess.run(
                ["git", "remote", "set-url", "origin", authenticated_url],
                cwd=repo_dir, check=True,
            )
    else:
        print("[git_utils] warning: GIT_REPO_URL or GITHUB_TOKEN not set — push will likely fail "
              "against Render's default read-only clone credential")

    subprocess.run(["git", "add", file_path], cwd=repo_dir, check=True)

    commit_result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
        raise ValidationError(f"git commit failed: {commit_result.stderr}")

    target_branch = os.environ.get("GIT_BRANCH", "main")
    push_result = subprocess.run(
        ["git", "push", "origin", f"HEAD:{target_branch}"], cwd=repo_dir, capture_output=True, text=True,
    )
    validate_git_push_succeeded(push_result.returncode, push_result.stderr)
