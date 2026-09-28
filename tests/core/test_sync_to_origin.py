"""Every cron syncs to origin/main before it works.

render.yaml's buildFilter stops data-only commits from rebuilding the crons,
so a cron runs from the checkout of the last CODE commit. These run
deploy/git_utils.sync_to_origin against a real bare remote.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

import deploy.git_utils as G
from deploy.validate import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """A bare 'origin', a stale detached checkout of it (Render's shape), and a
    second clone that has since pushed a data commit."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(origin), str(seed))
    for c in (seed,):
        _git(c, "config", "user.email", "t@t"); _git(c, "config", "user.name", "t")
        _git(c, "checkout", "-b", "main")
    (seed / "data").mkdir()
    (seed / "data" / "a.json").write_text("1")
    _git(seed, "add", "."); _git(seed, "commit", "-m", "code"); _git(seed, "push", "origin", "main")

    stale = tmp_path / "stale"
    _git(tmp_path, "clone", str(origin), str(stale))
    _git(stale, "checkout", "--detach")
    _git(stale, "remote", "remove", "origin")      # Render leaves no usable origin

    (seed / "data" / "b.json").write_text("2")
    _git(seed, "add", "."); _git(seed, "commit", "-m", "bot data"); _git(seed, "push", "origin", "main")

    # configure_origin builds https://TOKEN@<url>; route that back to the bare repo.
    monkeypatch.setenv("GIT_REPO_URL", "example.invalid/repo.git")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{origin}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://tok@example.invalid/repo.git")
    return origin, seed, stale


def test_a_stale_checkout_is_moved_to_the_tip_of_main(repos):
    origin, seed, stale = repos
    assert not (stale / "data" / "b.json").exists()
    G.sync_to_origin(str(stale))
    assert _git(stale, "rev-parse", "HEAD") == _git(seed, "rev-parse", "HEAD")
    assert (stale / "data" / "b.json").read_text() == "2"


def test_it_never_discards_modified_tracked_files(repos):
    _, _, stale = repos
    (stale / "data" / "a.json").write_text("local edit")
    with pytest.raises(ValidationError, match="modified"):
        G.sync_to_origin(str(stale))
    assert (stale / "data" / "a.json").read_text() == "local edit"


def test_it_never_resets_away_unpushed_commits(repos):
    _, _, stale = repos
    _git(stale, "config", "user.email", "t@t"); _git(stale, "config", "user.name", "t")
    (stale / "mine.txt").write_text("x")
    _git(stale, "add", "."); _git(stale, "commit", "-m", "unpushed")
    head = _git(stale, "rev-parse", "HEAD")
    with pytest.raises(ValidationError, match="not on origin"):
        G.sync_to_origin(str(stale))
    assert _git(stale, "rev-parse", "HEAD") == head


def test_without_credentials_it_does_nothing(repos, monkeypatch):
    _, _, stale = repos
    monkeypatch.delenv("GIT_REPO_URL")
    head = _git(stale, "rev-parse", "HEAD")
    G.sync_to_origin(str(stale))
    assert _git(stale, "rev-parse", "HEAD") == head


def _render_crons():
    import yaml
    return [s for s in yaml.safe_load((ROOT / "render.yaml").read_text())["services"]
            if s["type"] == "cron"]


@pytest.mark.parametrize("svc", _render_crons(), ids=lambda s: s["name"])
def test_every_cron_entrypoint_syncs_before_it_works(svc):
    """Named for close-capture-job especially: its snapshots are the one thing
    that cannot be fetched again, and it pushes the most."""
    cmd = svc["startCommand"]
    m = re.search(r"((?:deploy|scripts)/\w+\.py)", cmd) or re.search(r"import (deploy\.\w+)", cmd)
    path = m.group(1)
    path = path if path.endswith(".py") else path.replace(".", "/") + ".py"
    assert "sync_to_origin(" in (ROOT / path).read_text(), f"{svc['name']}: {path} never syncs"


def test_python_services_do_not_rebuild_on_data():
    import yaml
    services = yaml.safe_load((ROOT / "render.yaml").read_text())["services"]
    for s in services:
        paths = (s.get("buildFilter") or {}).get("paths")
        assert paths, f"{s['name']} has no buildFilter and rebuilds on every commit"
        if s.get("runtime") == "python" or s.get("env") == "python":
            assert not any(p.startswith("data") for p in paths), s["name"]
