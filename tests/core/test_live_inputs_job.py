"""The live-inputs cron: one league failing must not stop the others, and an
unchanged day must not commit.

The ingests themselves are tested elsewhere on captured shapes; this is the
orchestration around them, which is where a scheduled job quietly stops doing
its work.
"""

import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy import live_inputs_job as J  # noqa: E402


def _fake_steps(monkeypatch, outcomes, ran):
    def make(name, ok):
        def fn():
            ran.append(name)
            if not ok:
                raise RuntimeError(f"{name} endpoint down")
        return fn
    monkeypatch.setattr(J, "steps", lambda d: [(n, make(n, ok)) for n, ok in outcomes])


def _fake_commit(monkeypatch, commits):
    import deploy.git_utils as G
    monkeypatch.setattr(G, "git_commit_and_push", lambda p, m: commits.append((p, m)))


def test_one_failed_league_does_not_stop_the_rest_and_the_job_fails(monkeypatch):
    ran, commits = [], []
    _fake_steps(monkeypatch, [("nba", True), ("nhl", False), ("mlb", True), ("audit", True)], ran)
    monkeypatch.setattr(J, "changed", lambda: ["data/raw/sportsdataverse/nba_2027.parquet"])
    _fake_commit(monkeypatch, commits)
    alerts = []
    import deploy.notify as N
    monkeypatch.setattr(N, "send_webhook_alert", lambda m: alerts.append(m))
    assert J.run(date(2026, 11, 3)) == 1
    assert ran == ["nba", "nhl", "mlb", "audit"]
    assert len(commits) == 1 and "FAILED nhl" in commits[0][1]
    assert list(commits[0][0]) == list(J.PATHS)
    assert alerts and "nhl endpoint down" in alerts[0]


def test_an_unchanged_day_commits_nothing(monkeypatch):
    ran, commits = [], []
    _fake_steps(monkeypatch, [("nba", True), ("audit", True)], ran)
    monkeypatch.setattr(J, "changed", lambda: [])
    _fake_commit(monkeypatch, commits)
    assert J.run(date(2026, 11, 3)) == 0
    assert commits == []


def test_the_season_is_named_by_the_year_it_ends():
    assert J.season_ending(date(2026, 9, 29)) == 2027
    assert J.season_ending(date(2027, 4, 10)) == 2027
    assert J.season_ending(date(2027, 7, 1)) == 2027


def test_every_path_it_commits_is_a_data_path():
    """Staged by name. A job that could stage source code is one bad
    checkout away from pushing it."""
    for p in J.PATHS:
        assert p.startswith("data/raw/") or p == "model/data_integrity.json", p


def test_the_blueprint_schedules_it_after_the_mlb_results_job():
    services = {s["name"]: s for s in yaml.safe_load((ROOT / "render.yaml").read_text())["services"]}
    job = services["live-inputs-job"]
    assert job["startCommand"] == "python3 deploy/live_inputs_job.py"
    minute, hours = job["schedule"].split()[:2]
    daily = services["mlb-daily-job"]["schedule"].split()
    assert min(int(h) for h in hours.split(",")) > int(daily[1])
    assert minute != "0", "the :00 slot is taken by the odds jobs"
    keys = {e["key"] for e in job["envVars"]}
    assert "ODDS_API_KEY" not in keys, "this job must never be able to spend credits"
    assert {"GIT_REPO_URL", "GITHUB_TOKEN"} <= keys


def test_git_utils_stages_every_path_it_is_given(monkeypatch):
    import deploy.git_utils as G
    calls = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(argv, **kw):
        calls.append(argv)
        return R()
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    monkeypatch.setattr(G, "validate_git_push_succeeded", lambda *a: None)
    G.git_commit_and_push(["data/raw/nhl", "model/data_integrity.json"], "m")
    adds = [c for c in calls if c[:2] == ["git", "add"]]
    assert adds == [["git", "add", "--", "data/raw/nhl", "model/data_integrity.json"]]
    G.git_commit_and_push("data/bronze", "m")          # the old single-path call
    assert ["git", "add", "--", "data/bronze"] in calls


def test_an_off_day_writes_no_slate(monkeypatch, tmp_path):
    from model.ingest import mlb_slate
    monkeypatch.setattr(mlb_slate, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mlb_slate, "load_id_bridge", lambda: {})

    class Resp:
        def json(self):
            return {"dates": []}
    monkeypatch.setattr("requests.get", lambda *a, **k: Resp())
    assert mlb_slate.main(["--date", "2026-12-25"]) == 0
    assert list(tmp_path.iterdir()) == []
