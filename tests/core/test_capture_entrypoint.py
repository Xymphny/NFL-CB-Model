"""The capture entrypoint, and the free endpoint that makes it affordable.

WHY THE EVENTS ENDPOINT MATTERS
A close-capture job has to know when games start before it can decide when to
poll. Paying for that with an odds request would double the cost of every
window. The vendor's guide says of `/v4/sports/{sport}/events`: "This endpoint
does not count against the usage quota" -- and `odds_client.py` said for months
that `/sports` was "the one endpoint that costs nothing", which was wrong and
expensive to believe.

WHAT THE CRON WOULD COST
1,290 credits a month with every league in season, against a 100,000-credit
quota. Capture is cheap; the backfill at 89,760 is the thing that eats a month.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from coverline.execution.capture import estimate_monthly  # noqa: E402
from coverline.execution.odds_client import (  # noqa: E402
    CreditLedger, OddsAPIClient,
)


class _Transport:
    """Returns an events payload and records what was asked for."""

    def __init__(self, starts):
        self.urls: list[str] = []
        self._starts = starts

    def get(self, url: str):
        self.urls.append(url)
        body = json.dumps([
            {"id": f"e{i}", "commence_time": t} for i, t in enumerate(self._starts)
        ]).encode()
        return 200, body, {"x-requests-remaining": "480"}


def _client(starts, budget=50):
    t = _Transport(starts)
    ledger = CreditLedger(budget=budget)
    return OddsAPIClient(api_key="k", ledger=ledger, transport=t), ledger, t


def test_the_events_endpoint_costs_nothing():
    """Asserted with a budget of ZERO, which is the only proof that counts."""
    client, ledger, t = _client(["2026-01-02T00:00:00Z"], budget=0)
    resp = client.events(sport="icehockey_nhl", captured_at="2026-01-01T00:00:00Z")
    assert resp.cost_predicted == 0
    assert resp.cost_actual == 0
    assert ledger.spent_predicted == 0
    assert ledger.remaining_reported == 480
    assert len(t.urls) == 1
    assert "/events" in t.urls[0]


def test_the_api_key_never_appears_in_a_stored_url():
    """The url is kept for provenance, so it must not carry the secret."""
    client, _, _ = _client(["2026-01-02T00:00:00Z"])
    resp = client.events(sport="icehockey_nhl", captured_at="2026-01-01T00:00:00Z")
    assert "REDACTED" in resp.url
    assert "k" not in resp.url.split("apiKey=")[-1].split("&")[0].replace(
        "REDACTED", "")


def test_a_slate_collapses_into_the_fewest_polls(monkeypatch):
    """The cost argument, end to end through the entrypoint's planner.

    Twelve games at three distinct tip times must cost three polls, not
    twelve, or the monthly estimate is fiction.
    """
    import capture as entry

    starts = ([f"2026-01-02T00:0{i}:00Z" for i in range(4)]
              + [f"2026-01-02T01:0{i}:00Z" for i in range(4)]
              + [f"2026-01-02T02:0{i}:00Z" for i in range(4)])
    client, _, _ = _client(starts)
    monkeypatch.setattr(entry, "SPORTS", {"basketball_nba": ("eu",)})
    windows = entry.plan(client, horizon_hours=12, now="2026-01-01T23:00:00Z",
                         lead_minutes=5, cluster_minutes=20, cache_minutes=0)
    assert len(windows) == 3, [w.at for w in windows]
    assert all(w.cost == 3 for w in windows)


def test_the_plan_is_cached_so_a_free_endpoint_is_not_hammered(tmp_path,
                                                               monkeypatch):
    """Free to us is not free to serve.

    At this cadence an uncached job issues five requests a run, about 480 a
    day. A slate does not change between two polls fifteen minutes apart.
    """
    import capture as entry

    monkeypatch.setattr(entry, "PLAN_CACHE", tmp_path / "plan.json")
    monkeypatch.setattr(entry, "SPORTS", {"basketball_nba": ("eu",)})
    now = "2026-01-01T23:00:00Z"
    client, _, t = _client(["2026-01-02T00:00:00Z"])

    first = entry.plan(client, horizon_hours=12, now=now, lead_minutes=5,
                       cluster_minutes=20, cache_minutes=60)
    assert len(t.urls) == 1

    later = "2026-01-01T23:15:00Z"
    second = entry.plan(client, horizon_hours=12, now=later, lead_minutes=5,
                        cluster_minutes=20, cache_minutes=60)
    assert len(t.urls) == 1, "the cache was not used"
    assert [w.at for w in second] == [w.at for w in first]

    stale = "2026-01-02T01:00:00Z"
    entry.plan(client, horizon_hours=12, now=stale, lead_minutes=5,
               cluster_minutes=20, cache_minutes=60)
    assert len(t.urls) == 2, "a stale cache was reused"


def test_a_corrupt_cache_is_ignored_rather_than_crashing(tmp_path, monkeypatch):
    """A cron that dies on a truncated file stops capturing silently."""
    import capture as entry

    cache = tmp_path / "plan.json"
    cache.write_text("{not json")
    monkeypatch.setattr(entry, "PLAN_CACHE", cache)
    assert entry._cached_plan("2026-01-01T23:00:00Z", 60) is None


def test_the_monthly_estimate_is_a_small_fraction_of_the_quota():
    """The number that decides whether this can run all season.

    If capture ever approaches the backfill's 89,760 the design is wrong and
    the cadence has to change rather than the budget.
    """
    m = estimate_monthly(
        {"americanfootball_nfl": 6, "americanfootball_ncaaf": 8,
         "basketball_nba": 30, "icehockey_nhl": 20, "baseball_mlb": 35},
        markets=3, regions=1)
    assert m["TOTAL"] < 5_000, f"capture now costs {m['TOTAL']:,} a month"
    assert m["TOTAL"] > 500, "the estimate looks too small to be real"


def test_the_entrypoint_is_dry_without_run(monkeypatch, capsys):
    """Same contract as scripts/backfill.py: nothing spends without --run."""
    import capture as entry

    monkeypatch.setenv("CAPTURE_ENABLED", "1")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(entry, "SPORTS", {"basketball_nba": ("eu",)})
    monkeypatch.setattr(entry, "OddsAPIClient",
                        lambda **kw: _client(["2026-01-02T00:00:00Z"])[0])
    monkeypatch.setattr(entry, "plan", lambda *a, **k: [])
    assert entry.main([]) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "no credits were spent" in out


# ---------------------------------------------------------------------------
# The gate and the persistence path.
# ---------------------------------------------------------------------------

def test_the_job_cannot_spend_until_it_is_enabled(monkeypatch, capsys):
    """The whole reason the schedule can exist before the subscription does.

    An unset or non-"1" CAPTURE_ENABLED must exit BEFORE an API client is
    constructed -- not merely before a paid call, because even the free
    events endpoint is somebody else's bandwidth and the key may not exist
    yet.
    """
    import capture as entry

    monkeypatch.delenv("CAPTURE_ENABLED", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "k")

    def _boom(**kw):
        raise AssertionError("an API client was constructed while disabled")

    monkeypatch.setattr(entry, "OddsAPIClient", _boom)
    assert entry.main([]) == 0
    assert "costs nothing" in capsys.readouterr().out

    for value in ("0", "true", "yes", ""):
        monkeypatch.setenv("CAPTURE_ENABLED", value)
        assert entry.main([]) == 0, f"{value!r} enabled the job"


def test_nothing_captured_means_nothing_committed(monkeypatch, capsys):
    """Most runs have no window due.

    A push per run would be 96 a day against a branch three other crons and a
    human already share. Only a run that captured or gapped something writes.
    """
    import capture as entry

    monkeypatch.setenv("CAPTURE_ENABLED", "1")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(entry, "SPORTS", {"basketball_nba": ("eu",)})
    monkeypatch.setattr(entry, "OddsAPIClient",
                        lambda **kw: _client(["2026-01-02T00:00:00Z"])[0])
    monkeypatch.setattr(entry, "plan", lambda *a, **k: [])

    def _boom(*a, **k):
        raise AssertionError("committed with nothing to commit")

    monkeypatch.setitem(sys.modules, "deploy.git_utils",
                        type(sys)("deploy.git_utils"))
    sys.modules["deploy.git_utils"].git_commit_and_push = _boom

    assert entry.main(["--run", "--persist"]) == 0
    assert "no commit" in capsys.readouterr().out


def test_a_run_that_only_meets_known_gaps_does_not_commit(monkeypatch, capsys, tmp_path):
    """Live 2026-09-25: two missed MLB windows were re-gapped and pushed to
    main every fifteen minutes. A gap already on record is not news."""
    import capture as entry
    from coverline.execution.bronze import BronzeStore
    from coverline.execution.capture import CaptureWindow

    at = "2020-01-01T00:00:00Z"                       # long overdue
    BronzeStore(tmp_path).record_gap(sport="basketball_nba", intended_at=at,
                                     reason="window_missed")
    monkeypatch.setenv("CAPTURE_ENABLED", "1")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(entry, "BRONZE_ROOT", tmp_path)
    monkeypatch.setattr(entry, "OddsAPIClient", lambda **kw: object())
    monkeypatch.setattr(entry, "plan", lambda *a, **k: [
        CaptureWindow(sport="basketball_nba", at=at)])

    def _boom(*a, **k):
        raise AssertionError("committed a gap that was already on record")

    monkeypatch.setitem(sys.modules, "deploy.git_utils",
                        type(sys)("deploy.git_utils"))
    sys.modules["deploy.git_utils"].git_commit_and_push = _boom

    assert entry.main(["--run", "--persist"]) == 0
    out = capsys.readouterr().out
    assert "already on record" in out and "no commit" in out
    assert len(BronzeStore(tmp_path).gaps()) == 1


def test_the_store_lives_in_the_checkout():
    """On Render nothing else survives a cron run.

    The filesystem is discarded when the process exits, so a snapshot written
    outside the checkout is a snapshot that does not exist -- which would be a
    CLV series with silent holes, the failure capture.py exists to prevent.
    """
    import capture as entry

    assert entry.BRONZE_ROOT == ROOT / "data" / "bronze"
    assert entry.BRONZE_ROOT.is_relative_to(ROOT)


def test_the_blueprint_gate_is_the_decision_that_was_made():
    """Capture was held OFF until the Odds API subscription existed, because a
    schedule that starts spending the moment a Blueprint syncs is a decision
    nobody made. The owner made that decision on 2026-09-24 and turned it on.

    So this pins the gate to "1" now. Turning capture off again -- or back on
    after that -- is equally a decision, and should fail here until someone
    edits this test on purpose. Anything other than exactly "0" or "1" is a
    typo the job would treat as OFF, silently.
    """
    import yaml

    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text())
    job = next(s for s in blueprint["services"]
               if s["name"] == "close-capture-job")
    gate = next(e for e in job["envVars"] if e["key"] == "CAPTURE_ENABLED")
    assert gate["value"] in ("0", "1"), (
        f"CAPTURE_ENABLED is {gate['value']!r}; the job only honours exactly "
        "\"1\", so anything else is off without saying so"
    )
    assert gate["value"] == "1", (
        "capture is switched off in the Blueprint. It was turned on "
        "2026-09-24 with the Odds API subscription; if turning it off is "
        "deliberate, change this assertion in the same commit"
    )
    assert "--persist" in job["startCommand"], (
        "without --persist every captured close is discarded when the run ends"
    )
    for secret in ("ODDS_API_KEY", "GIT_REPO_URL", "GITHUB_TOKEN"):
        e = next(x for x in job["envVars"] if x["key"] == secret)
        assert e.get("sync") is False and "value" not in e, (
            f"{secret} must not have a literal value in the Blueprint"
        )
