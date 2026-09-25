"""The backfill, with its three protective properties actually exercised.

Cost-before-spend, resumability, and stop-on-budget are the reasons this
module exists. Each is tested by making it matter: a plan that is costed
without a network, a run that is interrupted and resumed, and a budget that
runs out mid-job.
"""

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.execution import backfill as B  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.odds_client import (  # noqa: E402
    CreditLedger, OddsAPIClient,
)

EVENTS = [{"id": "e1", "sport_key": "americanfootball_nfl",
           "home_team": "H", "away_team": "A",
           "bookmakers": [{"key": "pinnacle", "markets": [
               {"key": "h2h", "outcomes": [{"name": "H", "price": 1.9091},
                                           {"name": "A", "price": 1.9091}]}]}]}]


class FixtureTransport:
    """Charges what the real API would, parsed from the URL."""

    def __init__(self, fail_on: set[str] | None = None):
        self.urls: list[str] = []
        self.remaining = 1_000_000
        self.fail_on = fail_on or set()

    def get(self, url):
        self.urls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        date = q.get("date", [""])[0]
        if date in self.fail_on:
            return 500, b'{"message":"upstream boom"}', {}
        m = len(q.get("markets", [""])[0].split(",")) if q.get("markets") else 0
        r = len(q.get("regions", [""])[0].split(",")) if q.get("regions") else 0
        cost = m * r * (10 if "/historical/" in url else 1)
        self.remaining -= cost
        return 200, json.dumps(EVENTS).encode(), {
            "x-requests-last": str(cost),
            "x-requests-remaining": str(self.remaining),
        }


def _client(budget=100_000, transport=None):
    t = transport or FixtureTransport()
    return OddsAPIClient("K", CreditLedger(budget=budget), t), t


# ------------------------------------------------------------- planning ----

def test_a_plan_costs_itself_with_no_network():
    """The property that makes the spend decision informed rather than brave."""
    snaps = B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-12",
                              times_of_day=["17:00", "20:00"])
    p = B.plan(snaps)
    assert len(p) == 16              # 8 days x 2 windows
    assert p.total_cost == 16 * 30   # 10 x 3 markets x 1 region
    assert "16 snapshots" in p.describe()


def test_weekday_filtering_is_what_keeps_nfl_honest():
    """Without it, an NFL backfill pays for six empty days a week."""
    every = B.daily_snapshots(sport="nfl", start="2024-09-01", end="2024-09-28",
                              times_of_day=["17:00"])
    gamedays = B.daily_snapshots(sport="nfl", start="2024-09-01", end="2024-09-28",
                                 times_of_day=["17:00"], weekdays=[3, 6, 0])
    assert len(every) == 28
    assert len(gamedays) == 12
    assert B.plan(gamedays).total_cost < B.plan(every).total_cost / 2


def test_regions_and_markets_multiply_the_cost_as_documented():
    one = B.Snapshot("nfl", "2024-09-05T17:00:00Z", ("h2h",), ("eu",))
    many = B.Snapshot("nfl", "2024-09-05T17:00:00Z",
                      ("h2h", "spreads", "totals"), ("eu", "us"))
    assert one.cost == 10
    assert many.cost == 60


def test_duplicate_snapshots_collapse():
    a = B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-06",
                          times_of_day=["17:00"])
    p = B.plan(a, a, a)
    assert len(p) == 2


def test_an_inverted_date_range_is_refused():
    with pytest.raises(ValueError, match="precedes start"):
        B.daily_snapshots(sport="nfl", start="2024-09-10", end="2024-09-01",
                          times_of_day=["17:00"])


def test_the_plan_breaks_cost_down_by_sport():
    p = B.plan(
        B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-06",
                          times_of_day=["17:00"]),
        B.daily_snapshots(sport="nba", start="2024-09-05", end="2024-09-06",
                          times_of_day=["23:00", "01:00"]),
    )
    by = p.by_sport()
    assert by["nfl"] == 60 and by["nba"] == 120


# ---------------------------------------------------------------- runs ----

def test_a_run_fetches_and_stores_everything(tmp_path):
    store = BronzeStore(tmp_path)
    client, t = _client()
    p = B.plan(B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-07",
                                 times_of_day=["17:00"]))
    res = B.run(p, client, store)
    assert len(res.fetched) == 3
    assert res.credits_spent == 90
    assert len(t.urls) == 3
    assert all("/historical/" in u for u in t.urls)
    assert store.coverage("nfl") == {"snapshots": 3, "gaps": 0}


def test_resuming_refetches_only_what_is_missing(tmp_path):
    """A crash at 80% must cost the last 20%, not the whole job."""
    store = BronzeStore(tmp_path)
    p = B.plan(B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-09",
                                 times_of_day=["17:00"]))
    assert len(p) == 5

    # first pass: budget only covers three
    client, _ = _client(budget=90)
    first = B.run(B.remaining(p, store), client, store)
    assert len(first.fetched) == 3
    assert first.stopped_early is True

    # second pass with a fresh budget picks up exactly the two left
    client2, t2 = _client(budget=100_000)
    second = B.run(B.remaining(p, store), client2, store)
    assert len(second.fetched) == 2
    assert len(t2.urls) == 2, "resume refetched snapshots already in bronze"
    assert B.remaining(p, store).snapshots == ()


def test_running_out_of_budget_gaps_the_rest_rather_than_going_quiet(tmp_path):
    """Holes that are not recorded are indistinguishable from timestamps the
    API never had data for."""
    store = BronzeStore(tmp_path)
    p = B.plan(B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-09",
                                 times_of_day=["17:00"]))
    client, _ = _client(budget=60)
    res = B.run(p, client, store)

    assert len(res.fetched) == 2
    assert res.stopped_early is True
    assert len(res.gapped) == 3
    gaps = store.gaps("nfl")
    assert len(gaps) == 3
    assert {g["reason"] for g in gaps} == {"quota_exhausted"}
    assert "STOPPED EARLY" in res.summary()


def test_a_failed_fetch_is_gapped_and_the_run_continues(tmp_path):
    """One bad timestamp must not abandon the other 91,999 credits of work."""
    store = BronzeStore(tmp_path)
    bad = "2024-09-06T17:00:00Z"
    client, _ = _client(transport=FixtureTransport(fail_on={bad}))
    p = B.plan(B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-07",
                                 times_of_day=["17:00"]))
    res = B.run(p, client, store)

    assert len(res.fetched) == 2
    assert [k for k, _ in res.gapped] == [f"nfl|{bad}"]
    assert store.gaps("nfl")[0]["reason"] == "fetch_failed"
    assert res.stopped_early is False


def test_snapshots_are_written_as_historical_not_current(tmp_path):
    """remaining() keys off this. If the kind were wrong, every resume would
    refetch everything and the job would cost double."""
    store = BronzeStore(tmp_path)
    client, _ = _client()
    p = B.plan(B.daily_snapshots(sport="nfl", start="2024-09-05", end="2024-09-05",
                                 times_of_day=["17:00"]))
    B.run(p, client, store)
    rows = list(store.snapshots("nfl"))
    assert rows and all(r["kind"] == "historical" for r in rows)
    assert B.remaining(p, store).snapshots == ()


def test_an_already_present_snapshot_is_skipped_not_overwritten(tmp_path):
    store = BronzeStore(tmp_path)
    at = "2024-09-05T17:00:00Z"
    store.write_snapshot(sport="nfl", captured_at=at, payload=[{"pre": "existing"}],
                         cost=0, source_url="u", kind="historical")
    client, _ = _client()
    p = B.Plan(snapshots=(B.Snapshot("nfl", at, ("h2h", "spreads", "totals"), ("eu",)),))
    res = B.run(p, client, store)
    assert res.skipped_existing == [f"nfl|{at}"]
    stored = store.read_snapshot(store.snapshot_path("nfl", at, "historical"))
    assert stored["payload"] == [{"pre": "existing"}]


def test_the_realistic_five_league_plan_fits_one_month_of_the_100k_tier():
    """The arithmetic ADR 0002 rests on, recomputed from the planner itself
    rather than quoted."""
    p = B.plan(
        B.daily_snapshots(sport="americanfootball_nfl", start="2024-09-05",
                          end="2025-01-05", times_of_day=["17:00", "20:05", "20:25", "00:15"],
                          weekdays=[3, 5, 6, 0]),
        B.daily_snapshots(sport="basketball_nba", start="2024-10-22", end="2025-04-13",
                          times_of_day=["23:00", "23:30", "00:00", "02:00", "02:30"]),
        B.daily_snapshots(sport="icehockey_nhl", start="2024-10-04", end="2025-04-17",
                          times_of_day=["23:00", "23:30", "00:00", "02:00"]),
    )
    assert p.total_cost < 100_000, (
        f"the three-league plan costs {p.total_cost:,}, over a month of the "
        "100K tier; re-read ADR 0002 before running it"
    )
    assert p.total_cost > 30_000, "suspiciously cheap -- is the plan empty?"


# ------------------------------------------------------------------ cli ----

import subprocess  # noqa: E402

CLI = ROOT / "scripts" / "backfill.py"


def _cli(*args, env=None):
    import os
    e = dict(os.environ); e.pop("ODDS_API_KEY", None); e.update(env or {})
    return subprocess.run([sys.executable, str(CLI), *args],
                          capture_output=True, text=True, env=e, cwd=ROOT)


def test_the_cli_is_dry_by_default_and_spends_nothing():
    """A ~90,000-credit job must not be one typo from starting."""
    # data/bronze exists in the checkout once live capture has persisted
    # anything (it commits there), so "bronze does not exist" stopped being
    # the property. The property is that a dry run WRITES nothing.
    bronze = ROOT / "data" / "bronze"

    def snapshot():
        if not bronze.exists():
            return {}
        return {p: p.stat().st_mtime_ns for p in bronze.rglob("*") if p.is_file()}

    before = snapshot()
    r = _cli("--sports", "nfl")
    assert r.returncode == 0
    assert "Dry run" in r.stdout
    assert "no credits were spent" in r.stdout
    assert snapshot() == before, "a dry run wrote to bronze"


def test_the_cli_refuses_to_run_without_a_key():
    r = _cli("--sports", "nfl", "--run")
    assert r.returncode == 2
    assert "ODDS_API_KEY is not set" in r.stdout


def test_the_cli_warns_when_the_plan_crowds_out_live_capture():
    """90% of a month leaves nothing for the capture the subscription is
    mainly for. The planner should say so rather than let it be discovered."""
    r = _cli()
    assert "90%" in r.stdout or "8" in r.stdout
    assert "little headroom" in r.stdout
    assert "resumable" in r.stdout


def test_the_cli_points_at_the_historical_shakeout_first():
    src = CLI.read_text()
    assert "shakeout_odds_api.py --historical" in src, (
        "the 30-credit check that protects this 90,000-credit job should be "
        "named where someone about to run it will see it"
    )


def test_every_configured_league_builds_a_plan():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bf_cli", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in mod.LEAGUES:
        p = mod.build_plan(2024, [name], ["eu"])
        assert len(p) > 0, f"{name} produced an empty plan"
        assert p.total_cost > 0
