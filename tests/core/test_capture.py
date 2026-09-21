"""Scheduled capture, with the refusal behaviour tested as hard as the capture.

The easy half of this module is fetching. The half that protects the CLV
series is refusing: not capturing a window whose moment has passed, and
recording every miss with a reason. Both get more tests here than the happy
path, because a silent hole in a CLV series is invisible, non-random -- outages
cluster on busy slates -- and biases every summary computed over it.
"""

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.execution import capture as C  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.odds_client import CreditLedger, OddsAPIClient  # noqa: E402

EVENTS = [{"id": "e1", "home_team": "H", "away_team": "A",
           "bookmakers": [{"key": "pinnacle", "markets": [
               {"key": "h2h", "outcomes": [{"name": "H", "price": 1.91},
                                           {"name": "A", "price": 1.91}]}]}]}]


class FixtureTransport:
    def __init__(self, fail=False):
        self.urls = []
        self.remaining = 100_000
        self.fail = fail

    def get(self, url):
        self.urls.append(url)
        if self.fail:
            return 500, b'{"message":"boom"}', {}
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        m = len(q.get("markets", [""])[0].split(",")) if q.get("markets") else 0
        r = len(q.get("regions", [""])[0].split(",")) if q.get("regions") else 0
        cost = m * r
        self.remaining -= cost
        return 200, json.dumps(EVENTS).encode(), {
            "x-requests-last": str(cost),
            "x-requests-remaining": str(self.remaining)}


def _client(budget=10_000, fail=False):
    t = FixtureTransport(fail=fail)
    return OddsAPIClient("K", CreditLedger(budget=budget), t), t


# ------------------------------------------------------------ clustering ----

def test_games_close_together_share_one_poll():
    """The cost argument. Twelve games at five tip times must cost five polls."""
    tips = ["2026-09-21T23:00:00Z", "2026-09-21T23:10:00Z", "2026-09-21T23:15:00Z",
            "2026-09-22T02:00:00Z", "2026-09-22T02:05:00Z"]
    w = C.windows_for_slate(sport="nba", commence_times=tips)
    assert len(w) == 2


def test_the_poll_lands_before_the_earliest_game_in_its_cluster():
    """Late is worse than early: a poll after kickoff returns an in-play price,
    which is a different instrument from a close."""
    w = C.windows_for_slate(sport="nba",
                            commence_times=["2026-09-21T23:00:00Z",
                                            "2026-09-21T23:15:00Z"],
                            lead_minutes=5)
    assert len(w) == 1
    assert w[0].at == "2026-09-21T22:55:00Z"


def test_an_empty_slate_produces_no_windows():
    assert C.windows_for_slate(sport="nba", commence_times=[]) == []


def test_duplicate_kickoff_times_collapse():
    same = ["2026-09-21T23:00:00Z"] * 5
    assert len(C.windows_for_slate(sport="nba", commence_times=same)) == 1


def test_cluster_width_is_configurable_and_changes_the_bill():
    tips = ["2026-09-21T23:00:00Z", "2026-09-21T23:25:00Z"]
    tight = C.windows_for_slate(sport="nba", commence_times=tips, cluster_minutes=10)
    loose = C.windows_for_slate(sport="nba", commence_times=tips, cluster_minutes=40)
    assert len(tight) == 2 and len(loose) == 1


# ------------------------------------------------------------ due / late ----

def test_a_window_is_due_from_its_time_until_the_tolerance_expires():
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z")]
    assert C.due(w, "2026-09-21T22:59:00Z") == []          # not yet
    assert len(C.due(w, "2026-09-21T23:00:00Z")) == 1      # exactly on time
    assert len(C.due(w, "2026-09-21T23:09:00Z")) == 1      # inside tolerance
    assert C.due(w, "2026-09-21T23:11:00Z") == []          # expired


def test_an_expired_window_is_overdue_not_due():
    """The distinction the whole module turns on."""
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z")]
    late = "2026-09-21T23:30:00Z"
    assert C.due(w, late) == []
    assert len(C.overdue(w, late)) == 1


# ----------------------------------------------------------------- runs ----

def test_a_due_window_is_captured_and_stored(tmp_path):
    store = BronzeStore(tmp_path)
    client, t = _client()
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z")]
    res = C.run(w, client, store, now="2026-09-21T23:01:00Z")
    assert res.captured == ["nba|2026-09-21T23:00:00Z"]
    assert res.credits_spent == 3
    assert len(t.urls) == 1
    assert store.coverage("nba") == {"snapshots": 1, "gaps": 0}


def test_an_overdue_window_is_gapped_and_never_fetched(tmp_path):
    """The most important test here. Capturing it would file an in-play price
    under a pre-game timestamp -- a quieter corruption than missing it."""
    store = BronzeStore(tmp_path)
    client, t = _client()
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z")]
    res = C.run(w, client, store, now="2026-09-21T23:45:00Z")

    assert res.captured == []
    assert t.urls == [], "an overdue window was fetched anyway"
    assert res.gapped == [("nba|2026-09-21T23:00:00Z", "window_missed")]
    gap = store.gaps("nba")[0]
    assert gap["reason"] == "window_missed"
    assert "in-play price is not a close" in gap["detail"]


def test_a_future_window_is_neither_captured_nor_gapped(tmp_path):
    """It has not happened yet. Gapping it would manufacture a hole."""
    store = BronzeStore(tmp_path)
    client, t = _client()
    w = [C.CaptureWindow(sport="nba", at="2026-09-22T23:00:00Z")]
    res = C.run(w, client, store, now="2026-09-21T23:00:00Z")
    assert res.captured == [] and res.gapped == []
    assert t.urls == []
    assert store.gaps("nba") == []


def test_budget_exhaustion_gaps_rather_than_going_quiet(tmp_path):
    store = BronzeStore(tmp_path)
    client, _ = _client(budget=3)
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z"),
         C.CaptureWindow(sport="nhl", at="2026-09-21T23:02:00Z")]
    res = C.run(w, client, store, now="2026-09-21T23:03:00Z")

    assert len(res.captured) == 1
    assert [r for _, r in res.gapped] == ["quota_exhausted"]
    assert store.gaps("nhl")[0]["reason"] == "quota_exhausted"


def test_a_fetch_failure_is_gapped_and_the_run_continues(tmp_path):
    store = BronzeStore(tmp_path)
    client, _ = _client(fail=True)
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z"),
         C.CaptureWindow(sport="nhl", at="2026-09-21T23:01:00Z")]
    res = C.run(w, client, store, now="2026-09-21T23:02:00Z")
    assert len(res.gapped) == 2
    assert {r for _, r in res.gapped} == {"fetch_failed"}


def test_an_already_captured_window_is_not_refetched(tmp_path):
    """Re-running the job must be free, so it can be run on a tight cron
    without paying twice for the same close."""
    store = BronzeStore(tmp_path)
    w = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z")]
    c1, t1 = _client()
    C.run(w, c1, store, now="2026-09-21T23:01:00Z")
    c2, t2 = _client()
    res = C.run(w, c2, store, now="2026-09-21T23:02:00Z")
    assert res.already_had == ["nba|2026-09-21T23:00:00Z"]
    assert t2.urls == [], "a second run refetched a snapshot already in bronze"
    assert res.credits_spent == 0


def test_coverage_makes_a_degrading_job_countable(tmp_path):
    store = BronzeStore(tmp_path)
    client, _ = _client()
    ok = [C.CaptureWindow(sport="nba", at="2026-09-21T23:00:00Z")]
    C.run(ok, client, store, now="2026-09-21T23:01:00Z")
    missed = [C.CaptureWindow(sport="nba", at=f"2026-09-2{d}T23:00:00Z")
              for d in (2, 3, 4)]
    C.run(missed, client, store, now="2026-09-25T23:00:00Z")
    assert store.coverage("nba") == {"snapshots": 1, "gaps": 3}


# ------------------------------------------------------------ planning ----

def test_the_monthly_estimate_reproduces_the_figure_adr_0002_rests_on():
    """Recomputed here rather than quoted, so the ADR and the code cannot
    drift apart silently."""
    est = C.estimate_monthly({"nfl": 7, "ncaaf": 15, "nba": 35, "nhl": 30})
    assert est["TOTAL"] == 1131
    assert est["TOTAL"] < 20_000, "peak capture must fit the $30 tier with room"


def test_two_regions_double_the_capture_bill():
    one = C.estimate_monthly({"nba": 35}, regions=1)["TOTAL"]
    two = C.estimate_monthly({"nba": 35}, regions=2)["TOTAL"]
    assert two == 2 * one
