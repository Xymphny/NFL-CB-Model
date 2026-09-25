"""The ingestion layer, proven without a key.

Transport is injected, so every path here runs offline against recorded
fixtures shaped like real Odds API responses. That was a requirement rather
than a convenience: ADR 0002 says not to subscribe until there is something to
consume the data, so the client had to be provable before a dollar was spent.

The fixture prices are a real -110/-110 two-way market and a lopsided one, so
the normalizer's output can be handed straight to core.pricing and checked
end to end.
"""

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.execution.bronze import BronzeStore, SnapshotExists  # noqa: E402
from coverline.execution.normalize import (  # noqa: E402
    best_price, normalize, two_sided,
)
from coverline.execution.odds_client import (  # noqa: E402
    CostModelDrift, CreditLedger, OddsAPIClient, QuotaExceeded,
)

AT = "2026-09-21T18:00:00Z"

EVENTS = [
    {
        "id": "evt1", "sport_key": "americanfootball_nfl",
        "commence_time": "2026-09-21T20:00:00Z",
        "home_team": "Home HT", "away_team": "Away AT",
        "bookmakers": [
            {"key": "pinnacle", "last_update": AT, "markets": [
                {"key": "h2h", "last_update": AT, "outcomes": [
                    {"name": "Home HT", "price": 1.9091},
                    {"name": "Away AT", "price": 1.9091}]},
                {"key": "spreads", "last_update": AT, "outcomes": [
                    {"name": "Home HT", "price": 1.9091, "point": -3.0},
                    {"name": "Away AT", "price": 1.9091, "point": 3.0}]},
            ]},
            {"key": "draftkings", "last_update": AT, "markets": [
                {"key": "h2h", "last_update": AT, "outcomes": [
                    {"name": "Home HT", "price": 1.9524},
                    {"name": "Away AT", "price": 1.8696}]},
            ]},
        ],
    },
    {
        "id": "evt2", "sport_key": "americanfootball_nfl",
        "commence_time": "2026-09-21T23:00:00Z",
        "home_team": "Big Fav", "away_team": "Big Dog",
        "bookmakers": [
            {"key": "pinnacle", "last_update": AT, "markets": [
                {"key": "h2h", "last_update": AT, "outcomes": [
                    {"name": "Big Fav", "price": 1.3333},
                    {"name": "Big Dog", "price": 3.40}]},
            ]},
        ],
    },
]


class FixtureTransport:
    """Replays a canned response and records the URLs it was asked for.

    By default it CHARGES what the real API would -- markets x regions parsed
    back out of the URL, x10 for historical -- rather than a fixed number.
    That was not the first design: a constant header made two tests fail with
    CostModelDrift because the fixture claimed 6 credits on a 1-credit
    request. The detector was right and the fixture was lying, so the fixture
    was made faithful. A stub that cannot disagree with the code it stands in
    for tests nothing.
    """

    def __init__(self, status=200, payload=None, headers=None, body=None):
        self.status = status
        self.payload = EVENTS if payload is None else payload
        self.headers = headers          # None means "charge realistically"
        self.body = body
        self.urls: list[str] = []
        self.remaining = 100_000

    def _charge(self, url: str) -> int:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        markets = len(q.get("markets", [""])[0].split(",")) if q.get("markets") else 0
        regions = len(q.get("regions", [""])[0].split(",")) if q.get("regions") else 0
        if not markets or not regions:
            return 0
        cost = markets * regions
        return cost * 10 if "/historical/" in url else cost

    def get(self, url):
        self.urls.append(url)
        body = self.body if self.body is not None else json.dumps(self.payload).encode()
        if self.headers is not None:
            return self.status, body, self.headers
        cost = self._charge(url)
        self.remaining -= cost
        return self.status, body, {
            "x-requests-last": str(cost),
            "x-requests-remaining": str(self.remaining),
            "x-requests-used": str(100_000 - self.remaining),
        }


def _client(ledger=None, transport=None):
    return OddsAPIClient("KEY123", ledger or CreditLedger(budget=1000),
                         transport or FixtureTransport())


# ------------------------------------------------------------ cost model ----

def test_cost_formulas_match_the_published_ones():
    mk, rg = ["h2h", "spreads", "totals"], ["eu", "us"]
    assert OddsAPIClient.cost_current(mk, ["eu"]) == 3
    assert OddsAPIClient.cost_current(mk, rg) == 6
    assert OddsAPIClient.cost_historical(mk, rg) == 60
    assert OddsAPIClient.cost_event_props(4, ["us"]) == 4


def test_historical_is_exactly_ten_times_current():
    mk, rg = ["h2h", "totals"], ["eu"]
    assert (OddsAPIClient.cost_historical(mk, rg)
            == 10 * OddsAPIClient.cost_current(mk, rg))


def test_cost_requires_both_markets_and_regions():
    with pytest.raises(ValueError):
        OddsAPIClient.cost_current([], ["eu"])
    with pytest.raises(ValueError):
        OddsAPIClient.cost_current(["h2h"], [])


# --------------------------------------------------------------- budget ----

def test_a_request_over_budget_is_not_issued():
    """The guard must fire BEFORE the network call, or it has not saved
    anything."""
    t = FixtureTransport()
    c = _client(CreditLedger(budget=5), t)
    with pytest.raises(QuotaExceeded, match="Nothing was issued"):
        c.current_odds(sport="x", markets=["h2h", "spreads", "totals"],
                       regions=["eu", "us"], captured_at=AT)
    assert t.urls == [], "a request was issued despite exceeding the budget"


def test_budget_accumulates_across_calls():
    t = FixtureTransport(headers={"x-requests-last": "3", "x-requests-remaining": "10"})
    led = CreditLedger(budget=7)
    c = _client(led, t)
    c.current_odds(sport="a", markets=["h2h", "spreads", "totals"],
                   regions=["eu"], captured_at=AT)
    assert led.spent_predicted == 3
    c.current_odds(sport="b", markets=["h2h", "spreads", "totals"],
                   regions=["eu"], captured_at=AT)
    assert led.spent_predicted == 6
    with pytest.raises(QuotaExceeded):
        c.current_odds(sport="c", markets=["h2h", "spreads", "totals"],
                       regions=["eu"], captured_at=AT)
    assert len(t.urls) == 2


def test_the_api_key_never_appears_in_a_stored_url():
    """Snapshots record their source URL, and bronze is committed."""
    r = _client().current_odds(sport="x", markets=["h2h"], regions=["eu"],
                               captured_at=AT)
    assert "KEY123" not in r.url
    assert "REDACTED" in r.url


# ------------------------------------------------------- drift detection ----

def test_cost_model_drift_raises_rather_than_being_logged():
    """If predicted and charged disagree, every budget guard downstream is
    unreliable. Fail loudly."""
    t = FixtureTransport(headers={"x-requests-last": "9", "x-requests-remaining": "1"})
    c = _client(CreditLedger(budget=100), t)
    with pytest.raises(CostModelDrift, match="cost model is wrong"):
        c.current_odds(sport="x", markets=["h2h", "spreads", "totals"],
                       regions=["eu"], captured_at=AT)


def test_matching_cost_does_not_raise_and_is_reconciled():
    t = FixtureTransport(headers={"x-requests-last": "3", "x-requests-remaining": "77"})
    led = CreditLedger(budget=100)
    _client(led, t).current_odds(sport="x", markets=["h2h", "spreads", "totals"],
                                 regions=["eu"], captured_at=AT)
    assert led.spent_predicted == led.spent_actual == 3
    assert led.remaining_reported == 77


def test_a_tolerance_can_be_set_deliberately():
    t = FixtureTransport(headers={"x-requests-last": "4", "x-requests-remaining": "5"})
    led = CreditLedger(budget=100, drift_tolerance=1)
    _client(led, t).current_odds(sport="x", markets=["h2h", "spreads", "totals"],
                                 regions=["eu"], captured_at=AT)
    assert led.spent_actual == 4


def test_a_missing_quota_header_does_not_invent_one():
    t = FixtureTransport(headers={})
    led = CreditLedger(budget=100)
    _client(led, t).current_odds(sport="x", markets=["h2h"], regions=["eu"],
                                 captured_at=AT)
    assert led.spent_actual == 0
    assert led.remaining_reported is None
    assert led.history[-1]["actual"] is None


def test_an_http_error_surfaces_the_body_and_books_nothing():
    t = FixtureTransport(status=401, body=b'{"message":"invalid api key"}')
    led = CreditLedger(budget=100)
    with pytest.raises(RuntimeError, match="invalid api key"):
        _client(led, t).current_odds(sport="x", markets=["h2h"], regions=["eu"],
                                     captured_at=AT)
    assert led.spent_predicted == 0, "credits booked for a request that failed"


def test_historical_costs_ten_times_and_the_fixture_agrees():
    """Both sides of the cost model checked against each other: the client
    predicts 10x, and the fixture independently charges 10x from the URL."""
    t = FixtureTransport()
    led = CreditLedger(budget=500)
    _client(led, t).historical_odds(
        sport="americanfootball_nfl", markets=["h2h", "spreads", "totals"],
        regions=["eu"], date="2026-01-05T18:00:00Z", captured_at=AT,
    )
    assert led.spent_predicted == 30
    assert led.spent_actual == 30


def test_the_fixture_can_disagree_with_the_client():
    """Guard on the guard. If the fixture were rewritten to echo whatever the
    client predicted, every drift test above would pass vacuously."""
    t = FixtureTransport()
    url = "https://x/v4/sports/s/odds?markets=h2h,spreads&regions=eu,us"
    assert t._charge(url) == 4
    assert t._charge(url.replace("/sports/", "/historical/sports/")) == 40
    assert t._charge("https://x/v4/sports?apiKey=k") == 0


def test_the_free_sports_endpoint_costs_nothing():
    led = CreditLedger(budget=10)
    r = _client(led, FixtureTransport(payload=[{"key": "nfl"}])).sports(captured_at=AT)
    assert r.cost_predicted == 0
    assert led.spent_predicted == 0
    assert r.payload == [{"key": "nfl"}]


# -------------------------------------------------------------- bronze ----

def test_a_snapshot_round_trips(tmp_path):
    store = BronzeStore(tmp_path)
    s = store.write_snapshot(sport="nfl", captured_at=AT, payload=EVENTS,
                             cost=3, source_url="https://x?apiKey=REDACTED")
    assert s.n_events == 2
    rec = store.read_snapshot(s.path)
    assert rec["payload"] == EVENTS
    assert rec["_meta"]["cost_credits"] == 3


def test_bronze_refuses_to_overwrite(tmp_path):
    """Raw data is write-once: a close that was not captured cannot be
    regenerated, and two versions of one timestamp is a bug, not a merge."""
    store = BronzeStore(tmp_path)
    store.write_snapshot(sport="nfl", captured_at=AT, payload=EVENTS, cost=3,
                         source_url="u")
    with pytest.raises(SnapshotExists, match="write-once"):
        store.write_snapshot(sport="nfl", captured_at=AT, payload=[], cost=3,
                             source_url="u")


def test_a_gap_is_recorded_with_a_reason(tmp_path):
    store = BronzeStore(tmp_path)
    store.record_gap(sport="nfl", intended_at=AT, reason="quota_exhausted",
                     detail="budget of 5 credits reached")
    gaps = store.gaps("nfl")
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "quota_exhausted"


def test_a_gap_must_have_a_reason(tmp_path):
    with pytest.raises(ValueError, match="must have a reason"):
        BronzeStore(tmp_path).record_gap(sport="nfl", intended_at=AT, reason="")


def test_coverage_makes_a_degrading_capture_visible(tmp_path):
    """A rising gap count against a flat snapshot count is the shape of a job
    that has quietly stopped working."""
    store = BronzeStore(tmp_path)
    store.write_snapshot(sport="nfl", captured_at=AT, payload=EVENTS, cost=3,
                         source_url="u")
    for i in range(3):
        store.record_gap(sport="nfl", intended_at=f"2026-09-2{i}T19:00:00Z",
                         reason="http_error")           # windows never taken
    assert store.coverage("nfl") == {"snapshots": 1, "gaps": 3}


def test_nothing_in_the_store_can_interpolate(tmp_path):
    """There is deliberately no API for filling a gap. If one ever appears,
    this test should be the thing that objects."""
    store = BronzeStore(tmp_path)
    for name in dir(store):
        assert "interpolate" not in name and "fill" not in name, name


# ----------------------------------------------------------- normalize ----

def test_normalize_flattens_to_one_row_per_outcome():
    quotes = normalize(EVENTS, captured_at=AT)
    # evt1: pinnacle h2h 2 + pinnacle spreads 2 + dk h2h 2 = 6; evt2: 2
    assert len(quotes) == 8
    assert {q.bookmaker for q in quotes} == {"pinnacle", "draftkings"}
    spread = next(q for q in quotes if q.market == "spreads" and q.outcome == "Home HT")
    assert spread.point == pytest.approx(-3.0)


def test_normalize_drops_unusable_prices_rather_than_defaulting():
    bad = [{"id": "e", "bookmakers": [{"key": "b", "markets": [
        {"key": "h2h", "outcomes": [
            {"name": "A", "price": None}, {"name": "B", "price": 1.0},
            {"name": "C", "price": "nonsense"}, {"name": "D", "price": 2.5}]}]}]}]
    quotes = normalize(bad, captured_at=AT)
    assert [q.outcome for q in quotes] == ["D"]


def test_normalize_survives_empty_and_malformed_payloads():
    assert normalize([], captured_at=AT) == []
    assert normalize(None, captured_at=AT) == []
    assert normalize([{"no_id": True}], captured_at=AT) == []


def test_two_sided_returns_nothing_rather_than_a_partial_market():
    """Handing devig one side is the commonest way to get CLV wrong."""
    one_side = [{"id": "e", "bookmakers": [{"key": "b", "markets": [
        {"key": "h2h", "outcomes": [{"name": "A", "price": 1.9}]}]}]}]
    quotes = normalize(one_side, captured_at=AT)
    assert len(quotes) == 1
    assert two_sided(quotes, event_id="e", market="h2h", bookmaker="b") == []


def test_best_price_picks_the_maximum_across_books():
    quotes = normalize(EVENTS, captured_at=AT)
    best = best_price(quotes, event_id="evt1", market="h2h", outcome="Home HT")
    assert best is not None
    assert best.bookmaker == "draftkings"
    assert best.price_decimal == pytest.approx(1.9524)


# ---------------------------------------------------------- end to end ----

def test_a_captured_snapshot_devigs_to_a_fair_price(tmp_path):
    """The whole point of the layer: raw response -> stored -> re-read ->
    normalized -> devigged fair probability, with no key and no network."""
    store = BronzeStore(tmp_path)
    resp = _client().current_odds(sport="americanfootball_nfl",
                                  markets=["h2h", "spreads"], regions=["eu"],
                                  captured_at=AT)
    s = store.write_snapshot(sport="nfl", captured_at=AT, payload=resp.payload,
                             cost=resp.cost_predicted, source_url=resp.url)

    quotes = normalize(store.read_snapshot(s.path)["payload"], captured_at=AT)
    sides = two_sided(quotes, event_id="evt1", market="h2h", bookmaker="pinnacle")
    assert len(sides) == 2

    fair = P.devig([q.price_decimal for q in sides], "power")
    assert fair.sum() == pytest.approx(1.0)
    assert fair[0] == pytest.approx(0.5, abs=1e-3)  # a -110/-110 market


def test_the_lopsided_fixture_reproduces_the_known_devig_spread(tmp_path):
    """evt2 is the -300/+240 market the pricing tests pin, arriving through
    the ingestion path instead of as literals."""
    quotes = normalize(EVENTS, captured_at=AT)
    sides = two_sided(quotes, event_id="evt2", market="h2h", bookmaker="pinnacle")
    fav = next(q for q in sides if q.outcome == "Big Fav")
    idx = sides.index(fav)
    cmp = P.devig_all([q.price_decimal for q in sides], index=idx)
    assert cmp.fair_by_method["power"] == pytest.approx(0.7331, abs=1e-3)
    assert cmp.fair_by_method["multiplicative"] == pytest.approx(0.7183, abs=1e-3)
    assert cmp.spread == pytest.approx(0.0147, abs=1e-3)
