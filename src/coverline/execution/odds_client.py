"""The Odds API client, with credit accounting that refuses to overspend.

WHY THE ACCOUNTING IS THE INTERESTING PART
Credits are money: the 100K tier is $59/month (ADR 0002), and the published
cost formulas make it easy to spend a month's quota in an afternoon by
accident. Historical requests cost 10x current ones, and the props endpoint
charges PER EVENT rather than per sport -- so a loop over a night's NBA slate
costs roughly a thousand times a single main-market poll. A client that just
issues requests and hopes will find that out at the end of the month, when the
capture it was supposed to be doing has silently stopped.

So every call computes its cost BEFORE issuing, checks it against a budget,
and reconciles the estimate against the quota headers the API returns. That
last part matters most: a divergence between predicted and actual spend means
the cost model is wrong, and a wrong cost model is how a budget guard gives
false confidence. ``CreditLedger.reconcile`` raises on drift rather than
quietly trusting either number.

NO NETWORK IS REQUIRED TO TEST ANY OF THIS
Transport is injected. The real one is a thin urllib wrapper; the tests use a
recorded-fixture transport. This is deliberate: the client was built before the
subscription was bought (ADR 0002 says not to subscribe until there is
something to consume the data), so it had to be provable without a key.

WHAT THIS CLIENT WILL NOT DO
It will not interpolate. If a snapshot is missed -- outage, quota exhaustion,
a crash -- the gap is recorded as a gap by the bronze store. A market close
that was not captured is gone at any price, and inventing a plausible one
would corrupt the CLV series that the whole evaluation approach rests on.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

BASE = "https://api.the-odds-api.com/v4"

#: Cost multiplier for historical endpoints, per the published formulas.
HISTORICAL_MULTIPLIER = 10

#: Quota headers the API returns on every response.
H_REMAINING = "x-requests-remaining"
H_USED = "x-requests-used"
H_LAST = "x-requests-last"


class QuotaExceeded(RuntimeError):
    """The request would breach the configured budget. Nothing was issued."""


class CostModelDrift(RuntimeError):
    """Predicted cost disagreed with what the API actually charged.

    Raised rather than logged. A silently wrong cost model makes every budget
    guard downstream of it meaningless, and the failure is cheap to fix and
    expensive to ignore.
    """


class Transport(Protocol):
    """Anything that can turn a URL into (status, body, headers)."""

    def get(self, url: str) -> tuple[int, bytes, dict[str, str]]: ...


class UrllibTransport:
    """The real one. Kept trivial so there is nothing in it worth testing."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def get(self, url: str) -> tuple[int, bytes, dict[str, str]]:
        req = urllib.request.Request(url, headers={"User-Agent": "coverline/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, resp.read(), headers
        except urllib.error.HTTPError as exc:
            headers = {k.lower(): v for k, v in (exc.headers or {}).items()}
            return exc.code, exc.read(), headers


@dataclass
class CreditLedger:
    """Tracks spend against a budget, and against what the API says.

    ``budget`` is the number of credits this process is allowed to spend, which
    is normally well below the monthly quota -- the point is to stop one job
    from eating the month.
    """

    budget: int
    spent_predicted: int = 0
    spent_actual: int = 0
    calls: int = 0
    remaining_reported: int | None = None
    drift_tolerance: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def predicted_remaining(self) -> int:
        return self.budget - self.spent_predicted

    def check(self, cost: int) -> None:
        if cost <= 0:
            raise ValueError(f"a request must cost at least 1 credit, got {cost}")
        if self.spent_predicted + cost > self.budget:
            raise QuotaExceeded(
                f"request costs {cost} credits; {self.predicted_remaining} left "
                f"of a {self.budget}-credit budget. Nothing was issued -- raise "
                "the budget deliberately or narrow the request."
            )

    def record(self, *, predicted: int, headers: dict[str, str], label: str) -> None:
        """Book the spend and reconcile against the API's own accounting."""
        self.calls += 1
        self.spent_predicted += predicted

        last = headers.get(H_LAST)
        actual = int(float(last)) if last not in (None, "") else None
        if actual is not None:
            self.spent_actual += actual
            if abs(actual - predicted) > self.drift_tolerance:
                raise CostModelDrift(
                    f"{label}: predicted {predicted} credits, API charged "
                    f"{actual}. The cost model is wrong, so every budget guard "
                    "built on it is unreliable until this is understood."
                )

        rem = headers.get(H_REMAINING)
        if rem not in (None, ""):
            self.remaining_reported = int(float(rem))

        self.history.append(
            {"label": label, "predicted": predicted, "actual": actual,
             "remaining_reported": self.remaining_reported}
        )


@dataclass(frozen=True)
class OddsResponse:
    """A raw response plus everything needed to audit what it cost."""

    payload: Any
    cost_predicted: int
    cost_actual: int | None
    remaining: int | None
    url: str
    captured_at: str


class OddsAPIClient:
    """Thin, auditable wrapper. Does no parsing beyond JSON decoding.

    Normalisation lives in execution/normalize.py so that what lands in bronze
    is exactly what the API said, byte for byte.
    """

    def __init__(
        self,
        api_key: str,
        ledger: CreditLedger,
        transport: Transport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self._key = api_key
        self.ledger = ledger
        self.transport = transport or UrllibTransport()

    # -- cost model -------------------------------------------------------

    @staticmethod
    def cost_current(markets: Sequence[str], regions: Sequence[str]) -> int:
        """markets x regions, per the published formula."""
        if not markets or not regions:
            raise ValueError("both markets and regions are required")
        return len(markets) * len(regions)

    @staticmethod
    def cost_historical(markets: Sequence[str], regions: Sequence[str]) -> int:
        return HISTORICAL_MULTIPLIER * OddsAPIClient.cost_current(markets, regions)

    @staticmethod
    def cost_event_props(n_markets_returned: int, regions: Sequence[str]) -> int:
        """PER EVENT, and charged on markets RETURNED, not requested.

        Cannot be known exactly before the call, which is why props requests
        are budgeted with an upper bound rather than an estimate.
        """
        return max(1, n_markets_returned) * len(regions)

    # -- requests ---------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any], cost: int, label: str,
             captured_at: str) -> OddsResponse:
        self.ledger.check(cost)
        q = {k: v for k, v in params.items() if v is not None}
        q["apiKey"] = self._key
        url = f"{BASE}{path}?{urllib.parse.urlencode(q)}"

        status, body, headers = self.transport.get(url)
        if status != 200:
            # Do NOT book credits on a failure the API did not charge for, but
            # do surface the body: its message is usually the actual problem.
            raise RuntimeError(
                f"{label}: HTTP {status} -- {body[:300].decode('utf-8', 'replace')}"
            )

        self.ledger.record(predicted=cost, headers=headers, label=label)
        last = headers.get(H_LAST)
        return OddsResponse(
            payload=json.loads(body),
            cost_predicted=cost,
            cost_actual=int(float(last)) if last not in (None, "") else None,
            remaining=self.ledger.remaining_reported,
            url=url.replace(self._key, "REDACTED"),
            captured_at=captured_at,
        )

    def current_odds(
        self, *, sport: str, markets: Sequence[str], regions: Sequence[str],
        captured_at: str, odds_format: str = "decimal",
    ) -> OddsResponse:
        """One request returns every upcoming game for the sport.

        Cost tracks POLLS, not games, which is why closing capture across five
        leagues is affordable at all.
        """
        cost = self.cost_current(markets, regions)
        return self._get(
            f"/sports/{sport}/odds",
            {"markets": ",".join(markets), "regions": ",".join(regions),
             "oddsFormat": odds_format},
            cost, f"current:{sport}", captured_at,
        )

    def historical_odds(
        self, *, sport: str, markets: Sequence[str], regions: Sequence[str],
        date: str, captured_at: str, odds_format: str = "decimal",
    ) -> OddsResponse:
        """A snapshot as it stood at `date` (ISO-8601). Ten times the cost."""
        cost = self.cost_historical(markets, regions)
        return self._get(
            f"/historical/sports/{sport}/odds",
            {"markets": ",".join(markets), "regions": ",".join(regions),
             "date": date, "oddsFormat": odds_format},
            cost, f"historical:{sport}@{date}", captured_at,
        )

    def events(self, *, sport: str, captured_at: str) -> OddsResponse:
        """Upcoming games and their commence times. FREE.

        The vendor's own guide says of this endpoint: "This endpoint does not
        count against the usage quota." It is the second free one, and the
        docstring below said for months that /sports was "the one endpoint
        that costs nothing" -- which was wrong and expensive to believe.

        THIS IS WHAT MAKES A CAPTURE CRON AFFORDABLE. A close-capture job has
        to know when the games start before it can decide when to poll. Paying
        for that with an odds request would double the cost of every window;
        learning it here costs nothing, so credits are spent only on the
        snapshot that CLV is actually measured against.
        """
        q = urllib.parse.urlencode({"apiKey": self._key})
        url = f"{BASE}/sports/{sport}/events?{q}"
        status, body, headers = self.transport.get(url)
        if status != 200:
            raise RuntimeError(
                f"events {sport}: HTTP {status} -- "
                f"{body[:300].decode('utf-8', 'replace')}"
            )
        rem = headers.get(H_REMAINING)
        if rem not in (None, ""):
            self.ledger.remaining_reported = int(float(rem))
        return OddsResponse(
            payload=json.loads(body), cost_predicted=0, cost_actual=0,
            remaining=self.ledger.remaining_reported,
            url=url.replace(self._key, "REDACTED"), captured_at=captured_at,
        )

    def sports(self, captured_at: str) -> OddsResponse:
        """List available sports. Free, and not the only free one -- see events."""
        q = urllib.parse.urlencode({"apiKey": self._key})
        url = f"{BASE}/sports?{q}"
        status, body, headers = self.transport.get(url)
        if status != 200:
            raise RuntimeError(
                f"sports: HTTP {status} -- {body[:300].decode('utf-8', 'replace')}"
            )
        rem = headers.get(H_REMAINING)
        if rem not in (None, ""):
            self.ledger.remaining_reported = int(float(rem))
        return OddsResponse(
            payload=json.loads(body), cost_predicted=0, cost_actual=0,
            remaining=self.ledger.remaining_reported,
            url=url.replace(self._key, "REDACTED"), captured_at=captured_at,
        )
