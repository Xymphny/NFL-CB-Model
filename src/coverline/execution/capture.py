"""Scheduled capture: poll near the close, and record every miss.

WHAT THIS IS FOR
CLV is measured against the closing price, so the capture that matters is the
last snapshot before a game starts. One request returns every game live for a
sport at that instant, so the cost is driven by how many distinct windows a
slate has, not by how many games are played -- which is what makes five
leagues affordable at all.

THE PROPERTY THAT MATTERS MOST IS NOT CAPTURING
It is knowing when capture did not happen. A CLV series with silent holes in
it is worse than a short one: the holes are invisible, they are not random --
outages cluster on busy slates -- and every summary computed over the series
inherits the bias without showing it. So every window that should have been
captured and was not gets a gap row with a machine-readable reason, and
`coverage()` reports captured against missed so a degrading job shows up as a
number rather than as a gradual absence.

BUDGET IS A FIRST-CLASS INPUT, NOT A BACKSTOP
The monthly quota is shared with the backfill and with line shopping. A
capture run takes a budget and stops inside it, gapping the rest. Running out
of credits mid-season is not an error condition to be handled later; it is a
planning input, and `estimate_monthly()` exists so it can be checked before a
season starts rather than discovered in February.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from coverline.execution.bronze import BronzeStore
from coverline.execution.odds_client import OddsAPIClient, QuotaExceeded

DEFAULT_MARKETS = ("h2h", "spreads", "totals")


def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _fmt(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CaptureWindow:
    """One intended poll: a sport at an instant, with the reason it exists."""

    sport: str
    at: str
    markets: tuple[str, ...] = DEFAULT_MARKETS
    regions: tuple[str, ...] = ("eu",)
    reason: str = "close"

    @property
    def cost(self) -> int:
        return OddsAPIClient.cost_current(list(self.markets), list(self.regions))

    @property
    def key(self) -> str:
        return f"{self.sport}|{self.at}"


def windows_for_slate(
    *,
    sport: str,
    commence_times: Iterable[str],
    lead_minutes: int = 5,
    cluster_minutes: int = 20,
    markets: Sequence[str] = DEFAULT_MARKETS,
    regions: Sequence[str] = ("eu",),
) -> list[CaptureWindow]:
    """Collapse a slate's kickoff times into the fewest polls that cover it.

    Games starting within `cluster_minutes` of each other share one poll, taken
    `lead_minutes` before the EARLIEST of them -- early rather than late,
    because a poll after kickoff has missed the close for that game and the
    price it returns is an in-play price, which is a different instrument.

    This clustering is the whole cost argument. An NBA night with twelve games
    at five distinct tip times costs five polls, not twelve.
    """
    times = sorted({_parse(t) for t in commence_times})
    if not times:
        return []

    out: list[CaptureWindow] = []
    cluster_start = times[0]
    for t in times:
        if t - cluster_start > timedelta(minutes=cluster_minutes):
            out.append(CaptureWindow(
                sport=sport, at=_fmt(cluster_start - timedelta(minutes=lead_minutes)),
                markets=tuple(markets), regions=tuple(regions)))
            cluster_start = t
    out.append(CaptureWindow(
        sport=sport, at=_fmt(cluster_start - timedelta(minutes=lead_minutes)),
        markets=tuple(markets), regions=tuple(regions)))
    return out


def due(windows: Sequence[CaptureWindow], now: str,
        tolerance_minutes: int = 10) -> list[CaptureWindow]:
    """Windows whose moment has arrived and has not yet passed out of reach.

    A window is due from its time until `tolerance_minutes` after. Past that
    it is LATE, not due: capturing it would store an in-play price under a
    pre-game timestamp, which is a quieter corruption than missing it. Late
    windows should be gapped, and `overdue()` finds them.
    """
    t = _parse(now)
    return [w for w in windows
            if _parse(w.at) <= t <= _parse(w.at) + timedelta(minutes=tolerance_minutes)]


def overdue(windows: Sequence[CaptureWindow], now: str,
            tolerance_minutes: int = 10) -> list[CaptureWindow]:
    """Windows that came and went uncaptured. These are gaps, not retries."""
    t = _parse(now)
    return [w for w in windows
            if _parse(w.at) + timedelta(minutes=tolerance_minutes) < t]


def estimate_monthly(windows_per_week: dict[str, int], *,
                     markets: int = 3, regions: int = 1) -> dict[str, int]:
    """Credits per month by sport, for planning before a season starts.

    Checking this in September is the difference between a budget decision and
    a February surprise.
    """
    per_poll = markets * regions
    out = {s: round(n * 52 / 12) * per_poll for s, n in windows_per_week.items()}
    out["TOTAL"] = sum(out.values())
    return out


@dataclass
class CaptureResult:
    captured: list[str] = field(default_factory=list)
    gapped: list[tuple[str, str]] = field(default_factory=list)
    already_had: list[str] = field(default_factory=list)
    #: Overdue windows whose gap was recorded on an earlier run. Not news, so
    #: they neither re-enter the gap log nor count toward a commit.
    already_gapped: list[str] = field(default_factory=list)
    credits_spent: int = 0

    def summary(self) -> str:
        return (f"{len(self.captured)} captured, {len(self.already_had)} already "
                f"present, {len(self.gapped)} gapped"
                + (f" ({len(self.already_gapped)} already on record)"
                   if self.already_gapped else "")
                + f", {self.credits_spent} credits")


def run(
    windows: Sequence[CaptureWindow],
    client: OddsAPIClient,
    store: BronzeStore,
    *,
    now: str,
    tolerance_minutes: int = 10,
) -> CaptureResult:
    """Capture what is due, gap what is overdue, gap what the budget refuses.

    Deliberately does NOT retry an overdue window. The price it would return
    is not the one the window was for.
    """
    res = CaptureResult()

    # A window is gapped ONCE. The slate plan is cached for an hour and games
    # stay on the events feed until they finish, so the same missed window
    # comes back as overdue on every fifteen-minute run. Recording it again
    # each time duplicated the gap log and pushed a commit to main for news
    # that was already written down (2026-09-25: the same two MLB windows,
    # four runs in a row).
    known = {(g["sport"], g["intended_at"]) for g in store.gaps()}

    for w in overdue(windows, now, tolerance_minutes):
        if (w.sport, w.at) in known:
            res.already_gapped.append(w.key)
            continue
        store.record_gap(sport=w.sport, intended_at=w.at, reason="window_missed",
                         detail=f"still uncaptured {tolerance_minutes}min after its "
                                f"time at {now}; an in-play price is not a close")
        res.gapped.append((w.key, "window_missed"))

    for w in due(windows, now, tolerance_minutes):
        if store.snapshot_path(w.sport, w.at).exists():
            res.already_had.append(w.key)
            continue
        try:
            resp = client.current_odds(sport=w.sport, markets=list(w.markets),
                                       regions=list(w.regions), captured_at=w.at)
        except QuotaExceeded as exc:
            store.record_gap(sport=w.sport, intended_at=w.at,
                             reason="quota_exhausted", detail=str(exc)[:200])
            res.gapped.append((w.key, "quota_exhausted"))
            continue
        except Exception as exc:
            store.record_gap(sport=w.sport, intended_at=w.at, reason="fetch_failed",
                             detail=f"{type(exc).__name__}: {exc}"[:400])
            res.gapped.append((w.key, "fetch_failed"))
            continue

        try:
            store.write_snapshot(sport=w.sport, captured_at=w.at,
                                 payload=resp.payload, cost=resp.cost_predicted,
                                 source_url=resp.url)
        except FileExistsError:
            res.already_had.append(w.key)
            continue

        res.captured.append(w.key)
        res.credits_spent += resp.cost_actual or resp.cost_predicted

    return res
