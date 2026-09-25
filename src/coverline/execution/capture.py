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

#: The once-a-day EARLY capture, 14:00 UTC (10 AM Eastern in season). A paper
#: trade priced from the closing capture IS the close, so its CLV is zero by
#: construction and the Record's mean CLV measured nothing. One early poll per
#: sport per day returns every upcoming game, so each game gets one price well
#: before its close for ~450 credits a month -- and that is what CLV is
#: measured on. Early snapshots are stored under their own kind and are never
#: used as a close (grade.snapshots_for).
EARLY_UTC_HOUR = 14
#: An early price at 10:30 is as early as one at 10:00; a close is not. So an
#: early window stays due for hours, where a close window has ten minutes.
EARLY_TOLERANCE_MINUTES = 180


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
    def kind(self) -> str:
        """The snapshot kind it is stored under: "current" for a close, so
        every existing reader is unchanged, "early" for the daily early poll."""
        return "early" if self.reason == "early" else "current"

    @property
    def key(self) -> str:
        return f"{self.sport}|{self.at}" + ("|early" if self.reason == "early" else "")

    def tolerance(self, close_minutes: int) -> int:
        return EARLY_TOLERANCE_MINUTES if self.reason == "early" else close_minutes


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
            if _parse(w.at) <= t <= _parse(w.at) + timedelta(minutes=w.tolerance(tolerance_minutes))]


def overdue(windows: Sequence[CaptureWindow], now: str,
            tolerance_minutes: int = 10) -> list[CaptureWindow]:
    """Windows that came and went uncaptured. These are gaps, not retries."""
    t = _parse(now)
    return [w for w in windows
            if _parse(w.at) + timedelta(minutes=w.tolerance(tolerance_minutes)) < t]


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
    #: window key -> the snapshot it wrote. A close is filed under its window
    #: time; an early poll under the moment it was actually taken.
    paths: dict[str, str] = field(default_factory=dict)

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
    # An early window's gaps carry an "early_" reason, so an early and a close
    # window that happen to share an instant stay two different holes.
    known = {(g["sport"], g["intended_at"], g["reason"].startswith("early_"))
             for g in store.gaps()}

    def why(w: CaptureWindow, reason: str) -> str:
        return f"early_{reason}" if w.reason == "early" else reason

    for w in overdue(windows, now, tolerance_minutes):
        if (w.sport, w.at, w.reason == "early") in known:
            res.already_gapped.append(w.key)
            continue
        detail = (f"still uncaptured {w.tolerance(tolerance_minutes)}min after its time at "
                  f"{now}" + ("; the day's early price was missed" if w.reason == "early"
                              else "; an in-play price is not a close"))
        store.record_gap(sport=w.sport, intended_at=w.at, reason=why(w, "window_missed"),
                         detail=detail)
        res.gapped.append((w.key, why(w, "window_missed")))

    early_days = {(r["sport"], r["captured_at"][:10]) for r in store.snapshots()
                  if r.get("kind") == "early"}

    for w in due(windows, now, tolerance_minutes):
        # A close is filed under its window time (ten minutes of slack, and
        # the key readers look it up by). An early poll can run hours after
        # 14:00, so it is filed under the moment it is taken -- never labelled
        # earlier than the price it holds -- and one per sport per UTC day.
        at = now if w.reason == "early" else w.at
        if (w.reason == "early" and (w.sport, w.at[:10]) in early_days) or \
                (w.reason != "early" and store.snapshot_path(w.sport, w.at).exists()):
            res.already_had.append(w.key)
            continue
        try:
            resp = client.current_odds(sport=w.sport, markets=list(w.markets),
                                       regions=list(w.regions), captured_at=at)
        except QuotaExceeded as exc:
            store.record_gap(sport=w.sport, intended_at=w.at,
                             reason=why(w, "quota_exhausted"), detail=str(exc)[:200])
            res.gapped.append((w.key, why(w, "quota_exhausted")))
            continue
        except Exception as exc:
            store.record_gap(sport=w.sport, intended_at=w.at, reason=why(w, "fetch_failed"),
                             detail=f"{type(exc).__name__}: {exc}"[:400])
            res.gapped.append((w.key, why(w, "fetch_failed")))
            continue

        try:
            stored = store.write_snapshot(sport=w.sport, captured_at=at,
                                          payload=resp.payload, cost=resp.cost_predicted,
                                          source_url=resp.url, kind=w.kind)
        except FileExistsError:
            res.already_had.append(w.key)
            continue

        res.captured.append(w.key)
        res.paths[w.key] = str(stored.path)
        res.credits_spent += resp.cost_actual or resp.cost_predicted

    return res
