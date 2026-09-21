"""Historical backfill: cost it before spending, and make it resumable.

THE JOB THIS PROTECTS
Roughly 92,000 credits buys one season of closing snapshots across five
leagues -- most of a month on the 100K tier. It is one-shot, expensive, and
the single most expensive mistake available to this project is running it
wrong and paying twice.

Three properties follow, and they are the whole module:

1. COST IS COMPUTED BEFORE ANYTHING IS SPENT. `plan()` enumerates every
   snapshot the job will fetch and totals the credits, with no network access
   at all. You read the number, then decide.

2. THE JOB IS RESUMABLE. Every fetched snapshot is recorded in bronze's
   manifest, and `remaining()` subtracts what is already there. A crash at 80%
   costs the last 20%, not the whole thing. This is also what makes it safe to
   run the backfill in slices across several days rather than one long job.

3. IT CANNOT SILENTLY OVERSPEND. The ledger's budget is checked before each
   request, and when it refuses, the run stops and records a gap for every
   snapshot it did not take -- rather than continuing and leaving holes nobody
   can distinguish from data that never existed.

WHY CLOSING SNAPSHOTS AND NOT FULL RESOLUTION
The design research costed a 5-minute-resolution backfill at a scale that made
the $119 tier look necessary. It is not: CLV needs the CLOSE, and one
historical request returns every game for that sport at that timestamp. A few
snapshots per game-day per league is enough, which is why the whole thing fits
inside one month of the 100K tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from coverline.execution.bronze import BronzeStore
from coverline.execution.odds_client import OddsAPIClient, QuotaExceeded

DEFAULT_MARKETS = ("h2h", "spreads", "totals")


@dataclass(frozen=True)
class Snapshot:
    """One historical request: a sport at an instant."""

    sport: str
    at: str           # ISO-8601, the timestamp to ask the API for
    markets: tuple[str, ...]
    regions: tuple[str, ...]

    @property
    def cost(self) -> int:
        return OddsAPIClient.cost_historical(list(self.markets), list(self.regions))

    @property
    def key(self) -> str:
        """Stable identity, used to tell done from outstanding."""
        return f"{self.sport}|{self.at}"


@dataclass(frozen=True)
class Plan:
    snapshots: tuple[Snapshot, ...]

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def total_cost(self) -> int:
        return sum(s.cost for s in self.snapshots)

    def by_sport(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.snapshots:
            out[s.sport] = out.get(s.sport, 0) + s.cost
        return out

    def describe(self) -> str:
        lines = [f"{len(self)} snapshots, {self.total_cost:,} credits"]
        for sport, cost in sorted(self.by_sport().items(), key=lambda kv: -kv[1]):
            n = sum(1 for s in self.snapshots if s.sport == sport)
            lines.append(f"  {sport:28} {n:>5} snapshots  {cost:>8,} credits")
        return "\n".join(lines)


def daily_snapshots(
    *,
    sport: str,
    start: str,
    end: str,
    times_of_day: Sequence[str],
    markets: Sequence[str] = DEFAULT_MARKETS,
    regions: Sequence[str] = ("eu",),
    weekdays: Sequence[int] | None = None,
) -> list[Snapshot]:
    """Snapshots at fixed clock times across a date range.

    `times_of_day` are UTC "HH:MM" strings chosen to sit just before the
    kickoff windows that matter -- one request covers every game live at that
    instant, so the cost is driven by how many distinct windows a league has,
    not by how many games it plays.

    `weekdays` (0=Monday) restricts to the days a league actually plays, which
    is the difference between costing an NFL backfill honestly and paying for
    six empty days a week.
    """
    d0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    d1 = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    if d1 < d0:
        raise ValueError(f"end {end} precedes start {start}")

    out: list[Snapshot] = []
    day = d0
    while day <= d1:
        if weekdays is None or day.weekday() in weekdays:
            for hhmm in times_of_day:
                hh, mm = (int(x) for x in hhmm.split(":"))
                at = day.replace(hour=hh, minute=mm, second=0, microsecond=0)
                out.append(Snapshot(sport=sport,
                                    at=at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    markets=tuple(markets),
                                    regions=tuple(regions)))
        day += timedelta(days=1)
    return out


def plan(*groups: Iterable[Snapshot]) -> Plan:
    """Combine snapshot groups, dropping duplicates deterministically."""
    seen: dict[str, Snapshot] = {}
    for g in groups:
        for s in g:
            seen.setdefault(s.key, s)
    return Plan(snapshots=tuple(sorted(seen.values(), key=lambda s: (s.at, s.sport))))


@dataclass
class BackfillResult:
    fetched: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    gapped: list[tuple[str, str]] = field(default_factory=list)
    credits_spent: int = 0
    stopped_early: bool = False

    def summary(self) -> str:
        line = (f"{len(self.fetched)} fetched, "
                f"{len(self.skipped_existing)} already present, "
                f"{len(self.gapped)} gapped, {self.credits_spent:,} credits")
        if self.stopped_early:
            line += " -- STOPPED EARLY on budget"
        return line


def remaining(p: Plan, store: BronzeStore) -> Plan:
    """The part of the plan not already in bronze. This is what resumability
    means in practice: rerun the same plan and only the gaps get fetched."""
    have = {f"{r['sport']}|{r['captured_at']}"
            for r in store.snapshots() if r.get("kind") == "historical"}
    return Plan(snapshots=tuple(s for s in p.snapshots if s.key not in have))


def run(
    p: Plan,
    client: OddsAPIClient,
    store: BronzeStore,
    *,
    stop_on_budget: bool = True,
) -> BackfillResult:
    """Execute a plan. Call `remaining()` first unless you mean to refetch.

    On a budget refusal the run STOPS and records a gap for every remaining
    snapshot. Continuing would leave holes indistinguishable from timestamps
    the API never had data for, and the whole point of the gap log is that
    absence is legible.
    """
    res = BackfillResult()
    todo = list(p.snapshots)

    for i, snap in enumerate(todo):
        try:
            resp = client.historical_odds(
                sport=snap.sport, markets=list(snap.markets),
                regions=list(snap.regions), date=snap.at, captured_at=snap.at,
            )
        except QuotaExceeded as exc:
            res.stopped_early = True
            for rest in todo[i:]:
                store.record_gap(sport=rest.sport, intended_at=rest.at,
                                 reason="quota_exhausted", detail=str(exc)[:200])
                res.gapped.append((rest.key, "quota_exhausted"))
            if stop_on_budget:
                break
            continue
        except Exception as exc:
            store.record_gap(sport=snap.sport, intended_at=snap.at,
                             reason="fetch_failed", detail=f"{type(exc).__name__}: {exc}"[:400])
            res.gapped.append((snap.key, "fetch_failed"))
            continue

        try:
            store.write_snapshot(
                sport=snap.sport, captured_at=snap.at, payload=resp.payload,
                cost=resp.cost_predicted, source_url=resp.url, kind="historical",
            )
        except FileExistsError:
            res.skipped_existing.append(snap.key)
            continue

        res.fetched.append(snap.key)
        res.credits_spent += resp.cost_actual or resp.cost_predicted

    return res
