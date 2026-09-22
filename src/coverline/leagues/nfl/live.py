"""Live NFL features: this week's ratings, not a historical cache.

WHAT MAKES THIS "LIVE" AND WHY IT IS STILL BRANCH-BY-ABSTRACTION
CachedWalkForwardSource serves finished seasons out of a committed artifact.
This one serves the current week from the ratings snapshot the weekly job
publishes (data/ratings/{season}-week-NN.json) joined to the nflverse
schedule, which is where rest days and neutral-site status come from.

It still does not reimplement the rating pipeline. rating_diff is
home total_rating minus away total_rating, read from a snapshot the legacy
job produced -- the same arithmetic model/prediction.predict_game does. The
pipeline that computes those ratings stays upstream until there is a reason
to move it, and parity against the live board is what would justify moving it.

THE POINT-IN-TIME CONTRACT IS REAL HERE, NOT DECORATIVE
A ratings snapshot is stamped with the week it was computed for. Serving week
3 features from a week 5 snapshot would be lookahead of the most ordinary
kind, and it is easy to do by accident because the newest file is always the
most convenient one. `for_week()` selects by week rather than by mtime, and
`features()` refuses a snapshot whose week is later than the game's.

WHAT IT DOES NOT PROVIDE, AND THEREFORE DECLARES FALSE
NGS features and Elo come from runtime fetches inside the weekly job, not
from committed artifacts. So this source reports ngs_present=False, which
selects MARGIN_COEFFICIENTS_V1_RATING_ONLY -- the vector the live board
itself used for the rating-only games on the 2026 week 2 slate. Claiming NGS
with zero-valued differences would silently select the full ensemble and
misprice every game, which is the failure that ran for weeks before the audit.

A full-ensemble live source needs the NGS feature fetch ported and has its own
ledger row. Until then this source is honest about covering the rating-only
path and nothing else.

DE-BIAS IS NOT APPLIED HERE
The board adds a slate de-bias offset measured across the week's games. That
is a property of the SLATE, not of a game, so it belongs to whatever assembles
a board rather than to a feature source. `reconstruct_board_margin()` exists
so the parity test can apply it explicitly and show its working.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coverline.leagues.nfl.model import GameFeatures, predict_margin

_ROOT = Path(__file__).resolve().parents[4]
RATINGS_DIR = _ROOT / "data" / "ratings"
SCHEDULE_URL = ("https://raw.githubusercontent.com/nflverse/nfldata/"
                "master/data/games.csv")


class NoRatingsSnapshot(FileNotFoundError):
    """No snapshot for that week. Refusing beats serving a later one."""


class LookaheadRefused(ValueError):
    """A snapshot computed after the game it is being asked about."""


def _parse_ts(ts: str) -> datetime:
    """ISO-8601, with or without a zone. Naive means UTC here."""
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def snapshot_week(path: Path) -> int:
    m = re.search(r"week-(\d+)", path.name)
    if not m:
        raise ValueError(f"cannot read a week number from {path.name}")
    return int(m.group(1))


def available_snapshots(season: int, ratings_dir: Path = RATINGS_DIR) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(Path(ratings_dir).glob(f"{season}-week-*.json")):
        out[snapshot_week(p)] = p
    return out


def history_versions(season: int, week: int,
                     ratings_dir: Path = RATINGS_DIR) -> list[Path]:
    """Every immutable version of one week's snapshot, oldest first.

    WHY THIS EXISTS. The canonical path is OVERWRITTEN. Across the committed
    history, 2026 week 1 was written three times in one morning and week 2
    twice, four days apart -- the second time after the week had finished, so
    the ratings at that name had seen the results they were being asked to
    predict. Pricing from the canonical file is therefore only point-in-time
    by luck.

    deploy/weekly_job.py now also writes data/ratings/history/, which is
    write-once, and the versions already lost to overwriting were recovered
    from git -- the exact bytes, not a reconstruction.
    """
    hist = Path(ratings_dir) / "history"
    if not hist.is_dir():
        return []
    return sorted(hist.glob(f"{season}-week-{week:02d}-*.json"))


def version_current_at(season: int, week: int, asof: str,
                       ratings_dir: Path = RATINGS_DIR) -> Path | None:
    """The newest version computed at or before `asof`, or None.

    None rather than a fallback: substituting a later version is the lookahead
    this whole module exists to refuse, and doing it quietly is worse than
    doing it loudly.
    """
    cut = _parse_ts(asof)
    best: tuple[datetime, Path] | None = None
    for p in history_versions(season, week, ratings_dir):
        try:
            ca = _parse_ts(json.loads(p.read_text()).get("computed_at", ""))
        except (ValueError, json.JSONDecodeError):
            continue
        if ca <= cut and (best is None or ca > best[0]):
            best = (ca, p)
    return best[1] if best else None


def load_schedule(seasons: list[int], url: str = SCHEDULE_URL) -> pd.DataFrame:
    """nflverse schedule. Carries home_rest/away_rest and location directly,
    so rest does not have to be recomputed from kickoff dates."""
    g = pd.read_csv(url)
    return g[g.season.isin(seasons)].copy()


@dataclass(frozen=True)
class RatingsSnapshotSource:
    """Implements nfl.model.FeatureSource for a single published week."""

    season: int
    week: int
    ratings: dict[str, dict]
    schedule: pd.DataFrame
    computed_at: str
    path: Path

    @classmethod
    def for_week(cls, season: int, week: int, schedule: pd.DataFrame,
                 ratings_dir: Path = RATINGS_DIR,
                 asof: str | None = None) -> "RatingsSnapshotSource":
        """Load the snapshot FOR a week, selected by week and not by mtime.

        Picking the newest file is the obvious shortcut and is lookahead: the
        newest snapshot has seen results the week in question had not.

        `asof` picks the IMMUTABLE version that was current at that instant,
        from data/ratings/history. Without it the canonical path is used, and
        that path is overwritten in place -- so it is point-in-time only by
        luck, and the guard below is what turns that luck into an error
        instead of a number.
        """
        snaps = available_snapshots(season, ratings_dir)
        if week not in snaps:
            raise NoRatingsSnapshot(
                f"no ratings snapshot for {season} week {week}; have weeks "
                f"{sorted(snaps)}. Refusing to substitute another week."
            )
        path = snaps[week]
        if asof is not None:
            versioned = version_current_at(season, week, asof, ratings_dir)
            if versioned is None:
                raise NoRatingsSnapshot(
                    f"no version of {season} week {week} was computed at or "
                    f"before {asof}; have "
                    f"{[p.name for p in history_versions(season, week, ratings_dir)]}. "
                    "Refusing to substitute a later one."
                )
            path = versioned
        doc = json.loads(path.read_text())
        if doc.get("week") != week or doc.get("season") != season:
            raise ValueError(
                f"{path.name} is named for {season} week {week} but contains "
                f"season {doc.get('season')} week {doc.get('week')}"
            )
        return cls(
            season=season, week=week,
            ratings={r["team"]: r for r in doc["ratings"]},
            schedule=schedule[(schedule.season == season) & (schedule.week == week)],
            computed_at=doc.get("computed_at", ""), path=path,
        )

    # -- FeatureSource ----------------------------------------------------

    def features(self, game_id: str, asof: str) -> GameFeatures:
        home, away = self.teams(game_id)
        row = self._schedule_row(home, away)

        for t in (home, away):
            if t not in self.ratings:
                raise KeyError(
                    f"{t} is not in {self.path.name}; refusing to price "
                    f"{game_id} rather than treating it as league-average"
                )

        self._refuse_lookahead(row, asof)

        rating_diff = (self.ratings[home]["total_rating"]
                       - self.ratings[away]["total_rating"])
        rest_diff = float(row.get("home_rest", 0) or 0) - float(row.get("away_rest", 0) or 0)
        neutral = str(row.get("location", "Home")).lower() == "neutral"

        return GameFeatures(
            rating_diff=float(rating_diff),
            rest_diff=rest_diff,
            is_neutral_site=neutral,
            ngs_present=False,   # see module docstring -- this is the truth
            elo_diff=0.0,
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def teams(game_id: str) -> tuple[str, str]:
        """'2026-W02-LA-NYG' -> ('LA', 'NYG')."""
        parts = game_id.split("-")
        if len(parts) < 4:
            raise ValueError(f"unparseable game id {game_id!r}")
        return parts[-2], parts[-1]

    def game_ids(self) -> list[str]:
        return [f"{self.season}-W{self.week:02d}-{r.home_team}-{r.away_team}"
                for r in self.schedule.itertuples()]

    def _schedule_row(self, home: str, away: str) -> dict:
        m = self.schedule[(self.schedule.home_team == home)
                          & (self.schedule.away_team == away)]
        if m.empty:
            raise KeyError(
                f"{home} vs {away} is not on the {self.season} week "
                f"{self.week} schedule"
            )
        return m.iloc[0].to_dict()

    def _refuse_lookahead(self, row: dict, asof: str) -> None:
        """The snapshot must not have been computed after the game started."""
        kickoff = row.get("gameday")
        if not kickoff or not self.computed_at:
            return
        try:
            ko = datetime.fromisoformat(str(kickoff)).replace(tzinfo=timezone.utc)
            ca = datetime.fromisoformat(self.computed_at.replace("Z", "+00:00"))
        except ValueError:
            return
        if ca.date() > ko.date():
            raise LookaheadRefused(
                f"{self.path.name} was computed {ca.date()}, after a game on "
                f"{ko.date()}. Those ratings have seen the result."
            )


def reconstruct_board_margin(features: GameFeatures, debias_offset: float) -> float:
    """The live board's model margin: the linear model plus the slate de-bias.

    Kept here rather than in the feature source because the offset is a
    property of the whole slate, measured as the median market-minus-model
    residual across the week's games. A per-game source cannot know it.
    """
    return predict_margin(features) + debias_offset
