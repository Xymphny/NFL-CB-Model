"""Live CFB features: the weekly ratings snapshots, against ESPN's schedule.

WHY THIS EXISTS
CachedWalkForwardSource (sources.py) prices only games in the committed
2021-2023 walk-forward cache, so CFB could not be paper-traded and would never
reach the ledger floor that sizes a stake (ADR 0024). This source prices a
current game from the same two things the legacy CFB board already reads:

  ratings   data/cfb_ratings/{season}-week-NN.json, written weekly by
            deploy/cfb_weekly_job.py. Week N is rated on games BEFORE week N,
            so it is the snapshot that prices week N.
  schedule  data/raw/cfb/espn_{season}.parquet (model/ingest/cfb_espn.py),
            refreshed daily by deploy/live_inputs_job.py. It carries ESPN's
            event id, kickoff, neutral-site flag and completion flag, and its
            team names are the ratings' own spelling -- no name matching
            between the two.

WHICH MODEL VECTOR: DVOA-ONLY, ON PURPOSE
The snapshots carry an elo_rating, but it is the 2021-2025 schedule-cache
walk-forward, not updated by this season's games. The ensemble vector was fit
on in-season Elo; feeding it a preseason number would be a different model
from the one fit. And the market grade (model/grade_market_weight.py) and the
tier bands (model/derive_tier_thresholds.py) were both measured on the
DVOA-only vector, because the walk-forward cache has no Elo. So elo_present is
False here: what is paper-traded is what was graded.

WHAT IT REFUSES, EACH NAMED
  - a game not on the ESPN schedule, or already started
  - a neutral-site game: the fitted vector has no home term, only an
    intercept doing its job (model.py), so it would hand a home edge to a
    team that is not at home
  - a team the snapshot does not rate (FCS opponents, mostly)
  - ratings more than MAX_WEEKS_BEHIND behind the game's week: the weekly job
    stalling must stop the price, not quietly age it
  - any snapshot computed after kickoff (point in time)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coverline.execution.matching import ModelGame, local_date
from coverline.leagues.cfb.model import CFBModel, GameFeatures

_ROOT = Path(__file__).resolve().parents[4]
RATINGS_DIR = _ROOT / "data" / "cfb_ratings"
RAW = _ROOT / "data" / "raw" / "cfb"

#: Week W is priced from snapshot W. One week behind is tolerated -- a single
#: failed Sunday run -- and anything older is refused.
MAX_WEEKS_BEHIND = 1

_NAME = re.compile(r"^(\d{4})-week-(\d{2})\.json$")


class MissingSeasonData(FileNotFoundError):
    """No schedule or no ratings for the season. Carries the fixing command."""


class GameNotPriceable(KeyError):
    """This one game cannot be priced; the reason says why."""


class StaleRatings(GameNotPriceable):
    """The newest usable snapshot is too far behind the game's week."""


class LookaheadRefused(ValueError):
    pass


def _ts(asof) -> pd.Timestamp:
    if isinstance(asof, str) and asof == "now":
        return pd.Timestamp(datetime.now(timezone.utc))
    t = pd.Timestamp(asof)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def season_of(day: str) -> int:
    """CFB seasons are named for the autumn they start in; January bowls
    belong to the previous year's season."""
    d = datetime.fromisoformat(day[:10])
    return d.year if d.month >= 7 else d.year - 1


@dataclass(frozen=True)
class Snapshot:
    week: int
    computed_at: pd.Timestamp
    path: Path


@dataclass
class CFBLiveSource:
    schedule: pd.DataFrame          # ESPN, indexed by str(game_id)
    snapshots: list[Snapshot]       # oldest first
    season: int
    _ratings: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, season: int, raw: Path = RAW,
             ratings_dir: Path = RATINGS_DIR) -> "CFBLiveSource":
        sched_path = Path(raw) / f"espn_{season}.parquet"
        if not sched_path.exists():
            raise MissingSeasonData(
                f"{sched_path} is missing. Pull it with `python "
                f"model/ingest/cfb_espn.py --seasons {season}-{season} --force`.")
        s = pd.read_parquet(sched_path)
        if "start" not in s.columns:
            raise MissingSeasonData(
                f"{sched_path} predates the kickoff column. Re-pull it with "
                f"`python model/ingest/cfb_espn.py --seasons {season}-{season} --force`.")
        s = s.assign(game_id=s.game_id.astype(str),
                     start=pd.to_datetime(s.start, utc=True)).set_index("game_id")
        if s.index.duplicated().any():
            raise ValueError(f"{sched_path}: duplicate ESPN ids")

        snaps = []
        for p in sorted(Path(ratings_dir).glob(f"{season}-week-*.json")):
            m = _NAME.match(p.name)
            if not m:
                continue
            doc = json.loads(p.read_text())
            if int(doc.get("season", season)) != season or not doc.get("ratings"):
                continue
            snaps.append(Snapshot(int(m.group(2)), _ts(doc["computed_at"]), p))
        if not snaps:
            raise MissingSeasonData(
                f"no {season} CFB ratings in {ratings_dir}. The cfb-weekly-job "
                "writes them (deploy/cfb_weekly_runner.py).")
        snaps.sort(key=lambda v: (v.computed_at, v.week))
        return cls(schedule=s, snapshots=snaps, season=season)

    # -- ratings ----------------------------------------------------------------

    def snapshot_for(self, kickoff: pd.Timestamp, asof: pd.Timestamp) -> Snapshot:
        cut = min(kickoff, asof)
        ok = [v for v in self.snapshots if v.computed_at <= cut]
        if not ok:
            raise GameNotPriceable(
                f"no {self.season} CFB ratings were computed before {cut:%Y-%m-%d %H:%M}Z")
        return max(ok, key=lambda v: (v.week, v.computed_at))

    def _table(self, snap: Snapshot) -> dict[str, float]:
        if snap.path not in self._ratings:
            doc = json.loads(snap.path.read_text())
            self._ratings[snap.path] = {r["team"]: float(r["total_rating"])
                                        for r in doc["ratings"]
                                        if r.get("total_rating") is not None}
        return self._ratings[snap.path]

    # -- FeatureSource ----------------------------------------------------------

    def features(self, game_id: str, asof: str) -> GameFeatures:
        g = self._game(game_id)
        t = _ts(asof)
        who = f"{g.away_team}@{g.home_team}"
        if t >= g.start:
            raise LookaheadRefused(
                f"{who} kicked off at {g.start}; pricing it as of {t} would use "
                "a moment after kickoff")
        if bool(g.neutral_site):
            raise GameNotPriceable(
                "neutral site: the CFB vector has no home term to remove, so it "
                "would give a home edge to a team not at home")
        snap = self.snapshot_for(g.start, t)
        behind = int(g.week) - snap.week
        if behind > MAX_WEEKS_BEHIND:
            # Worded without the matchup so a whole stale slate groups as one
            # reason on the board's evidence strip.
            raise StaleRatings(
                f"week {int(g.week)} game, but the newest CFB ratings are week "
                f"{snap.week} ({snap.path.name}), {behind} weeks behind; the "
                "cfb-weekly-job has not produced a newer snapshot")
        table = self._table(snap)
        missing = [x for x in (g.home_team, g.away_team) if x not in table]
        if missing:
            # Generic for the same reason: a week's FCS opponents group as one.
            raise GameNotPriceable(
                f"a team is not rated in {snap.path.name} (usually an FCS opponent)")
        return GameFeatures(rating_diff=table[g.home_team] - table[g.away_team],
                            elo_diff=0.0, elo_present=False)

    def _game(self, game_id: str) -> pd.Series:
        try:
            return self.schedule.loc[str(game_id)]
        except KeyError:
            raise GameNotPriceable(
                f"{game_id!r} is not in the {self.season} ESPN schedule") from None

    # -- slates -----------------------------------------------------------------

    def _upcoming(self) -> pd.DataFrame:
        s = self.schedule[~self.schedule.completed.astype(bool)]
        return s.sort_values(["start"]).assign(
            day=[local_date(x.strftime("%Y-%m-%dT%H:%M:%SZ")) for x in s.sort_values("start").start])

    def slate(self, day: str) -> list[ModelGame]:
        """Unplayed games on one US-Eastern date."""
        s = self._upcoming()
        return [ModelGame(game_id=gid, date=day, home=g.home_team, away=g.away_team)
                for gid, g in s[s.day == day].iterrows()]

    def week_slate(self, day: str) -> tuple[int | None, list[ModelGame]]:
        """The first week with an unplayed game on or after `day`, and its
        unplayed games -- the board shows a CFB week, as it does an NFL one."""
        s = self._upcoming()
        s = s[s.day >= day]
        if s.empty:
            return None, []
        week = int(s.week.min())
        s = s[s.week == week]
        return week, [ModelGame(game_id=gid, date=g.day, home=g.home_team,
                                away=g.away_team) for gid, g in s.iterrows()]

    def game_ids(self) -> list[str]:
        return self.schedule.index.tolist()

    def latest_final(self) -> str | None:
        done = self.schedule[self.schedule.completed.astype(bool)]
        return None if done.empty else done.start.max().strftime("%Y-%m-%dT%H:%M:%SZ")


def build_model(source: CFBLiveSource) -> CFBModel:
    return CFBModel(source)
