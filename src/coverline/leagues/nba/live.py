"""Live NBA features: the graded walk-forward, replayed up to the moment asked.

WHAT THIS IS, AND WHAT IT IS NOT
data/nba_fitted.json holds the ratings at the END of 2022-23. Pricing a
2026-27 game from them would carry three seasons of stale ratings, the
exact failure the static fit measured at t = -6.96. So this source does not
read those ratings to price anything. It REPLAYS the graded procedure --
model/fit_nba_walkforward.walk_forward, with the hyperparameters from the
artifact -- over every completed game from the first tuning season up to the
instant `asof`, and prices from the state that leaves.

It imports the loader and the walk-forward rather than re-implementing them,
so a live price and a graded one cannot drift apart. The parity tests hold
it to that: a full replay of 2021-2023 reproduces every rating in the
artifact, and a historical game priced as of its own tip-off reproduces the
margin the graded walk-forward predicted for it.

THE SEASON IN FRONT OF IT HAS TO BE COMPLETE BEHIND IT
The ratings carry across seasons (carryover 0.5), so a missing season is not
a gap the replay can step over -- it would price 2026-27 off 2024-25 ratings
with one carryover where two were due. Every season from the first tuning
season to the current one must be on disk, and `load` names the first file
that is not.

FRESHNESS IS PROVEN FROM THE FILE, NOT ASSUMED
The current-season file is a snapshot. If it was pulled yesterday morning,
last night's games are in it as unfinished, and the ratings would silently
miss them. Any regular-season game that tipped more than STALE_AFTER before
`asof` and is neither final nor postponed means the snapshot is behind, and
this refuses rather than pricing from ratings a night old. A game involving
either team that has started and not finished refuses that game alone.

NEUTRAL-SITE GAMES ARE REFUSED. The model applies a home advantage to the
listed home team; at a neutral site there is none to apply.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from coverline.execution.matching import ModelGame, local_date
from coverline.leagues.nba.model import (
    FITTED_PATH, GameFeatures, NBAModel, constant_sigma, load_fitted,
)

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import fit_nba, fit_nba_walkforward  # noqa: E402

DATA = fit_nba.DATA

#: A game this long past tip-off that is still not final means the snapshot
#: is behind, not that the game is running long.
STALE_AFTER = timedelta(hours=6)


class SeasonGap(FileNotFoundError):
    """A season the replay needs is not on disk."""


class StaleResults(RuntimeError):
    """The current-season file is behind the moment being priced."""


class GameNotPriceable(KeyError):
    """This game cannot be priced honestly; the message says why."""


class LookaheadRefused(ValueError):
    """Asked to price a game as of a moment after it started."""


def season_of(d: date) -> int:
    """NBA seasons are named by the year they END. October 2026 is 2027."""
    return d.year + 1 if d.month >= 8 else d.year


def _ts(asof: str | datetime) -> pd.Timestamp:
    if isinstance(asof, str) and asof == "now":
        return pd.Timestamp(datetime.now(timezone.utc))
    t = pd.Timestamp(asof)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _schedule(path: str) -> pd.DataFrame:
    """Every regular-season game in one season file, played or not."""
    d = pd.read_parquet(path)
    d = d[(d.season_type == 2) & (d.type_abbreviation == "STD")]
    d = d.dropna(subset=["home_abbreviation", "away_abbreviation"])
    status = d.get("status_type_name", pd.Series("", index=d.index)).fillna("")
    out = pd.DataFrame({
        "game_id": d["id"].astype(str),
        "season": d["season"].astype(int),
        "tip": pd.to_datetime(d["date"], utc=True),
        "home": d["home_abbreviation"],
        "away": d["away_abbreviation"],
        "completed": d["status_type_completed"].fillna(False).astype(bool),
        "postponed": status.str.contains("POSTPONED|CANCELED|CANCELLED"),
        "neutral": d.get("neutral_site", pd.Series(False, index=d.index))
                    .fillna(False).astype(bool),
    })
    return out.set_index("game_id")


@dataclass
class NBALiveSource:
    history: pd.DataFrame
    schedule: pd.DataFrame
    hyperparameters: dict
    season: int
    _memo: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, season: int, data: str = DATA,
             fitted: Path = FITTED_PATH) -> "NBALiveSource":
        art = load_fitted(fitted)
        first = int(art["_provenance"]["tune_seasons"][0])
        seasons = list(range(first, season + 1))
        missing = [data.format(year=y) for y in seasons
                   if not Path(data.format(year=y)).exists()]
        if missing:
            raise SeasonGap(
                f"the replay needs every season from {first} to {season}, and "
                f"{len(missing)} are missing: {missing}. Ratings carry across "
                "seasons, so skipping one would price off ratings a season "
                "stale. Pull them (model/ingest/nba_espn.py) and retry."
            )
        history = fit_nba.load(seasons, data=data)
        history = history.assign(tip=pd.to_datetime(history["date"], utc=True))
        return cls(history=history, schedule=_schedule(data.format(year=season)),
                   hyperparameters=dict(art["hyperparameters"]), season=season)

    # -- the replay ---------------------------------------------------------

    def ratings_before(self, asof: str | datetime) -> tuple[dict, int | None]:
        """Ratings from every game completed strictly before `asof`, and the
        last season those games belong to."""
        t = _ts(asof)
        past = self.history[self.history.tip < t]
        key = len(past)
        if key not in self._memo:
            state: dict = {}
            hp = self.hyperparameters
            if len(past):
                fit_nba_walkforward.walk_forward(
                    past.drop(columns=["tip"]), hp["k"], hp["home_adv"],
                    hp["carryover"], state=state)
            last = int(past.season.max()) if len(past) else None
            self._memo = {key: (state.get("ratings", {}), last)}
        return self._memo[key]

    # -- FeatureSource --------------------------------------------------------

    def features(self, game_id: str, asof: str) -> GameFeatures:
        if game_id not in self.schedule.index:
            raise GameNotPriceable(f"{game_id!r} is not in the {self.season} schedule")
        g = self.schedule.loc[game_id]
        t = _ts(asof)
        if t > g.tip:
            raise LookaheadRefused(
                f"{g.away}@{g.home} tipped at {g.tip}; pricing it as of {t} "
                "would use a moment after the game began")
        if g.neutral:
            raise GameNotPriceable(
                f"{g.away}@{g.home} is at a neutral site; the model's home "
                "advantage would be applied to a team that is not at home")
        self._refuse_stale(t)
        live = self.schedule[(self.schedule.tip < t) & ~self.schedule.completed
                             & ~self.schedule.postponed
                             & (self.schedule.home.isin([g.home, g.away])
                                | self.schedule.away.isin([g.home, g.away]))]
        if len(live):
            raise GameNotPriceable(
                f"{g.away}@{g.home}: a game involving one of these teams has "
                "started and is not final, so its result is not in the ratings")

        ratings, last = self.ratings_before(t)
        for team in (g.home, g.away):
            if team not in ratings:
                raise GameNotPriceable(
                    f"{team} has no rating; refusing to price it as league "
                    "average (an abbreviation change is the likely cause)")
        rh, ra = ratings[g.home], ratings[g.away]
        if last is not None and int(g.season) != last:
            if int(g.season) != last + 1:
                raise SeasonGap(f"ratings end in {last}; the game is in {g.season}")
            c = self.hyperparameters["carryover"]
            rh, ra = rh * c, ra * c
        # No pace model exists. NaN, not a plausible-looking number, so any
        # code that starts using it fails where it starts.
        return GameFeatures(rating_diff=rh - ra, pace=math.nan)

    def _refuse_stale(self, t: pd.Timestamp) -> None:
        s = self.schedule
        behind = s[(s.tip < t - STALE_AFTER) & ~s.completed & ~s.postponed]
        if len(behind):
            raise StaleResults(
                f"{len(behind)} regular-season game(s) tipped more than "
                f"{STALE_AFTER} before {t} and are not final in the "
                f"{self.season} file (latest {behind.tip.max()}). The snapshot "
                "is behind; re-pull it before pricing.")

    # -- slate ----------------------------------------------------------------

    def slate(self, day: str) -> list[ModelGame]:
        s = self.schedule[~self.schedule.completed & ~self.schedule.postponed]
        out = []
        for gid, g in s.sort_values("tip").iterrows():
            iso = g.tip.strftime("%Y-%m-%dT%H:%M:%SZ")
            if local_date(iso) == day:
                out.append(ModelGame(game_id=str(gid), date=day,
                                     home=g.home, away=g.away))
        return out

    def commence(self, game_id: str) -> pd.Timestamp:
        return self.schedule.loc[game_id].tip


def build_model(source: NBALiveSource, fitted: Path = FITTED_PATH) -> NBAModel:
    """The graded model on a live source: rating difference plus the fitted
    home advantage, with the constant sigma the evidence supports."""
    art = load_fitted(fitted)
    ha = float(art["hyperparameters"]["home_adv"])
    return NBAModel(source, sigma=constant_sigma(art),
                    margin_model=lambda f: f.rating_diff + ha)
