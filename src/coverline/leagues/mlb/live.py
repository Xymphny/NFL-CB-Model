"""Live MLB features: tonight's games priced from the graded walk-forward state.

WHY sources.WalkForwardSource CANNOT DO THIS
It serves the walk-forward's own rows, and the walk-forward only emits a row
for a game that has a final score. Tonight's games have none, so it can price
yesterday perfectly and tonight not at all. It also replays the CURRENT
season alone, so every rating starts cold in March -- while every fit and
grade here (fit_mlb_rules, measure_mlb_rules, mlb_backtest) ran the
walk-forward continuously across seasons on the historical cache. Pricing
from a cold start would be a different model from the one that was measured.

WHAT THIS DOES
Replays model.mlb_model.run_walk_forward over the historical cache plus the
current season -- load_mlb_caches, the same frames the legacy runner reads --
up to the moment asked, takes the WalkForwardState it leaves, and asks that
state for expected runs given tonight's probable starters and park. The loop
and the arithmetic are the graded code's; this file only chooses the cutoff.

The parity test holds it there: a historical game priced from the state
before it reproduces the expected runs the walk-forward produced for it.

POINT IN TIME
The results cache has dates, not first-pitch times, so the cutoff is the
DATE: tonight's games are priced from every game before today. The second
game of a doubleheader is therefore priced without the first game's result,
which is the conservative side of the line and what a price taken before the
first pitch of the day would know anyway.

REFUSALS
No probable for either side refuses the game: books quote conditional on the
listed starters, and a model price without one is a price for a different
game. A results cache missing any game the slate says went final on the
previous date refuses everything, as does a slate pulled while earlier games
were still in progress (see model/ingest/mlb_slate.py and check_fresh).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coverline.execution.matching import ModelGame
from coverline.leagues.mlb.model import GameFeatures, MLBModel, load_rules

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.mlb_model import run_walk_forward  # noqa: E402

SLATE_DIR = _ROOT / "data" / "raw" / "mlb" / "slates"
DATA_DIR = _ROOT / "model"


class NoSlate(FileNotFoundError):
    """No slate pulled for that date."""


class StaleResults(RuntimeError):
    """The results cache does not reach the slate's previous final."""


class GameNotPriceable(KeyError):
    """This game cannot be priced honestly; the message says why."""


class LookaheadRefused(ValueError):
    """Asked to price a game as of a moment after it started."""


def _ts(asof) -> pd.Timestamp:
    if isinstance(asof, str) and asof == "now":
        return pd.Timestamp(datetime.now(timezone.utc))
    t = pd.Timestamp(asof)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def latest_slate(day: str, slate_dir: Path = SLATE_DIR) -> Path:
    found = sorted(Path(slate_dir).glob(f"{day}-*.json"))
    if not found:
        raise NoSlate(
            f"no slate for {day} in {slate_dir}. Pull one with "
            f"`python model/ingest/mlb_slate.py --date {day}`.")
    return found[-1]


def load_caches(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Historical plus current -- deploy.mlb_daily_update.load_mlb_caches,
    imported rather than restated."""
    from deploy.mlb_daily_update import load_mlb_caches
    return load_mlb_caches(str(data_dir))


def state_before(schedule: pd.DataFrame, pitching: pd.DataFrame,
                 day: str, game_key: str = ""):
    """The walk-forward state after every game sorting before (day, game_key).

    run_walk_forward orders by (date, game_key), so this cutoff is a prefix of
    exactly the sequence the graded code walks. game_key="" means "before the
    first game of `day`", which is what live pricing uses.
    """
    before = schedule[(schedule.date < day)
                      | ((schedule.date == day) & (schedule.game_key < game_key))]
    out: dict = {}
    run_walk_forward(before, pitching, state_out=out)
    return out["state"]


def check_fresh(slate: dict, sched: pd.DataFrame) -> None:
    """Refuse unless every result the slate says exists is in the cache.

    A date comparison was the first version and it was not enough: a cache
    holding five of a night's fifteen results reaches the same date as one
    holding all fifteen. See model/ingest/mlb_slate.py.
    """
    if slate.get("unfinished_keys"):
        raise StaleResults(
            f"this slate was pulled while {len(slate['unfinished_keys'])} earlier "
            f"game(s) were in progress ({', '.join(slate['unfinished_keys'][:4])}), "
            "so its record of what went final is incomplete. Re-pull it once "
            "they finish: `python model/ingest/mlb_slate.py --date "
            f"{slate['date']}`.")
    played = sched.dropna(subset=["home_score", "away_score"])
    prev = slate.get("previous_final_date")
    keys = slate.get("previous_final_keys")
    if prev and keys is None:
        raise StaleResults(
            "this slate predates game-level freshness checks; re-pull it")
    missing = sorted(set(keys or ()) - set(played.game_key))
    if missing:
        raise StaleResults(
            f"{len(missing)} of {len(keys)} results from {prev} are not in the "
            f"results cache ({', '.join(missing[:4])}). Pricing now would use "
            "ratings that have not seen them; run deploy/mlb_daily_update.py "
            "first.")


@dataclass
class MLBLiveSource:
    payload: dict
    schedule: pd.DataFrame
    pitching: pd.DataFrame
    _state: object = field(default=None, repr=False)

    @classmethod
    def load(cls, day: str, slate_dir: Path = SLATE_DIR,
             data_dir: Path = DATA_DIR) -> "MLBLiveSource":
        slate = json.loads(latest_slate(day, slate_dir).read_text())
        sched, pit = load_caches(data_dir)
        check_fresh(slate, sched)
        return cls(payload=slate, schedule=sched, pitching=pit)

    @property
    def day(self) -> str:
        return self.payload["date"]

    def _game(self, game_id: str) -> dict:
        for g in self.payload["games"]:
            if g["game_key"] == game_id:
                return g
        raise GameNotPriceable(f"{game_id!r} is not on the {self.day} slate")

    def features(self, game_id: str, asof: str) -> GameFeatures:
        g = self._game(game_id)
        if g.get("start_utc") and _ts(asof) > _ts(g["start_utc"]):
            raise LookaheadRefused(
                f"{g['away_team']}@{g['home_team']} started at {g['start_utc']}")
        if not (g.get("home_sp") and g.get("away_sp")):
            raise GameNotPriceable(
                f"{g['away_team']}@{g['home_team']}: no probable starter for "
                f"{'home' if not g.get('home_sp') else 'away'}. Lines are quoted "
                "conditional on the listed starters; refusing to price a game "
                "whose starters are unknown.")
        if g.get("park") in (None, "UNK"):
            raise GameNotPriceable(f"{g['home_team']} has no known park")
        if self._state is None:
            self._state = state_before(self.schedule, self.pitching, self.day)
        st = self._state
        eh = st.expected_runs(g["home_team"], g["away_sp"], g["away_team"], g["park"])
        ea = st.expected_runs(g["away_team"], g["home_sp"], g["home_team"], g["park"])
        return GameFeatures(exp_home=float(eh), exp_away=float(ea))

    def slate_games(self) -> list[ModelGame]:
        out = []
        for g in self.payload["games"]:
            if g.get("state") not in (None, "Preview"):
                continue          # started, final, or postponed
            if "postpon" in str(g.get("detailed_state", "")).lower():
                continue
            out.append(ModelGame(game_id=g["game_key"], date=self.day,
                                 home=g["home_team"], away=g["away_team"],
                                 order=int(g.get("game_number", 0))))
        return out

    def slate(self, day: str) -> list[ModelGame]:
        if day != self.day:
            raise NoSlate(f"this source holds {self.day}, not {day}")
        return self.slate_games()


def build_model(source: MLBLiveSource) -> MLBModel:
    """With the ninth-inning layer, which is what returns the moneyline."""
    return MLBModel(source, rules=load_rules())
