"""MLB features from the in-season caches the cron keeps current.

Same branch-by-abstraction pattern as the football packages: the interface
goes in front of model/mlb_model.run_walk_forward rather than reimplementing
its credibility machinery, which is the part most expensive to get subtly
wrong.

THE CACHE LAGS, AND THAT IS STATED RATHER THAN HIDDEN
model/mlb_schedule_current.csv is refreshed by a scheduled job, so its last
date is behind today by however long since the last run. A source that
silently served stale expected runs for "today's" game would be the worst
kind of wrong -- confident and plausible. `as_of_date` reports the cache's own
horizon, and `features()` refuses a game the cache does not contain rather
than approximating one.

POSTPONED GAMES ARE NOT FUTURE GAMES
The schedule carries rows with no score for both. Some are genuinely upcoming;
most, mid-season, are postponements. The walk-forward already refuses to
ingest them -- a single unplayed April row once turned 2,258 of 2,351
downstream predictions into NaN -- and this source inherits that guard rather
than re-deriving it. What it adds is refusing to SERVE a row the walk-forward
produced no expectation for, since a NaN expected-runs value would otherwise
reach the distribution and fail far from its cause.

FOR PRICING TONIGHT, USE live.py, NOT THIS. This source replays the CURRENT
season alone -- every rating starts cold in March, while every fit and grade
ran the historical cache continuously -- and it can only serve games that
already have a final score. It remains for what it does correctly: serving
the walk-forward's own rows for completed games.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coverline.leagues.mlb.model import GameFeatures

_ROOT = Path(__file__).resolve().parents[4]
SCHEDULE_CURRENT = _ROOT / "model" / "mlb_schedule_current.csv"
PITCHING_CURRENT = _ROOT / "model" / "mlb_pitching_current.csv"


class GameNotPriceable(KeyError):
    """The walk-forward produced no expectation for that game."""


@dataclass(frozen=True)
class WalkForwardSource:
    """Serves expected runs from a walk-forward over the current caches."""

    frame: pd.DataFrame
    as_of_date: str

    @classmethod
    def load(cls, schedule: Path | str = SCHEDULE_CURRENT,
             pitching: Path | str = PITCHING_CURRENT) -> "WalkForwardSource":
        import sys
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from model.mlb_model import run_walk_forward

        sched = pd.read_csv(schedule)
        pitch = pd.read_csv(pitching)
        walk = run_walk_forward(sched, pitch)

        before = len(walk)
        walk = walk.dropna(subset=["exp_home", "exp_away"])
        if len(walk) < before:
            # Not silent: a NaN expectation reaching the distribution would
            # fail far from its cause.
            print(f"[mlb] dropped {before - len(walk)} rows with no expectation")

        if "game_key" not in walk.columns:
            raise ValueError("walk-forward output has no game_key to index on")
        return cls(frame=walk.set_index("game_key"),
                   as_of_date=str(sched.date.max()))

    # -- FeatureSource ----------------------------------------------------

    def features(self, game_id: str, asof: str) -> GameFeatures:
        try:
            row = self.frame.loc[game_id]
        except KeyError:
            raise GameNotPriceable(
                f"{game_id!r} has no walk-forward expectation. The current "
                f"cache runs to {self.as_of_date}; a game after that has not "
                "been computed, and a postponed game never will be. Refusing "
                "to approximate one."
            ) from None
        return GameFeatures(exp_home=float(row.exp_home),
                            exp_away=float(row.exp_away))

    # -- helpers ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    def game_ids(self) -> list[str]:
        return self.frame.index.tolist()

    def games_on(self, date: str) -> list[str]:
        return self.frame[self.frame.date == date].index.tolist()

    def staleness_days(self, today: str) -> int:
        """How far the cache is behind a given date.

        Exposed so a caller can decide. A source that quietly served
        three-day-old expected runs for tonight's game would be confidently
        wrong, which is worse than being unavailable.
        """
        return (pd.Timestamp(today) - pd.Timestamp(self.as_of_date)).days
