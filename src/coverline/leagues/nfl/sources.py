"""Feature sources for the NFL model.

BRANCH BY ABSTRACTION, NOT REWRITE
The legacy rating pipeline (model/ratings.py and friends) computes rating_diff
from play-by-play through VOA, opponent adjustment and recency weighting. That
is a large body of numerically delicate code, and rewriting it against the new
interfaces first would mean the new core's numbers could differ from the
board's with no way to tell whether the difference was a bug or an improvement.

So the interface goes in front of the EXISTING implementation. This module
serves features out of the committed walk-forward cache the legacy pipeline
produced, which makes the NFL package able to price real historical boards
today and -- more usefully -- makes parity against the legacy model measurable.
Only once a parity harness exists is it safe to swap the implementation behind
the interface, which is the whole point of the pattern.

WHY THE CACHE IS POINT-IN-TIME CORRECT
model/expanded_walk_forward_cache.csv is a walk-forward artifact: each row's
rating_diff was computed using only weeks strictly before that game. That is a
property of how the cache was built, not something this module can verify, so
``CachedWalkForwardSource`` enforces what it CAN check -- that a request's asof
timestamp is not earlier than the game it asks about, which catches the obvious
misuse of asking for a game's features before its season started.

WHAT THIS SOURCE CANNOT DO, STATED PLAINLY
The cache carries rating_diff, home_field and rest_diff. It does NOT carry the
NGS differences (cpoe, separation, yac_oe, ryoe) or elo_diff. So it can only
drive MARGIN_COEFFICIENTS_V1_RATING_ONLY, the vector the board uses when NGS is
absent. Features returned therefore set ngs_present=False, which is the truth
rather than a convenience: pretending NGS is present with zero-valued
differences would silently select the full-ensemble vector and price every game
with the wrong coefficients -- the exact failure that ran for weeks before the
audit caught it.

A full-ensemble source needs the NGS feature cache ported too, and has its own
ledger row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coverline.leagues.nfl.model import GameFeatures

_ROOT = Path(__file__).resolve().parents[4]
WALK_FORWARD_CACHE = _ROOT / "model" / "expanded_walk_forward_cache.csv"

REQUIRED_COLUMNS = {
    "season", "week", "home_team", "away_team",
    "rating_diff", "home_field", "rest_diff",
}


class GameNotInCache(KeyError):
    """Asked for a game the cache does not contain.

    Raised rather than returning neutral features. A model that quietly
    prices an unknown game at rating_diff=0 produces a confident number with
    no information in it, which is worse than refusing.
    """


def game_id(season: int, week: int, home: str, away: str) -> str:
    """Canonical id: 2023-W07-KC-DEN. Stable, sortable, human-readable."""
    return f"{season}-W{int(week):02d}-{home}-{away}"


@dataclass(frozen=True)
class CachedWalkForwardSource:
    """Serves features from the legacy walk-forward cache.

    Implements nfl.model.FeatureSource.
    """

    frame: pd.DataFrame

    @classmethod
    def load(cls, path: Path | str = WALK_FORWARD_CACHE) -> "CachedWalkForwardSource":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"walk-forward cache not found at {path}. Without it the NFL "
                "package has no features and cannot price anything."
            )
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"cache is missing columns: {sorted(missing)}")
        if df[sorted(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(
                "cache contains nulls in required columns; a null rating_diff "
                "would silently become a neutral prediction"
            )
        df = df.copy()
        df["game_id"] = [
            game_id(s, w, h, a) for s, w, h, a
            in zip(df.season, df.week, df.home_team, df.away_team)
        ]
        dupes = df.game_id[df.game_id.duplicated()].tolist()
        if dupes:
            raise ValueError(f"duplicate game ids in cache: {dupes[:5]}")
        return cls(frame=df.set_index("game_id"))

    # -- FeatureSource ----------------------------------------------------

    def features(self, game_id: str, asof: str) -> GameFeatures:
        try:
            row = self.frame.loc[game_id]
        except KeyError:
            raise GameNotInCache(
                f"{game_id!r} is not in the walk-forward cache "
                f"({len(self.frame)} games, {self.seasons[0]}-{self.seasons[-1]}). "
                "Refusing to price it rather than returning neutral features."
            ) from None

        self._check_asof(game_id, asof)

        return GameFeatures(
            rating_diff=float(row.rating_diff),
            rest_diff=float(row.rest_diff),
            is_neutral_site=not bool(row.home_field),
            # The cache has no NGS features and no elo. Saying so is what
            # selects the correct coefficient vector.
            ngs_present=False,
            elo_diff=0.0,
        )

    # -- helpers ----------------------------------------------------------

    @property
    def seasons(self) -> list[int]:
        return sorted(self.frame.season.unique().tolist())

    def __len__(self) -> int:
        return len(self.frame)

    def game_ids(self, season: int | None = None) -> list[str]:
        f = self.frame if season is None else self.frame[self.frame.season == season]
        return f.index.tolist()

    def actual_margin(self, game_id: str) -> float:
        """The realised result, for grading. Separate from features() on
        purpose: a feature source that could hand back the outcome alongside
        the inputs is one bad import away from leaking it into a fit."""
        if "actual_margin" not in self.frame.columns:
            raise KeyError("this cache carries no results")
        return float(self.frame.loc[game_id].actual_margin)

    def _check_asof(self, gid: str, asof: str) -> None:
        """The part of the point-in-time contract this source CAN enforce.

        The cache's walk-forward property is a fact about how it was built and
        cannot be checked from here. What can be checked is that nobody asks
        for a game's features dated before that game's season began -- which
        is the shape of a backtest looping over the wrong index.
        """
        try:
            when = datetime.fromisoformat(asof.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"asof must be an ISO-8601 timestamp, got {asof!r}") from None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        season = int(self.frame.loc[gid].season)
        season_start = datetime(season, 8, 1, tzinfo=timezone.utc)
        if when < season_start:
            raise ValueError(
                f"asof {asof} is before the {season} season began; a feature "
                f"request dated before its own game is a backtest indexing bug"
            )
