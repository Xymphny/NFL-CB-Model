"""CFB features from the committed walk-forward cache.

Same pattern as leagues/nfl/sources.py and for the same reason: the interface
goes in front of the existing implementation so parity is measurable before
anything is rewritten.

WHAT THIS CACHE DOES NOT CARRY
No Elo. The cache is rating_diff and results only, so features report
elo_present=False and select MARGIN_COEFFICIENTS_DVOA_ONLY -- which is the
truth, and which is a genuinely different fitted vector rather than the
ensemble with a zero substituted in.

It also carries no rest and no neutral-site flag, but unlike NFL that costs
nothing here: the shipped CFB vector has no term for either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coverline.leagues.cfb.model import GameFeatures

_ROOT = Path(__file__).resolve().parents[4]
WALK_FORWARD_CACHE = _ROOT / "model" / "cfb_full_walk_forward_cache.csv"

REQUIRED_COLUMNS = {"season", "week", "home_team", "away_team", "rating_diff"}


class GameNotInCache(KeyError):
    """Asked for a game the cache does not contain."""


def game_id(season: int, week: int, home: str, away: str) -> str:
    return f"{season}-W{int(week):02d}-{home}-{away}"


@dataclass(frozen=True)
class CachedWalkForwardSource:
    frame: pd.DataFrame

    @classmethod
    def load(cls, path: Path | str = WALK_FORWARD_CACHE) -> "CachedWalkForwardSource":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CFB walk-forward cache not found at {path}")
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"cache is missing columns: {sorted(missing)}")
        if df[sorted(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError("cache has nulls in required columns")
        df = df.copy()
        df["game_id"] = [game_id(s, w, h, a) for s, w, h, a
                         in zip(df.season, df.week, df.home_team, df.away_team)]
        dupes = df.game_id[df.game_id.duplicated()].tolist()
        if dupes:
            raise ValueError(f"duplicate game ids: {dupes[:5]}")
        return cls(frame=df.set_index("game_id"))

    def features(self, game_id: str, asof: str) -> GameFeatures:
        try:
            row = self.frame.loc[game_id]
        except KeyError:
            raise GameNotInCache(
                f"{game_id!r} is not in the CFB cache ({len(self.frame)} games, "
                f"{self.seasons[0]}-{self.seasons[-1]}). Refusing to price it."
            ) from None
        return GameFeatures(rating_diff=float(row.rating_diff),
                            elo_diff=0.0, elo_present=False)

    @property
    def seasons(self) -> list[int]:
        return sorted(self.frame.season.unique().tolist())

    def __len__(self) -> int:
        return len(self.frame)

    def game_ids(self, season: int | None = None) -> list[str]:
        f = self.frame if season is None else self.frame[self.frame.season == season]
        return f.index.tolist()

    def actual_margin(self, game_id: str) -> float:
        return float(self.frame.loc[game_id].actual_margin)
