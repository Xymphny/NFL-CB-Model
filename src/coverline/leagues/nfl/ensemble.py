"""The full ensemble on the live board: NGS and Elo around the ratings source.

WHY THIS EXISTS
GameWeekSource serves rating_diff, rest and neutral site, and declares
ngs_present=False on every game -- honestly, because NGS and Elo came from
runtime fetches inside the legacy job and nothing in the core fetched them.
So the core board priced every NFL game on MARGIN_COEFFICIENTS_V1_RATING_ONLY
while the legacy odds-watch priced the same games on the full ensemble. In
the 2016-2025 walk-forward the rating-only path took the underdog 80-86% of
the time and went 47.0% ATS on those picks (n 387); the ensemble 53.6% (n 921).
Both are upper bounds. This is the ledger row live.py names.

WHAT IT FETCHES, AND FROM WHERE -- THE LEGACY JOB'S OWN INPUTS
  NGS   coverline.leagues.nfl.ngs.team_features(season, through_week=week):
        the legacy model/layer2_ngs frame, weeks strictly before the one
        priced, re-keyed so the Rams are LA (ADR 0004 -- see below).
  Elo   model.elo_rating.compute_elo_walk_forward over the last ten seasons
        of nflverse results, the same call and window odds_watch_job makes.
No new input and no coefficient change. The vectors are nfl.model's, which
carry model/prediction's verbatim.

ngs_present IS SET PER GAME FROM WHAT WAS ACTUALLY FETCHED
Both teams in the NGS frame AND both with an Elo rating, or the game stays
on the rating-only vector. The Elo half is stricter than legacy on purpose:
legacy prices an NGS game without Elo on MARGIN_COEFFICIENTS_PRE_ELO, a third
vector the core does not carry. Falling back to rating-only here is the
honest degradation; `reason` on each game says which happened.

ONE DELIBERATE DIFFERENCE FROM LEGACY: THE RAMS
The core's NGS frame normalises LAR -> LA; the legacy join does not, so
legacy prices every Rams game rating-only. ADR 0004 records this, measured
the fix as neutral (t = +0.12 on 26 held-out games), and names this moment
as its revisit trigger: "if the new core takes over the live board, the
normalisation comes with it and Rams games change coefficient vector."
model/nfl_ensemble_parity.py reports that game separately and checks it
under the legacy vocabulary too.

POINT IN TIME
NGS is weeks < W by the legacy filter. Elo is computed from results before
the priced game's date and not after `asof`, once per distinct game day --
a Sunday price never sees Sunday's scores, whenever it runs.

DE-BIAS IS STILL NOT APPLIED HERE (see live.py): it is a property of the
slate, measured by the board assembler (execution/board.slate_debias).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from coverline.leagues.nfl import ngs as N
from coverline.leagues.nfl.live import GameWeekSource, RATINGS_DIR, _parse_ts
from coverline.leagues.nfl.model import GameFeatures

#: Seasons of results the Elo walk runs over, counting back from the one
#: priced -- odds_watch_job's range(season - 10, season + 1).
ELO_SEASONS_BACK = 10

FULL = "full_ensemble"
RATING_ONLY = "v1_rating_only_no_ngs"   # the legacy board's label for it


def _default_ngs(season: int, week: int) -> pd.DataFrame:
    return N.team_features(season, through_week=week)


def _default_history(season: int) -> pd.DataFrame:
    from ingest.nfl_schedules import load_schedules
    return load_schedules(seasons=list(range(season - ELO_SEASONS_BACK, season + 1)))


def elo_before(history: pd.DataFrame, day: str) -> dict[str, float]:
    """Each team's Elo after every result dated strictly before `day`."""
    from model.elo_rating import compute_elo_walk_forward
    h = history.copy()
    h.loc[h["gameday"].astype(str).str[:10] >= str(day)[:10],
          ["home_score", "away_score"]] = None
    _, elo = compute_elo_walk_forward(h)
    return elo


@dataclass
class EnsembleWeekSource:
    """Implements nfl.model.FeatureSource, full ensemble where fetched."""

    base: GameWeekSource
    ngs_frame: pd.DataFrame | None
    history: pd.DataFrame | None
    status: dict
    _elo_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, season: int, week: int, schedule: pd.DataFrame,
             ratings_dir=RATINGS_DIR,
             ngs_loader: Callable[[int, int], pd.DataFrame] = _default_ngs,
             history_loader: Callable[[int], pd.DataFrame] = _default_history,
             ) -> "EnsembleWeekSource":
        """A fetch that fails is recorded and degrades that input, never the
        board: every game then prices on the rating-only vector, which is
        what the board did before this source existed."""
        base = GameWeekSource.for_game_week(season, week, schedule, ratings_dir)
        status = {"ngs": True, "elo": True}
        try:
            frame = ngs_loader(season, week)
        except Exception as e:
            frame = None
            status.update(ngs=False, ngs_error=f"{type(e).__name__}: {str(e)[:200]}")
        try:
            hist = history_loader(season)
        except Exception as e:
            hist = None
            status.update(elo=False, elo_error=f"{type(e).__name__}: {str(e)[:200]}")
        return cls(base=base, ngs_frame=frame, history=hist, status=status)

    # -- helpers ----------------------------------------------------------

    teams = staticmethod(GameWeekSource.teams)

    def game_ids(self) -> list[str]:
        return self.base.game_ids()

    def version_for(self, gameday: str, asof: str = "now") -> tuple:
        return self.base.version_for(gameday, asof)

    def _gameday(self, home: str, away: str) -> str:
        s = self.base.schedule
        m = s[(s.home_team == home) & (s.away_team == away)]
        if m.empty:
            raise KeyError(f"{home} vs {away} is not on the {self.base.season} "
                           f"week {self.base.week} schedule")
        return str(m.iloc[0]["gameday"])[:10]

    def _elo(self, gameday: str, asof: str) -> dict[str, float] | None:
        if self.history is None:
            return None
        cut = (datetime.now(timezone.utc) if asof in (None, "now") else _parse_ts(asof))
        # Results dated before the game, and none from asof's day onward: a
        # team plays once a week, so the only results this drops are ones
        # the two teams in this game were not in.
        day = min(gameday, cut.date().isoformat())
        if day not in self._elo_cache:
            self._elo_cache[day] = elo_before(self.history, day)
        return self._elo_cache[day]

    # -- per game ---------------------------------------------------------

    def resolve(self, game_id: str, asof: str = "now") -> tuple[GameFeatures, str, str | None]:
        """(features, coefficient set, why not the full ensemble or None)."""
        f = self.base.features(game_id, asof)
        home, away = self.teams(game_id)
        if self.ngs_frame is None:
            return f, RATING_ONLY, "NGS feed unavailable"
        if not N.ngs_present(self.ngs_frame, home, away):
            missing = [t for t in (home, away) if t not in self.ngs_frame.index]
            return f, RATING_ONLY, f"no NGS for {', '.join(missing)}"
        elo = self._elo(self._gameday(home, away), asof)
        if elo is None:
            return f, RATING_ONLY, "Elo unavailable"
        if home not in elo or away not in elo:
            missing = [t for t in (home, away) if t not in elo]
            return f, RATING_ONLY, f"no Elo for {', '.join(missing)}"
        g = self.ngs_frame
        return replace(
            f, ngs_present=True,
            cpoe_diff=float(g.loc[home, "team_cpoe"] - g.loc[away, "team_cpoe"]),
            separation_diff=float(g.loc[home, "team_avg_separation"]
                                  - g.loc[away, "team_avg_separation"]),
            yac_oe_diff=float(g.loc[home, "team_yac_over_expected"]
                              - g.loc[away, "team_yac_over_expected"]),
            ryoe_diff=float(g.loc[home, "team_ryoe"] - g.loc[away, "team_ryoe"]),
            elo_diff=float(elo[home] - elo[away]),
        ), FULL, None

    def features(self, game_id: str, asof: str) -> GameFeatures:
        return self.resolve(game_id, asof)[0]

    def provenance(self, game_id: str, asof: str = "now") -> dict:
        """What the board records per game: the legacy feature_values shape,
        plus the vector that ran and why, so a published margin can be
        re-derived (tests/core/test_board_provenance.py's standard)."""
        f, cset, reason = self.resolve(game_id, asof)
        return {
            "coefficient_set": cset,
            "ngs_present": f.ngs_present,
            "reason": reason,
            "feature_values": {
                "rating_diff": f.rating_diff, "rest_diff": f.rest_diff,
                "cpoe_diff": f.cpoe_diff, "separation_diff": f.separation_diff,
                "yac_oe_diff": f.yac_oe_diff, "ryoe_diff": f.ryoe_diff,
                "elo_diff": f.elo_diff if f.ngs_present else None,
                "is_neutral_site": f.is_neutral_site,
            },
        }
