"""Live NHL features: the graded no-pull walk-forward, replayed within season.

WHICH PROCEDURE, AND WHY IT RESETS EVERY SEASON
The rates were graded walking forward WITHIN one season from an opening
pseudo-game (model/fit_nhl_walkforward.py); a static cross-season fit
measured t = -1.33 and did not ship. The rules layer that makes the puck line
priceable was graded on NO-PULL regulation scores rated the same way
(model/fit_nhl_rules.rate). So a live price is that exact method applied to
the season in progress: every final game of the current season before the
moment asked, and nothing from any earlier season.

That means the prior season is NOT needed here, unlike basketball, and early
in a season the rates sit close to league average because that is what the
graded procedure does in its first weeks.

WHAT IT IMPORTS, SO IT CANNOT DRIFT
model.fit_nhl_rules.load builds the no-pull scores from the league's own
goal-level data; model.fit_nhl_walkforward.walk_forward rates them. The price
for the next game is read off the walk-forward itself, by appending that game
to the history and taking its row, so the base rate, the ordering and the
update are the graded code's and not a restatement of them.

ONLY NO-PULL RATES ARE PRODUCED
NHLModel composes the goalie-pull and overtime layers on top of rates with
both goalies on the ice. Final-score rates already contain the empty-net
goals that layer adds, so they are deliberately NOT produced here: lam_home
and lam_away are NaN, and build_model always composes the rules, which read
only the no-pull pair.

FRESHNESS
data/raw/nhl/schedule_{season}.parquet lists every game with its state. A
game that started more than STALE_AFTER before `asof` and is still not final
means the pull is behind, and pricing refuses. So does a final with no goal
rows, which would otherwise enter the no-pull ratings as a 0-0 game.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from coverline.execution.matching import ModelGame, local_date
from coverline.leagues.nhl.model import GameFeatures, NHLModel, load_rules

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import fit_nhl_rules, fit_nhl_walkforward  # noqa: E402
from model.ingest.nhl_api import FINAL_STATES, POSTPONED_STATES  # noqa: E402

RAW = fit_nhl_rules.RAW
STALE_AFTER = timedelta(hours=6)


class MissingSeasonData(FileNotFoundError):
    """A file the current season needs is not on disk."""


class StaleResults(RuntimeError):
    """The pull is behind the moment being priced."""


class GameNotPriceable(KeyError):
    """This game cannot be priced honestly; the message says why."""


class LookaheadRefused(ValueError):
    """Asked to price a game as of a moment after it started."""


def _ts(asof) -> pd.Timestamp:
    if isinstance(asof, str) and asof == "now":
        return pd.Timestamp(datetime.now(timezone.utc))
    t = pd.Timestamp(asof)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def nopull_rates_next(history: pd.DataFrame, home: str, away: str) -> tuple[float, float]:
    """No-pull expected goals for a game played after every row of `history`.

    `history` is fit_nhl_rules.load output for one season. The target is
    appended with a date that sorts after everything, and its walk-forward
    row is the prediction -- the graded code computes it, not this function.
    """
    cols = ["game_date", "home_team_abbr", "away_team_abbr", "nopull_h", "nopull_a"]
    target = pd.DataFrame([{"game_date": "9999-12-31#target",
                            "home_team_abbr": home, "away_team_abbr": away,
                            "nopull_h": 0, "nopull_a": 0}])
    frame = pd.concat([history[cols], target], ignore_index=True)
    frame = frame.rename(columns={"nopull_h": "home_score", "nopull_a": "away_score"})
    pred = fit_nhl_walkforward.walk_forward(frame, k=fit_nhl_rules.K)
    last = pred.iloc[-1]
    return float(last.lam_h), float(last.lam_a)


@dataclass
class NHLLiveSource:
    games: pd.DataFrame        # fit_nhl_rules.load output, with start times
    schedule: pd.DataFrame     # every regular-season game, indexed by id
    season: int
    _memo: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, season: int, raw: Path = RAW) -> "NHLLiveSource":
        raw = Path(raw)
        need = [raw / f"schedule_{season}.parquet", raw / f"nhl_{season}.parquet"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise MissingSeasonData(
                f"the {season} season needs {missing}. Pull them with "
                f"`python model/ingest/nhl_api.py --seasons {season}-{season} "
                "--force` and retry.")
        sched = pd.read_parquet(need[0])
        sched = sched.assign(game_id=sched.game_id.astype(str),
                             start=pd.to_datetime(sched.start_utc, utc=True))
        sched = sched.set_index("game_id")

        finals = pd.read_parquet(need[1])
        if finals.empty:
            games = pd.DataFrame(columns=["game_id", "game_date", "home_team_abbr",
                                          "away_team_abbr", "nopull_h", "nopull_a",
                                          "start"])
        else:
            goals_path = raw / f"goals_{season}.parquet"
            if not goals_path.exists():
                raise MissingSeasonData(
                    f"{goals_path} is missing; no-pull scores cannot be built. "
                    f"Run `python model/ingest/nhl_goals.py --seasons "
                    f"{season}-{season} --update`.")
            have = set(pd.read_parquet(goals_path, columns=["game_id"]).game_id)
            # A shootout decided 1-0 has no goal rows -- the deciding goal is
            # only in the final score. Every other final must have some.
            no_rows = finals[~finals.game_id.isin(have)
                             & ~((finals.last_period_type == "SO")
                                 & (finals.home_score + finals.away_score == 1))]
            if len(no_rows):
                raise StaleResults(
                    f"{len(no_rows)} final(s) have no goal rows (latest "
                    f"{no_rows.game_date.max()}); they would enter the no-pull "
                    "ratings as 0-0. Run nhl_goals.py --update.")
            games = fit_nhl_rules.load((season,), raw=raw)
            games = games.assign(game_id=games.game_id.astype(str))
            games = games.assign(start=games.game_id.map(sched.start))
            if games.start.isna().any():
                raise StaleResults("finals are missing from the schedule file; "
                                   "the two were not pulled together")
        return cls(games=games, schedule=sched, season=season)

    # -- FeatureSource --------------------------------------------------------

    def features(self, game_id: str, asof: str) -> GameFeatures:
        if game_id not in self.schedule.index:
            raise GameNotPriceable(f"{game_id!r} is not in the {self.season} schedule")
        g = self.schedule.loc[game_id]
        t = _ts(asof)
        if t > g.start:
            raise LookaheadRefused(
                f"{g.away_team_abbr}@{g.home_team_abbr} started at {g.start}; "
                f"pricing it as of {t} would use a moment after puck drop")
        if g.neutral_site:
            raise GameNotPriceable(
                f"{g.away_team_abbr}@{g.home_team_abbr} is at a neutral site; "
                "the home rate would go to a team that is not at home")
        self._refuse_stale(t)
        teams = [g.home_team_abbr, g.away_team_abbr]
        s = self.schedule
        live = s[(s.start < t) & ~s.game_state.isin(FINAL_STATES | POSTPONED_STATES)
                 & (s.home_team_abbr.isin(teams) | s.away_team_abbr.isin(teams))]
        if len(live):
            raise GameNotPriceable(
                f"{g.away_team_abbr}@{g.home_team_abbr}: a game involving one of "
                "these teams has started and is not final")

        past = self.games[self.games.start < t]
        key = (len(past), g.home_team_abbr, g.away_team_abbr)
        if key not in self._memo:
            self._memo[key] = nopull_rates_next(past, g.home_team_abbr, g.away_team_abbr)
        lh, la = self._memo[key]
        return GameFeatures(lam_home=math.nan, lam_away=math.nan,
                            lam_home_nopull=lh, lam_away_nopull=la)

    def _refuse_stale(self, t: pd.Timestamp) -> None:
        s = self.schedule
        behind = s[(s.start < t - STALE_AFTER)
                   & ~s.game_state.isin(FINAL_STATES | POSTPONED_STATES)]
        if len(behind):
            raise StaleResults(
                f"{len(behind)} game(s) started more than {STALE_AFTER} before "
                f"{t} and are not final in the {self.season} pull (latest "
                f"{behind.start.max()}). Re-pull before pricing.")

    # -- slate ----------------------------------------------------------------

    def slate(self, day: str) -> list[ModelGame]:
        s = self.schedule[~self.schedule.game_state.isin(FINAL_STATES | POSTPONED_STATES)]
        out = []
        for gid, g in s.sort_values("start").iterrows():
            if local_date(g.start.strftime("%Y-%m-%dT%H:%M:%SZ")) == day:
                out.append(ModelGame(game_id=str(gid), date=day,
                                     home=g.home_team_abbr, away=g.away_team_abbr))
        return out


def build_model(source: NHLLiveSource) -> NHLModel:
    """Always with the rules layer: the source produces no-pull rates only."""
    return NHLModel(source, rules=load_rules())
