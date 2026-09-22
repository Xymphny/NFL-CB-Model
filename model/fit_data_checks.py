"""Refuse to fit degenerate data.

WHY THIS EXISTS
The first NHL fit ran on sportsdataverse's 2021-2023 schedule files, in which
EVERY game carries an identical score -- 1,312 rows all 6-3. The fit completed,
the grade completed, and it reported t = -5.12 with a straight face. Every
number downstream of it was meaningless, and nothing in the pipeline objected.

A constant column is the easiest data fault to detect and the easiest to miss,
because it breaks nothing: means exist, variances exist, models converge. It
only shows up if something looks.

WHAT IS CHECKED
Variety, range, and plausibility -- the three ways a score column goes wrong
without going missing. These are deliberately crude. A subtle data fault
deserves a subtle check; this catches the ones that would otherwise produce a
confident, publishable, meaningless result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DegenerateData(ValueError):
    """The data cannot support a fit. Raised rather than returned."""


@dataclass(frozen=True)
class Expectation:
    """What a plausible score column looks like for a sport."""

    min_distinct: int
    plausible_mean: tuple[float, float]
    plausible_max: tuple[int, int]


NHL = Expectation(min_distinct=6, plausible_mean=(2.0, 4.5), plausible_max=(6, 15))
NBA = Expectation(min_distinct=30, plausible_mean=(95.0, 130.0), plausible_max=(130, 200))
MLB = Expectation(min_distinct=8, plausible_mean=(3.0, 6.0), plausible_max=(10, 30))


class ImpossibleOutcome(ValueError):
    """The data contains results the league's own rules forbid."""


#: The share of tied final scores each league permits. A tie is the cheapest
#: impossibility to check and the one that keeps turning up.
#:
#: WHY THIS EXISTS. model/cfb_full_walk_forward_cache.csv carries 76 rows in
#: 1,731 -- 4.39% -- with a final margin of zero, and college football has not
#: permitted a tie since 1996. Every one of them is also recorded as a home
#: LOSS, so a tie became an away win. That cache produced MARGIN_SD and
#: DVOA_ONLY_MEAN_RESIDUAL and backed the CFB parity claim, and nothing
#: objected, because a tie breaks nothing: means exist, variances exist,
#: models converge.
#:
#: Two causes were found in the upstream schedule cache: 83 rows stored as
#: 0-0 where no score was ever fetched, and 59 rows frozen at a mid-game or
#: intermediate score -- Auburn 22-22 Alabama in 2021, a game Alabama won
#: 24-22 in four overtimes.
#:
#: The NFL is the one league that genuinely permits ties, at roughly 0.2% of
#: games, so its allowance is small and non-zero rather than absent.
MAX_TIE_RATE = {
    "nfl": 0.01,
    "cfb": 0.0,
    "nba": 0.0,
    "nhl": 0.0,
    "mlb": 0.0,
}


def check_no_impossible_ties(df: pd.DataFrame, league: str, *, label: str,
                             home: str = "home_score",
                             away: str = "away_score",
                             margin: str | None = None) -> dict:
    """Raise if a frame contains more tied results than the league allows.

    `margin` names a precomputed margin column, for caches that carry one
    instead of the two scores.
    """
    if league not in MAX_TIE_RATE:
        raise ValueError(f"no tie policy recorded for {league!r}")
    if len(df) == 0:
        raise DegenerateData(f"{label}: no rows at all")

    if margin is not None:
        tied = df[margin] == 0
    else:
        tied = df[home] == df[away]
    rate = float(tied.mean())
    allowed = MAX_TIE_RATE[league]
    out = {"label": label, "league": league, "n": int(len(df)),
           "tied": int(tied.sum()), "tie_rate": round(rate, 5),
           "allowed": allowed}
    if rate > allowed:
        raise ImpossibleOutcome(
            f"{label}: {int(tied.sum())} of {len(df)} rows ({rate:.2%}) are "
            f"tied, and {league} permits at most {allowed:.2%}. A tie breaks "
            "nothing downstream -- means exist, variances exist, models "
            "converge -- so this has to be refused here or it is not refused "
            "at all."
        )
    return out


def check_scores(df: pd.DataFrame, exp: Expectation, *, label: str,
                 home: str = "home_score", away: str = "away_score") -> dict:
    """Raise if this frame cannot support a fit. Returns the diagnostics."""
    if len(df) == 0:
        raise DegenerateData(f"{label}: no rows at all")

    out: dict[str, object] = {"n": int(len(df))}
    for side, col in (("home", home), ("away", away)):
        s = df[col].dropna()
        distinct = int(s.nunique())
        mean = float(s.mean())
        mx = int(s.max())
        out[side] = {"distinct": distinct, "mean": round(mean, 4),
                     "max": mx, "variance": round(float(s.var()), 4)}

        if distinct < exp.min_distinct:
            raise DegenerateData(
                f"{label}: {col} has only {distinct} distinct value(s) across "
                f"{len(df)} rows. A constant or near-constant score column "
                "fits, grades, and means nothing -- this is what the "
                "2021-2023 NHL files did."
            )
        lo, hi = exp.plausible_mean
        if not lo <= mean <= hi:
            raise DegenerateData(
                f"{label}: {col} mean is {mean:.3f}, outside the plausible "
                f"[{lo}, {hi}] for this sport. The column is probably not "
                "what it is labelled."
            )
        mlo, mhi = exp.plausible_max
        if not mlo <= mx <= mhi:
            raise DegenerateData(
                f"{label}: {col} max is {mx}, outside [{mlo}, {mhi}]. Either "
                "the column is wrong or the filter let something through."
            )

    return out


def usable_seasons(loader, seasons, exp: Expectation, *, label: str) -> list[int]:
    """Which seasons pass. Reports the failures rather than dropping silently."""
    good = []
    for y in seasons:
        try:
            check_scores(loader(y), exp, label=f"{label} {y}")
            good.append(y)
        except DegenerateData as exc:
            print(f"  [data] {label} {y} REJECTED: {str(exc)[:120]}")
    return good
