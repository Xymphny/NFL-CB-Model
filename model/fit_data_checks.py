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
