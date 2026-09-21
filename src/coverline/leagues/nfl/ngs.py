"""NGS team features, with the team-code normalisation the legacy path lacks.

THE ONE BEHAVIOURAL DIFFERENCE FROM THE LEGACY PATH
model/layer2_ngs.py returns the nflverse NGS release's own team codes, which
call the Rams `LAR`. The ratings snapshot and the schedule call them `LA`. The
legacy join is a membership test on that index, so every Rams game falls out
of the full ensemble and into the rating-only vector -- 17 games a season in
every season since 2022, losing Elo along with the NGS block, and presenting
as "NGS unavailable" (ADR 0004).

This module normalises. That makes it INTENTIONALLY different from the legacy
path for Rams games, which is a divergence the migration ledger records rather
than a bug: everywhere else the two agree exactly, and a test asserts that the
only difference is the one expected.

NORMALISATION IS EXPLICIT, NOT FUZZY
No string similarity, no "close enough" matching. A closed alias table, and an
unknown code raises. Fuzzy team matching is how a genuinely new franchise code
gets silently folded into an existing team, which is a worse failure than the
one being fixed.

POINT-IN-TIME
`through_week` filters strictly earlier weeks, matching the legacy semantics:
features for week W use weeks < W. The filter lives in the query, not in a
caller's discipline.
"""

from __future__ import annotations

import pandas as pd

from model.layer2_ngs import compute_team_ngs_features as _legacy_features

#: Canonical code -> the codes other sources use for the same team.
#: Canonical is whatever the ratings snapshot and schedule use, because that
#: is what everything else is keyed on.
#:
#: CLOSED TABLE. An unrecognised code raises rather than being guessed at.
TEAM_CODE_ALIASES: dict[str, str] = {
    "LAR": "LA",   # nflverse NGS -> ratings/schedule (ADR 0004)
}

#: The 32 codes the ratings snapshot and schedule use. Used to catch a source
#: that has started emitting something nobody has mapped.
CANONICAL_CODES: frozenset[str] = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
})


class UnknownTeamCode(KeyError):
    """A source emitted a code that is neither canonical nor a known alias."""


def canonicalise(code: str) -> str:
    """Map one source code to the canonical vocabulary. Raises on surprises."""
    mapped = TEAM_CODE_ALIASES.get(code, code)
    if mapped not in CANONICAL_CODES:
        raise UnknownTeamCode(
            f"{code!r} is neither a canonical team code nor a known alias. "
            "Add it to TEAM_CODE_ALIASES deliberately -- guessing is how a new "
            "franchise code gets silently folded into an existing team."
        )
    return mapped


def normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite a team-indexed frame into the canonical vocabulary."""
    out = frame.copy()
    out.index = [canonicalise(str(t)) for t in out.index]
    dupes = [t for t in set(out.index) if list(out.index).count(t) > 1]
    if dupes:
        raise UnknownTeamCode(
            f"normalisation collapsed two rows onto {dupes}. Two source codes "
            "map to one team, which means the alias table is wrong."
        )
    return out


def team_features(season: int, through_week: int | None = None,
                  min_teams: int = 28) -> pd.DataFrame:
    """Legacy NGS team features, re-keyed to the canonical vocabulary.

    `min_teams` is passed through and matters: a season's NGS release can be
    severely incomplete while loading without error, and the legacy module
    raises rather than letting a caller treat four teams as a full season.
    """
    raw = _legacy_features(season, through_week=through_week, min_teams=min_teams)
    return normalise_index(raw)


def ngs_present(features: pd.DataFrame, home: str, away: str) -> bool:
    """The legacy gate, against a normalised frame.

    Kept as a named function rather than an inline membership test so the
    thing that silently dropped 85 games has somewhere to be tested.
    """
    return home in features.index and away in features.index
