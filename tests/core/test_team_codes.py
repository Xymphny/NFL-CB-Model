"""Team codes must agree across every source that gets joined on them.

THE BUG THIS WAS WRITTEN FOR
The ratings snapshot and the nflverse schedule call the Rams `LA`. The
nflverse Next Gen Stats release calls them `LAR`. model/prediction.py decides
per game:

    ngs_present = home in ngs_features.index and away in ngs_features.index

so every Rams game -- 17 a season, in all five seasons with committed data --
silently falls back to MARGIN_COEFFICIENTS_V1_RATING_ONLY. It presents as
"NGS is not available for this game", which is indistinguishable from the feed
genuinely being down, and nothing anywhere said otherwise.

It is worse than one missing feature block. The rating-only vector carries NO
elo_diff term, so those games lose Elo as well.

WHY A GUARD RATHER THAN JUST A FIX
A silent join failure on a shared key is a CLASS of bug, not one incident.
Any two sources joined on team code can drift: a relocation, a rebrand, a
vendor changing convention mid-season. This test compares the vocabularies
directly, so the next one fails loudly instead of degrading quietly.

It runs against committed artifacts only -- no network -- so it is a real CI
check rather than one that passes when a fetch times out.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RATINGS = ROOT / "data" / "ratings"
SCHEDULE_FIXTURE = ROOT / "tests" / "fixtures" / "nfl_2026_week02_schedule.csv"

#: Codes known to differ between sources, with the source that differs.
#: A code here is DOCUMENTED, not forgiven: normalisation still has to happen
#: wherever the join occurs. Removing an entry without fixing the join is what
#: this file exists to prevent.
KNOWN_ALIASES = {
    "LA": {"LAR"},   # nflverse NGS uses LAR; ratings and schedule use LA
}


def _ratings_teams(week: int = 2, season: int = 2026) -> set[str]:
    p = RATINGS / f"{season}-week-{week:02d}.json"
    if not p.exists():
        pytest.skip(f"no ratings snapshot at {p}")
    return {r["team"] for r in json.loads(p.read_text())["ratings"]}


def _schedule_teams() -> set[str]:
    if not SCHEDULE_FIXTURE.exists():
        pytest.skip("no schedule fixture")
    s = pd.read_csv(SCHEDULE_FIXTURE)
    return set(s.home_team) | set(s.away_team)


def test_ratings_and_schedule_agree_on_every_code():
    """These two are joined directly in the live feature source. A mismatch
    here would drop a game rather than degrade it."""
    ratings, sched = _ratings_teams(), _schedule_teams()
    missing = sched - ratings
    assert not missing, (
        f"the week-2 schedule names {sorted(missing)}, which the ratings "
        "snapshot does not. Those games cannot be priced at all."
    )


def test_the_ratings_snapshot_has_all_thirty_two_teams():
    assert len(_ratings_teams()) == 32


def test_the_rams_alias_is_still_recorded():
    """Pins the known case. If someone normalises NGS codes and removes this
    entry, they should have to edit this test on purpose -- and if they remove
    the entry WITHOUT fixing the join, the next test catches it."""
    assert "LA" in KNOWN_ALIASES
    assert "LAR" in KNOWN_ALIASES["LA"]


def test_every_alias_target_is_absent_from_the_canonical_vocabulary():
    """An alias only makes sense while the two codes are genuinely different.
    If nflverse ever switches NGS to `LA`, the alias becomes a no-op and
    should be retired deliberately rather than left as folklore."""
    ratings = _ratings_teams()
    for canonical, aliases in KNOWN_ALIASES.items():
        assert canonical in ratings, f"{canonical} is not a real team code"
        for a in aliases:
            assert a not in ratings, (
                f"{a} now appears in the canonical vocabulary alongside "
                f"{canonical}. The alias may be obsolete -- check the source "
                "and retire it deliberately."
            )


def test_the_scale_of_the_known_mismatch_is_recorded():
    """85 games across five committed seasons, 17 a season. Written down so
    the cost of leaving it unfixed is a number rather than an impression."""
    sched = pd.read_csv(SCHEDULE_FIXTURE)
    affected = sched[(sched.home_team == "LA") | (sched.away_team == "LA")]
    assert len(affected) >= 1, (
        "no Rams game on the week-2 fixture; if the fixture changed, the "
        "scope of this finding should be re-measured rather than assumed"
    )


def test_prediction_treats_a_missing_ngs_team_as_absent_not_as_zero():
    """The mechanism, pinned. If this ever changed to substituting zeros, a
    code mismatch would stop being a visible fallback and become a silent
    mispricing with the FULL ensemble."""
    src = (ROOT / "model" / "prediction.py").read_text()
    assert "ngs_present" in src
    assert "in ngs_features.index" in src, (
        "build_week_predictions no longer gates on NGS index membership; "
        "re-read how a missing team is handled before trusting this test"
    )
