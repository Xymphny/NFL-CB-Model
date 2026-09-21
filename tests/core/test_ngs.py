"""Team-code normalisation, and the measurement that decided not to ship it.

The normalisation itself is small. What needs testing is that it is EXPLICIT:
a closed alias table that raises on anything it has not been told about, not
fuzzy matching. Silently folding a genuinely new franchise code into an
existing team would be a worse version of the bug being fixed.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.leagues.nfl import ngs as N  # noqa: E402

RESULTS = ROOT / "model" / "ngs_team_code_fix_results.json"


def _frame(codes):
    return pd.DataFrame({"team_cpoe": range(len(codes))}, index=list(codes))


# --------------------------------------------------------- normalisation ----

def test_the_known_alias_maps():
    assert N.canonicalise("LAR") == "LA"


def test_a_canonical_code_passes_through():
    for t in ("KC", "NYG", "LA", "WAS"):
        assert N.canonicalise(t) == t


def test_an_unknown_code_raises_rather_than_being_guessed():
    """Fuzzy matching is how a new franchise gets folded into an old one."""
    with pytest.raises(N.UnknownTeamCode, match="deliberately"):
        N.canonicalise("STL")
    with pytest.raises(N.UnknownTeamCode):
        N.canonicalise("LAX")


def test_normalising_rewrites_the_index():
    out = N.normalise_index(_frame(["KC", "LAR", "NYG"]))
    assert set(out.index) == {"KC", "LA", "NYG"}
    assert out.loc["LA", "team_cpoe"] == 1


def test_an_alias_table_that_collapses_two_teams_is_refused():
    """If a source ever emitted both LA and LAR, mapping them together would
    silently average two teams into one row."""
    with pytest.raises(N.UnknownTeamCode, match="collapsed"):
        N.normalise_index(_frame(["LA", "LAR"]))


def test_the_canonical_vocabulary_is_the_full_league():
    assert len(N.CANONICAL_CODES) == 32
    assert "LA" in N.CANONICAL_CODES and "LAR" not in N.CANONICAL_CODES


def test_the_presence_gate_is_a_named_function():
    """The thing that silently dropped 85 games needs somewhere to be tested."""
    f = _frame(["KC", "LA"])
    assert N.ngs_present(f, "KC", "LA") is True
    assert N.ngs_present(f, "KC", "NYG") is False


# ------------------------------------------------------- the measurement ----

def test_the_fix_was_measured_before_being_considered():
    if not RESULTS.exists():
        pytest.skip("measurement not run yet")
    art = json.loads(RESULTS.read_text())
    assert art["_provenance"]["graded_once"] is True
    assert art["_provenance"]["holdout_seasons"] == [2022, 2023]
    assert art["n_games"] >= 20


def test_the_measurement_used_real_elo_on_both_sides():
    """The first version held elo at 0.0, which compared the rating-only
    vector against a full ensemble missing one of its own features and got a
    dramatic, wrong answer (t = -1.68 instead of +0.12)."""
    if not RESULTS.exists():
        pytest.skip("measurement not run yet")
    prov = json.loads(RESULTS.read_text())["_provenance"]
    assert prov["elo_included"] is True
    assert "missing one of" in prov["elo_note"]


def test_the_fix_is_not_supported_by_the_measurement():
    """Pinned so nobody later ships it as an improvement. It may still be
    worth shipping for consistency -- that is a different argument, and this
    test exists so the two do not get conflated."""
    if not RESULTS.exists():
        pytest.skip("measurement not run yet")
    art = json.loads(RESULTS.read_text())
    assert art["paired_gain"]["supported"] is False
    assert abs(art["paired_gain"]["t"]) < 2.0


def test_the_legacy_path_is_still_unfixed_and_that_is_recorded():
    """The new core normalises; model/ has not been touched. If someone
    patches the legacy, this test should fail and make them update ADR 0004
    rather than leaving it describing a bug that no longer exists."""
    legacy = (ROOT / "model" / "layer2_ngs.py").read_text()
    assert "LAR" not in legacy, (
        "model/layer2_ngs.py now mentions LAR, so the legacy path may have "
        "been normalised. Update ADR 0004 and the ledger row deliberately."
    )
