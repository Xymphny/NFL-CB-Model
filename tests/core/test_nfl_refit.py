"""The 2016-2022 refit (accuracy plan 2a, ADR 0031): recorded, not live.

The refit passed the plan's ship rule and was not applied -- swapping live
coefficients needs its own authorised change. These pin that state so it
cannot drift silently in either direction: the artifact keeps its window and
result, and the live vectors stay the legacy ones until the ADR is accepted.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.leagues.nfl import model as nfl  # noqa: E402

ART = ROOT / "data" / "nfl_refit_2016_2022.json"
ADR = ROOT / "docs" / "decisions" / "0031-nfl-refit-with-a-recorded-window.md"


def _art():
    return json.loads(ART.read_text())


def test_the_fit_window_and_the_held_out_seasons_are_recorded():
    a = _art()
    assert a["fit_window"]["seasons"] == [2016, 2022]
    assert a["held_out"]["seasons"] == [2023, 2024, 2025]
    assert not set(range(2016, 2023)) & set(a["held_out"]["seasons"])


def test_neutral_sites_were_in_the_fit_so_home_field_is_identified():
    for v in _art()["refit"].values():
        assert v["neutral_games"] > 0


def test_both_vectors_were_graded_on_their_own():
    a = _art()
    assert "passes_ship_rule" in a["primary"]
    assert "passes_ship_rule" in a["rating_only_vector"]


def test_live_vectors_are_unchanged_while_the_adr_is_only_proposed():
    status = re.search(r"^status:\s*(\w+)", ADR.read_text(), re.M).group(1)
    if status != "proposed":
        return          # accepted or rejected: this guard's job is done
    a = _art()
    assert dict(nfl.MARGIN_COEFFICIENTS) == a["current"]["full_ensemble"]
    assert dict(nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY) == a["current"]["rating_only"]
