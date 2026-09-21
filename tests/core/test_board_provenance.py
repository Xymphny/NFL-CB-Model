"""A published margin must be re-derivable from what the board recorded.

WHY THIS EXISTS
The board recorded which feature blocks were used and which coefficient
vector was selected, but not the feature VALUES. That is enough to describe a
prediction and not enough to check one.

Found by trying: reconstructing the 2026 week 2 full-ensemble games from
committed artifacts alone missed by -2.3 to +3.8 points, and the Elo
difference implied to close the gap even flipped sign on one game -- so the
error could not be attributed to any single input. NGS releases and Elo state
both move between runs, so a later reader cannot tell a model change from a
data change. For a project whose stated pillar is that every claim is graded
publicly, a claim whose inputs are unrecorded is not really gradeable.

`feature_values` closes that. These tests hold it to the only standard that
matters: the recorded values, run through the recorded coefficient vector,
must reproduce the published number.

Boards written BEFORE this change have no feature_values and are skipped by
name rather than silently passing -- an old artifact cannot retroactively
acquire provenance, and pretending otherwise would make this suite green for
the wrong reason.
"""

import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coverline.leagues.nfl.model import (  # noqa: E402
    MARGIN_COEFFICIENTS, MARGIN_COEFFICIENTS_V1_RATING_ONLY, GameFeatures,
    predict_margin,
)

REQUIRED_FIELDS = {"rating_diff", "rest_diff", "cpoe_diff", "separation_diff",
                   "yac_oe_diff", "ryoe_diff", "elo_diff", "is_neutral_site"}


def _boards():
    return sorted(glob.glob(str(ROOT / "data" / "divergence" / "*.json")))


def _rows_with_provenance():
    out = []
    for p in _boards():
        b = json.loads(Path(p).read_text())
        for d in b.get("divergences", []):
            if d.get("feature_values") and d.get("spread_gap") is not None:
                out.append((Path(p).name, b, d))
    return out


# ------------------------------------------------- the producing code ----

def test_the_prediction_layer_emits_feature_values():
    """Guards the source, so this suite cannot go quiet just because no new
    board has been published yet."""
    src = (ROOT / "model" / "prediction.py").read_text()
    assert '"feature_values"' in src or "'feature_values'" in src
    for f in ("rating_diff", "elo_diff", "separation_diff"):
        assert f in src


def test_the_board_records_what_the_prediction_layer_emits():
    src = (ROOT / "deploy" / "odds_watch_job.py").read_text()
    assert 'pred.get("feature_values")' in src, (
        "the board no longer records feature_values; a published margin stops "
        "being re-derivable the moment this line goes"
    )


def test_recording_is_additive_and_touches_no_published_number():
    """feature_values must be written alongside the prediction, never used to
    compute it. If it ever feeds back in, provenance becomes part of the
    model."""
    src = (ROOT / "model" / "prediction.py").read_text()
    idx = src.index('result["feature_values"]')
    after = src[idx:]
    assert "predict_game(" not in after.split("return predictions")[0], (
        "feature_values is assigned before a prediction is computed; it must "
        "only ever describe one"
    )


# ------------------------------------------------- published artifacts ----

def test_older_boards_are_skipped_by_name_not_silently():
    """An artifact written before this change cannot acquire provenance
    retroactively. Counting them makes the coverage visible."""
    total = sum(len(json.loads(Path(p).read_text()).get("divergences", []))
                for p in _boards())
    with_prov = len(_rows_with_provenance())
    print(f"\n  {with_prov} of {total} published rows carry feature_values")
    assert total > 0, "no board artifacts at all"


def test_every_recorded_row_has_the_full_field_set():
    rows = _rows_with_provenance()
    if not rows:
        pytest.skip("no board has been published since feature_values landed")
    for name, _, d in rows:
        missing = REQUIRED_FIELDS - set(d["feature_values"])
        assert not missing, f"{name} {d['home_team']}/{d['away_team']}: {sorted(missing)}"


def test_recorded_values_reproduce_the_published_margin():
    """The standard that makes provenance worth recording.

    model_margin = market_spread + spread_gap (generate_performance.py), and
    the board adds a slate de-bias offset on top of the linear model.
    """
    rows = _rows_with_provenance()
    if not rows:
        pytest.skip("no board has been published since feature_values landed")

    for name, board, d in rows:
        fv = d["feature_values"]
        ngs = d.get("coefficient_set") == "full_ensemble"
        f = GameFeatures(
            rating_diff=fv["rating_diff"], rest_diff=fv["rest_diff"],
            cpoe_diff=fv["cpoe_diff"], separation_diff=fv["separation_diff"],
            yac_oe_diff=fv["yac_oe_diff"], ryoe_diff=fv["ryoe_diff"],
            elo_diff=fv["elo_diff"] or 0.0,
            is_neutral_site=fv["is_neutral_site"], ngs_present=ngs,
        )
        published = d["market_spread"] + d["spread_gap"]
        offset = board["debias_offsets"][0]
        assert predict_margin(f) + offset == pytest.approx(published, abs=1e-6), (
            f"{name} {d['home_team']}/{d['away_team']}: recorded features do "
            "not reproduce the published margin"
        )
