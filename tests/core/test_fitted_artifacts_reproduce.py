"""A fitted artifact must be regenerable, not merely present.

WHY THIS EXISTS
data/nhl_fitted.json and data/nba_fitted.json shipped with no producer. The
ratings inside them existed only as loop locals in walk_forward, which never
returned them, so the files could be read and could not be rebuilt. That is
the repository's oldest wound exactly: eight production constants citing a
grid search whose output exists in no committed file.

Writing a generator afterwards is only half a fix, because a generator that
drifts from the file it claims to produce is worse than none -- it looks like
evidence. So this test re-runs the export and compares.

IT ALREADY EARNED ITS PLACE. The first version of the generator estimated the
NBA sigma on the HOLDOUT season instead of the tune seasons. Every one of the
30 ratings matched to five decimals, the holdout t matched at 5.96, and sigma
moved from 14.1666 to 12.9903 -- an 8% change in the number every NBA spread
price divides by, caused by fitting a parameter on the season being graded.
Nothing but this comparison would have shown it.

SLOW ON PURPOSE. It re-runs both walk-forward fits. That is the cost of the
claim being checkable.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ("nhl_fitted.json", "nba_fitted.json")


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> dict:
    """Build both artifacts in memory. Nothing on disk is touched."""
    from model import export_fitted

    return {
        "nhl_fitted.json": export_fitted.export_nhl(),
        "nba_fitted.json": export_fitted.export_nba(),
    }


@pytest.mark.parametrize("name", ARTIFACTS)
def test_committed_artifact_matches_a_fresh_export(name: str, regenerated) -> None:
    committed = json.loads((ROOT / "data" / name).read_text())
    fresh = regenerated[name]
    assert set(committed) == set(fresh), (
        f"data/{name} and its generator disagree about which keys exist"
    )
    for key in sorted(committed):
        assert committed[key] == fresh[key], (
            f"data/{name}[{key!r}] does not match a fresh export -- the "
            "artifact and the script that claims to produce it have drifted"
        )


@pytest.mark.parametrize("name", ARTIFACTS)
def test_artifact_is_registered_with_a_producer(name: str) -> None:
    """A fitted file with no producer in the manifest is how this started."""
    import yaml

    manifest = yaml.safe_load((ROOT / "artifacts.yml").read_text())
    rows = {a["path"]: a for a in manifest["artifacts"]}
    path = f"data/{name}"
    assert path in rows, f"{path} is not in artifacts.yml"
    producer = rows[path].get("produced_by")
    assert producer, f"{path} declares no producer"
    assert (ROOT / producer).exists(), f"{path} names a missing producer: {producer}"


def test_nba_sigma_is_not_estimated_on_the_graded_season() -> None:
    """The specific leak this file was written after.

    Asserted on the number rather than on the code, because the code can be
    rewritten and the property is what matters.
    """
    from model import fit_nba_walkforward as nbawf
    from model.fit_nba import load

    art = json.loads((ROOT / "data" / "nba_fitted.json").read_text())
    hp = art["hyperparameters"]
    tune = art["_provenance"]["tune_seasons"]
    hold = art["_provenance"]["holdout_season"]

    pred = nbawf.walk_forward(
        load(tuple(tune) + (hold,)), hp["k"], hp["home_adv"], hp["carryover"]
    )
    _, sd_tune, _ = nbawf.score(pred, tune)
    _, sd_hold, _ = nbawf.score(pred, [hold])

    assert art["sigma_constant"] == pytest.approx(sd_tune, abs=1e-3), (
        "sigma does not match the tune seasons"
    )
    assert art["sigma_constant"] != pytest.approx(sd_hold, abs=1e-3), (
        "sigma matches the HOLDOUT season's residual spread -- the graded "
        "season is being used to fit a parameter it then grades"
    )
