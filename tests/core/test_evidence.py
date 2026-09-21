"""The attempt log, and the guards that keep it honest.

The log's whole value is that it contains failures. Every test here is aimed at
one of the ways that property could quietly stop being true: attempts dropped,
estimands mixed, numbers transcribed wrong, or the weight computed from a
filtered view.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import evidence as E  # noqa: E402
from coverline.core.staking import shrinkage_weight, size_bet  # noqa: E402

LOG = ROOT / "evidence" / "attempts.yaml"


def test_the_log_exists_and_loads():
    log = E.load_attempts()
    assert len(log) >= 5, "the recovered log lost attempts"


def test_the_weight_is_what_the_arithmetic_says():
    log = E.load_attempts()
    assert log.mean_t_squared == pytest.approx(10.2483, abs=1e-3)
    assert log.weight() == pytest.approx(0.9024, abs=1e-3)
    assert E.current_weight() == pytest.approx(log.weight())


def test_the_weight_agrees_with_the_staking_module():
    """Two implementations of b = 1 - 1/E[t^2] must not drift apart."""
    log = E.load_attempts()
    assert log.weight() == pytest.approx(shrinkage_weight(log.t_statistics))


def test_the_first_prospective_attempt_is_logged():
    """The log stops being pure archaeology the moment something is recorded
    as it happens. n_recovered < len() is the marker."""
    log = E.load_attempts()
    assert log.n_recovered < len(log), "still entirely reconstructed"
    prospective = [a.id for a in log.attempts if not a.recovered]
    assert "nfl-key-number-weights" in prospective


def test_the_log_contains_failures():
    """If this ever reads zero, suspect the log before believing the pipeline."""
    log = E.load_attempts()
    assert log.n_negative >= 2, (
        "no non-positive attempts in the log. Either nothing has failed since "
        "it was written -- implausible -- or failures are being dropped, which "
        "makes every weight computed from it an overestimate."
    )


def test_dropping_failures_inflates_the_weight_measurably():
    """Quantifies the selection effect on this project's own data, so the cost
    of a filtered log is a number rather than a warning."""
    log = E.load_attempts()
    cherry = E.drop_negative_for_demonstration(log)
    assert cherry.weight() > log.weight()


def test_the_weight_is_currently_dominated_by_one_attempt():
    """Six observations, and one of them moves the answer by 30 points.

    Not a failure -- the key-number effect is real and large. But a pooled
    weight resting on a single experiment is not something to stake real money
    on without knowing, so the log says so out loud.
    """
    log = E.load_attempts()
    assert log.is_dominated_by_one is True
    assert "DOMINATED BY ONE ATTEMPT" in log.summary()
    loo = log.leave_one_out_weights()
    assert loo["nfl-key-number-weights"] == pytest.approx(0.5997, abs=1e-3)
    others = [w for k, w in loo.items() if k != "nfl-key-number-weights"]
    assert all(w > 0.90 for w in others), (
        "dropping any other single attempt should barely move the weight; if "
        "that stops being true the dominance has shifted and this test should "
        "be re-read rather than re-pinned"
    )


def test_the_robust_weight_survives_losing_the_dominant_attempt():
    log = E.load_attempts()
    assert log.robust_weight() == pytest.approx(0.5997, abs=1e-3)
    assert log.robust_weight() < log.weight()
    assert log.robust_weight() == min(log.leave_one_out_weights().values())


def test_robust_and_pooled_converge_when_no_row_dominates(tmp_path):
    """The diagnostic must go quiet on a healthy log, or it is just noise."""
    rows = [_row(id=f"a{i}", estimate=0.02 + 0.001 * i,
                 standard_error=0.01, t=2.0 + 0.1 * i) for i in range(8)]
    p = _write(tmp_path, rows)
    log = E.load_attempts(p)
    assert log.is_dominated_by_one is False
    assert log.robust_weight() == pytest.approx(log.weight(), abs=0.05)


def test_the_weight_is_flagged_as_a_ceiling_while_recovered():
    """A reconstructed log is missing whatever was never written up, and what
    goes unwritten skews toward nulls."""
    log = E.load_attempts()
    assert log.is_ceiling is True
    assert log.n_recovered > 0, (
        "no recovered attempts remain -- the log is fully prospective, so the "
        "ceiling framing no longer applies and is_ceiling should be False"
    )
    assert "CEILING" in log.summary()


def test_every_recorded_t_matches_its_own_estimate_and_error():
    """A transcription error in the log silently moves the weight."""
    for a in E.load_attempts().attempts:
        assert a.t == pytest.approx(a.t_implied, abs=0.02), (
            f"{a.id}: recorded t={a.t}, estimate/SE implies {a.t_implied:.4f}"
        )


#: Train-side coefficient t-statistics recorded in this repo, kept out of the
#: weight on purpose. Listed here so the exclusion can be measured.
TRAIN_SIDE = [8.51, 4.14, 2.63, 2.36, 2.13, 1.43, -0.89, -0.38, -0.35]


def test_train_side_statistics_would_have_inflated_the_recovered_weight():
    """The exclusion measured against the population the rule was written for.

    NOTE ON WHY THIS IS SCOPED TO THE RECOVERED ATTEMPTS. When the log held
    only the five recovered rows, admitting train-side statistics moved the
    weight from 0.60 to above 0.75 -- a large distortion, and the reason the
    rule exists. Adding nfl-key-number-weights at t=7.00 raised the log's own
    E[t^2] so far that the same pollution now moves the pooled weight by about
    0.01. That is not evidence the rule stopped mattering; it is evidence that
    one large attempt currently swamps everything, which is exactly what
    is_dominated_by_one reports. Measuring on the recovered subset keeps the
    demonstration honest instead of quietly weakening the assertion.
    """
    recovered = [a.t for a in E.load_attempts().attempts if a.recovered]
    assert len(recovered) == 5
    clean = shrinkage_weight(recovered)
    polluted = shrinkage_weight(recovered + TRAIN_SIDE)
    assert clean == pytest.approx(0.5997, abs=1e-3)
    assert polluted > clean + 0.15, (
        "admitting train-side t-statistics should visibly inflate the weight"
    )
    assert polluted > 0.75


def test_the_dominant_attempt_currently_masks_that_distortion():
    """Pinned so nobody later reads the small pooled difference as a licence
    to admit train-side statistics."""
    log = E.load_attempts()
    pooled_polluted = shrinkage_weight(list(log.t_statistics) + TRAIN_SIDE)
    assert pooled_polluted - log.weight() < 0.05
    assert log.is_dominated_by_one is True


def test_the_qb_continuity_row_is_the_reason_for_the_rule():
    """One hypothesis, +2.63 on train and -0.17 held out, in the same week."""
    doc = yaml.safe_load(LOG.read_text())
    qb = next(a for a in doc["attempts"] if a["id"] == "qb-continuity-conditional-prior")
    assert qb["t"] == pytest.approx(-0.17)
    assert qb["decision"] == "rejected"
    excluded = {e["id"]: e for e in doc["excluded"]}
    assert 2.63 in excluded["qb-continuity-train-interactions"]["t"]


def test_the_contradicted_half_life_stays_in_the_log():
    """A shipped value whose held-out grade came back negative is an attempt
    like any other, and removing it would raise the weight."""
    log = E.load_attempts()
    hl = next(a for a in log.attempts if a.id == "half-life-100-vs-default-6")
    assert hl.t < 0
    assert hl.decision == "shipped_but_contradicted"


# -------------------------------------------------------------- validation ----

def _write(tmp_path, attempts):
    p = tmp_path / "attempts.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "attempts": attempts, "excluded": []}))
    return p


def _row(**over):
    row = dict(
        id="x", date="2026-01-01", hypothesis="h", estimand="held_out_paired_gain",
        metric="mae", estimate=0.02, standard_error=0.01, t=2.0,
        decision="shipped", recovered=False, source="s",
    )
    row.update(over)
    return row


def test_a_mixed_estimand_is_refused(tmp_path):
    p = _write(tmp_path, [_row(estimand="train_interaction")])
    with pytest.raises(ValueError, match="only 'held_out_paired_gain'"):
        E.load_attempts(p)


def test_a_mismatched_t_is_refused(tmp_path):
    p = _write(tmp_path, [_row(t=5.0)])  # estimate/SE implies 2.0
    with pytest.raises(ValueError, match="implies"):
        E.load_attempts(p)


def test_a_missing_field_is_refused(tmp_path):
    bad = _row(); bad.pop("recovered")
    p = _write(tmp_path, [bad])
    with pytest.raises(ValueError, match="missing"):
        E.load_attempts(p)


def test_duplicate_ids_are_refused(tmp_path):
    p = _write(tmp_path, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate attempt ids"):
        E.load_attempts(p)


def test_an_unknown_decision_is_refused(tmp_path):
    p = _write(tmp_path, [_row(decision="probably_fine")])
    with pytest.raises(ValueError, match="decision"):
        E.load_attempts(p)


def test_an_empty_log_refuses_to_produce_a_weight(tmp_path):
    p = _write(tmp_path, [])
    with pytest.raises(ValueError, match="empty attempt log"):
        E.load_attempts(p).weight()


def test_a_missing_log_refuses_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no defensible"):
        E.load_attempts(tmp_path / "nope.yaml")


def test_a_noise_only_log_returns_zero(tmp_path):
    """RMS t near 1 is what pure noise produces. The answer must be 'ship
    nothing', not a small positive weight."""
    rows = [_row(id=f"a{i}", estimate=0.01, standard_error=0.01, t=1.0)
            for i in range(6)]
    p = _write(tmp_path, rows)
    assert E.load_attempts(p).weight() == 0.0


# ------------------------------------------------------------ end to end ----

def test_the_weight_flows_into_a_real_stake():
    """The blocker this module exists to clear: sizing can now be done from a
    derived weight instead of a judgement call."""
    plan = size_bet(
        p_model=0.56, p_market=0.52, decimal_price=1.9091,
        bankroll=100_000, shrinkage=E.current_weight(),
    )
    assert plan.placed is True
    assert plan.p_used == pytest.approx(0.52 + E.current_weight() * 0.04, abs=1e-3)
    assert plan.edge_claimed > plan.edge_used > 0.0


def test_a_ceiling_weight_still_produces_a_smaller_stake_than_full_confidence():
    derived = size_bet(p_model=0.56, p_market=0.52, decimal_price=1.9091,
                       bankroll=100_000, shrinkage=E.current_weight())
    unshrunk = size_bet(p_model=0.56, p_market=0.52, decimal_price=1.9091,
                        bankroll=100_000, shrinkage=1.0)
    assert derived.stake < unshrunk.stake
