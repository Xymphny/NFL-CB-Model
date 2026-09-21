"""The measured facts behind the NHL puck line being withheld.

WHY THESE ARE PINNED
The first reason recorded for withholding the puck line was wrong. It said
hockey's two scores are negatively correlated at -0.14, so games stay closer
than independent rates allow and independent Poisson under-predicts one-goal
games. The sign was right and the mechanism was backwards: negative covariance
RAISES margin variance, because Var(H - A) = Var(H) + Var(A) - 2Cov(H, A), so
that story predicts FEWER one-goal games, not more. The measured dispersion
ratio is above one, not below.

A wrong mechanism points at a wrong repair -- here, fitting a correlation
parameter to a quantity that is drifting from -0.06 to -0.14 across nine
seasons. So the facts that replace it are asserted here rather than left in
prose, where the next person would have to re-derive them to trust them.

WHAT IS ACTUALLY HAPPENING is two league rules and neither is a correlation:
the overtime rule, which is deterministic, and empty-net goals, which are
conditional on the score and move mass across exactly the puck line.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REPORT = ROOT / "model" / "nhl_joint_structure.json"


@pytest.fixture(scope="module")
def report() -> dict:
    assert REPORT.exists(), (
        f"{REPORT.relative_to(ROOT)} is missing -- run model/measure_nhl_joint.py"
    )
    return json.loads(REPORT.read_text())


def test_the_overtime_rule_is_deterministic(report) -> None:
    """Not 'usually one goal'. Every one of them, with no exceptions.

    This is what makes it a rule to be applied rather than a distribution to
    be fitted, and it is the single largest deviation from independent
    Poisson: about 22% of the tie mass is relocated by fiat.
    """
    ot = report["overtime_rule"]
    assert ot["ties_in_final_scores"] == 0
    assert ot["ot_games_not_decided_by_exactly_one"] == 0
    assert ot["deterministic"] is True
    assert 0.18 < ot["ot_or_so_share"] < 0.27


def test_margins_are_over_dispersed_not_under(report) -> None:
    """The measurement that refutes the original story.

    'Games stay closer than independent rates allow' means the margin
    variance is BELOW the independent benchmark. It is above it, in both
    sources.
    """
    for source, d in report["pooled"].items():
        assert d["dispersion_ratio"] > 1.0, (
            f"{source}: dispersion ratio {d['dispersion_ratio']} -- if this "
            "ever drops below one the original 'games stay closer' story "
            "becomes arguable again and this test should be re-read, not "
            "deleted"
        )
        assert d["corr"] < 0, f"{source}: correlation is not negative"


def test_the_margin_distribution_is_non_monotone(report) -> None:
    """More three-goal games than two-goal games, in both sources.

    A count difference has no business doing this, and it happens across
    exactly the 1.5 line the puck line is priced on. Independent Poisson
    cannot produce it at any rates.
    """
    for source, shape in report["margin_shape"].items():
        two, three = shape["2"]["empirical"], shape["3"]["empirical"]
        assert three > two, f"{source}: P(|m|=3)={three} is not above P(|m|=2)={two}"
        assert shape["3"]["independent_poisson"] < shape["2"]["independent_poisson"], (
            "independent Poisson should be monotone here; if it is not, the "
            "benchmark is wrong and the finding above means nothing"
        )


def test_removing_empty_net_goals_restores_monotonicity(report) -> None:
    """The mechanism, not just the anomaly.

    If the bump at three is the empty-net layer, stripping those goals should
    make the distribution decrease again. It does. That is why the repair is
    a goalie-pull layer and not a correlation parameter.
    """
    en = report["empty_net"]
    if not en.get("margin_shape_with_empty_net_goals_removed"):
        pytest.skip("goal-level files absent; run model/ingest/nhl_goals.py")
    s = en["margin_shape_with_empty_net_goals_removed"]
    emp = [s[str(j)]["empirical"] for j in range(1, 5)]
    assert emp == sorted(emp, reverse=True), (
        f"stripping empty-net goals leaves a non-monotone shape {emp} -- the "
        "empty-net explanation is then incomplete and should be said so "
        "rather than patched"
    )


def test_empty_net_goals_are_conditional_on_a_one_or_two_goal_lead(report) -> None:
    """Why they land on the puck line specifically.

    A homogeneous scoring process spreads these across every game state. They
    are not spread: they concentrate where the 1.5 line lives.
    """
    en = report["empty_net"]
    by_lead = en.get("lead_of_scoring_team_before_empty_net_goal")
    if not by_lead:
        pytest.skip("goal-level files absent; run model/ingest/nhl_goals.py")
    one_or_two = by_lead.get("1", 0.0) + by_lead.get("2", 0.0)
    assert one_or_two > 0.85, (
        f"only {one_or_two:.1%} of empty-net goals come with a one or two "
        "goal lead -- the conditional story is weaker than claimed"
    )


def test_the_league_flag_and_the_situation_code_agree(report) -> None:
    """Two independent fields, one fact. Neither is trusted alone.

    They agree on every goal only after two corrections: counting
    awarded-empty-net and own-goal-empty-net as empty-net, and testing
    whether the net the SCORER SHOT AT was empty rather than whether either
    net was. Before those, 846 of 21,027 goals disagreed and 830 of them were
    the pulled team scoring into a guarded net.
    """
    en = report["empty_net"]
    if "flag_agrees_with_situation_code" not in en:
        pytest.skip("goal-level files absent; run model/ingest/nhl_goals.py")
    assert en["flag_agrees_with_situation_code"] > 0.999, (
        "the empty-net flag and the goalie-presence code no longer agree -- "
        "one of the two definitions has drifted and the layer should not be "
        "fitted until it is known which"
    )


def test_where_the_two_fields_disagree_it_is_the_code_that_is_wrong(report) -> None:
    """The cross-check found errors in the field it was built to trust.

    Thirteen goals in 56,834 disagree. Twelve are filed empty-net by the
    league while the situation code reads both goalies on the ice, between
    15:04 and 19:53 of the third period -- which is when empty-net goals
    happen and when goalies are not on the ice. So the flag is the field to
    use downstream, and that conclusion is asserted rather than assumed
    because it is the opposite of what a cross-check usually concludes.
    """
    d = report["empty_net"].get("flag_code_disagreements")
    if not d:
        pytest.skip("goal-level files absent; run model/ingest/nhl_goals.py")
    assert d["n"] < 0.001 * d["of"], "disagreements are no longer a handful"
    assert d["in_the_last_five_minutes_of_the_third"] >= d["n"] - 2, (
        "the disagreements are no longer concentrated late in the third, so "
        "the 'bad situation code' diagnosis no longer explains them"
    )


def test_the_correlation_is_drifting(report) -> None:
    """Why a fitted correlation would be the wrong object anyway.

    Early seasons and late seasons do not share a number. Fitting one pools
    across a league that changed; fitting the recent one ships a constant
    with no reason to persist.
    """
    api = {k: v for k, v in report["by_season"].items() if k.startswith("nhle-api:")}
    years = sorted(int(k.split(":")[1]) for k in api)
    # Three seasons a side, not one. A single season is noisy enough that the
    # first version of this test -- earliest against latest -- would have
    # failed on 2025 alone (-0.114) after 2023 and 2024 both sat at -0.142,
    # and reported stability that is not there.
    early = [api[f"nhle-api:{y}"]["corr"] for y in years[:3]]
    late = [api[f"nhle-api:{y}"]["corr"] for y in years[-3:]]
    mean_early = sum(early) / len(early)
    mean_late = sum(late) / len(late)
    assert mean_late < mean_early - 0.02, (
        f"correlation moved from {mean_early:.4f} to {mean_late:.4f}; if this "
        "ever stabilises, a fitted correlation becomes defensible and this "
        "test is the place to notice"
    )


def test_the_two_sources_agree_where_they_overlap(report) -> None:
    """The vendor's NHL files were corrupt for three seasons. This bounds it.

    2024 and 2025 exist in both the league API and sportsdataverse. If they
    agreed on nothing, neither could be used. They agree on every game, which
    is what makes the 2021-2023 corruption a bounded defect in specific files
    rather than a reason to distrust the vendor wholesale -- and is equally a
    check on the API pull, which nothing else here validates.
    """
    a = report.get("source_agreement")
    if not a or not a["games_joined"]:
        pytest.skip("only one source present for the overlapping seasons")
    assert a["score_mismatches"] == 0, (
        f"{a['score_mismatches']} of {a['games_joined']} games disagree "
        "between the league API and sportsdataverse -- neither source should "
        "be used until it is known which is wrong"
    )
