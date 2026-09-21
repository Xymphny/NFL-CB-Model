"""Calibration of the three leagues that were never graded on it.

WHAT THIS IS FOR
NHL and NBA were graded on the likelihood of realised scorelines. NFL, CFB and
MLB had parity with the legacy pipeline -- which says the rewrite reproduces
the old numbers and nothing about whether the old numbers were well specified
-- and, for NFL, an ATS record, which asks whether the model beats a book.

Neither asks whether the DISTRIBUTION is the right shape, and that is what
every stake divides by. A model can be honestly negative against the market
and still be overconfident, and the overconfidence is invisible in an ATS
record while being exactly what blows up a bankroll through Kelly.

THE TEST IS ONE-SIDED AND SAYS SO
Ratings in these caches are walk-forward; coefficients and dispersion
constants were fitted on the same games. A distribution that is miscalibrated
in sample is miscalibrated. One that looks fine here has proved nothing.
`split_check` is the part that is not circular, and it is what the
CFB assertions below rest on.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GRADES = ROOT / "model" / "distribution_grades.json"


@pytest.fixture(scope="module")
def grades() -> dict:
    assert GRADES.exists(), (
        f"{GRADES.relative_to(ROOT)} is missing -- run "
        "model/grade_distributions.py"
    )
    return json.loads(GRADES.read_text())


def test_the_cfb_dispersion_ratio_is_labelled_circular(grades) -> None:
    """Because it is, and a 1.0000 reads like the best number on the page.

    CFB's MARGIN_SD was measured as the residual sd of the exact cache it is
    then judged against. The ratio is 1.0000 by construction. Without the
    label it sits beside NFL's 1.0337 and looks like the better-calibrated
    league.
    """
    cfb = grades["cfb"]
    assert cfb["dispersion_ratio"] == pytest.approx(1.0, abs=1e-4)
    assert "circular" in " ".join(cfb).lower() or (
        "dispersion_ratio_is_circular" in cfb
    ), "the tautology is unlabelled"


def test_cfb_dispersion_holds_up_when_the_circle_is_broken(grades) -> None:
    """Estimated on 2021-2022, judged on 2023, which it never saw.

    This is the assertion that makes CFB's constant defensible rather than
    merely self-consistent.
    """
    sc = grades["cfb"]["split_check"]
    assert sc["available"], "the split check did not run"
    assert 0.9 < sc["dispersion_ratio"] < 1.1, (
        f"out-of-sample dispersion ratio {sc['dispersion_ratio']} -- CFB's "
        "MARGIN_SD no longer describes seasons it was not measured on"
    )
    assert sc["coverage"]["95"] == pytest.approx(0.95, abs=0.03)


def test_cfb_carries_a_stable_positive_bias(grades) -> None:
    """The known +2.16, now shown to be systematic rather than a season.

    It is recorded in the model as DVOA_ONLY_MEAN_RESIDUAL and deliberately
    NOT corrected, because subtracting a constant from a vector-fitted model
    is a model change and goes through the gate like anything else. What is
    new here is that every season carries it, so it is not noise.
    """
    cfb = grades["cfb"]
    assert cfb["mean_residual"] > 1.5
    per = cfb["by_season"]
    assert all(v["mean_residual"] > 1.5 for v in per.values()), (
        "the bias is no longer present in every season, which would make it "
        "a period effect rather than a model defect"
    )
    assert grades["cfb"]["homogeneity"]["bias_varies_by_season"]["p"] > 0.05


def test_nfl_dispersion_is_not_constant_across_seasons(grades) -> None:
    """Bartlett p = 0.014 over ten seasons, sd ranging 11.73 to 15.15.

    SUGGESTIVE, NOT ESTABLISHED. It is one test chosen after looking at the
    season table, which is a garden of forking paths, and ten seasons is not
    many. It is asserted anyway because the consequence is concrete: a single
    MARGIN_SD cannot be right in every season, so every NFL probability is a
    little wrong in a direction that changes year to year, and Kelly divides
    by exactly that number.
    """
    h = grades["nfl"]["homogeneity"]["dispersion_varies_by_season"]
    assert h["p"] < 0.05, (
        f"NFL residual dispersion now looks constant across seasons "
        f"(p = {h['p']}). If that holds up, a single MARGIN_SD becomes "
        "defensible and this test should be re-read, not deleted."
    )
    lo, hi = h["sd_range"]
    assert hi - lo > 2 * h["sampling_se_of_one_seasons_sd"], (
        "the spread is no longer wider than sampling noise would give"
    )


def test_nfl_bias_does_not_vary_by_season(grades) -> None:
    """The other half, and it points the other way.

    The spread moves; the bias does not. NFL season mean residuals run from
    -1.90 to +1.55 and are consistent with noise at p = 0.24. So there is no
    systematic NFL bias to correct -- unlike CFB, where there is.
    """
    h = grades["nfl"]["homogeneity"]["bias_varies_by_season"]
    assert h["p"] > 0.05
    assert abs(h["pooled_mean_residual"]) < 1.0


def test_every_graded_league_beats_a_league_average_baseline(grades) -> None:
    """The same metric NHL and NBA were graded on, so five leagues read alike.

    A pass here is weak evidence -- it is in-sample for the coefficients. A
    failure would be decisive, which is the only reason to run it.
    """
    for lg in ("nfl", "cfb", "mlb"):
        assert grades[lg]["t"] > 2, (
            f"{lg} no longer beats predicting the league average, IN SAMPLE. "
            "That is not a close call and the model should be withheld."
        )


def test_tail_coverage_is_reported_for_every_league(grades) -> None:
    """A ratio can hide a shape problem; coverage cannot.

    NFL's pooled 95% interval covers 93.2%, so its tails are thinner than a
    normal at that sd implies. The band here is wide because the point is to
    catch a distribution that has stopped describing its own tails, not to
    pin a number that moves with the sample.
    """
    for lg in ("nfl", "cfb"):
        cov = grades[lg]["coverage"]
        assert 0.90 < cov["95"] < 0.98, f"{lg}: 95% interval covers {cov['95']}"
        assert 0.72 < cov["80"] < 0.88, f"{lg}: 80% interval covers {cov['80']}"


def test_the_two_measurable_total_sds_are_wrong_by_about_a_third(grades) -> None:
    """The numbers behind the refusal, pinned so the refusal stays justified.

    CFB's comparison is exact rather than indicative: its mu_total is a
    constant, so the residual and unconditional dispersions are the same
    quantity. NFL's comes from data/totals_validation.json -- the artifact
    that withheld the market was already carrying the number that contradicts
    its sd.
    """
    t = grades["totals"]
    assert t["cfb"]["dispersion_ratio"] > 1.25, (
        "CFB's total sd no longer understates dispersion; if a total model "
        "has been graded, wire total_validated=True and add the market"
    )
    assert t["cfb"]["coverage"]["95"] < 0.92
    assert t["nfl"]["dispersion_ratio"] > 1.25


def test_the_cfb_total_mean_placeholder_is_the_accurate_one(grades) -> None:
    """The labels are the wrong way round, and that is the point.

    52.0 was a guess and is right to seven hundredths of a point. 14.0 was
    merely 'unvalidated' and is the dangerous number. A label is not evidence
    in either direction.
    """
    c = grades["totals"]["cfb"]
    assert abs(c["mean_bias"]) < 0.5
    assert abs(c["mean_bias_t"]) < 2


def test_nba_total_dispersion_is_recorded_as_unmeasured(grades) -> None:
    """Unmeasured is reported as unmeasured, not as a pass.

    The unconditional sd bounds a residual only if the model has skill, which
    is exactly what is in question, so no ratio is computed and none is
    implied.
    """
    n = grades["totals"]["nba"]
    assert n["residual_sd"] is None
    assert n["not_measurable_here"]


def test_the_nba_sigma_claim_is_inside_the_bucketing_noise(grades) -> None:
    """ADR 0005's 0.60 was never evidence, and this is why.

    A correlation over a handful of bucket means has enormous sampling
    variability. Constant-variance noise pushed through the same five buckets
    gives a median absolute correlation near 0.4 and a 90th percentile near
    0.8, so "about 0.60" is unremarkable. The observed value on this data is
    -0.73 -- larger and opposite in sign, and equally meaningless.
    """
    b = grades["nba"]["bucketed_correlation_noise_ceiling"]["5"]
    assert b["p90_abs_from_constant_variance"] > 0.6, (
        "five-bucket correlations no longer reach 0.6 under constant "
        "variance, which would make ADR 0005's figure meaningful after all"
    )
    assert abs(b["observed"]) <= 1.0


def test_nba_dispersion_does_not_vary_with_the_predicted_spread(grades) -> None:
    """The per-game answer, on spreads that do reach where the claim lives.

    ADR 0006 recorded the effect as untested because these ratings were
    thought to separate games only across 0.9 to 8.6 points. Those were
    BUCKET MEANS. The predicted spread actually ranges to 23.3, which is
    where real NBA spreads live, so the question is answerable and the
    answer is no.
    """
    pg = grades["nba"]["per_game_correlation"]
    assert abs(pg["abs_spread_with_abs_residual"]) < 0.05
    assert abs(pg["abs_spread_with_squared_residual"]) < 0.05
    assert pg["predicted_spread_range"][1] > 15, (
        "the predicted spread no longer reaches where market spreads live, "
        "which would put the sigma question back out of reach"
    )


def test_nba_is_under_confident_on_its_graded_holdout(grades) -> None:
    """The safe direction, recorded rather than celebrated.

    Sigma is fitted on the tune seasons at 14.1666 and the graded holdout
    realised 12.99, a ratio of 0.917. Intervals are too wide, so edges are
    understated and stakes too small. That costs money slowly rather than
    quickly, which is the right way round but is not free.
    """
    h = grades["nba"]["holdout"]
    assert h["dispersion_ratio"] < 1.0, (
        "NBA is no longer under-confident on its holdout; if the ratio has "
        "gone above one the error has changed direction and the sigma choice "
        "should be revisited"
    )
    assert h["coverage"]["95"] > 0.95
