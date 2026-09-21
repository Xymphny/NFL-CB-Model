"""One assertion suite, run against every league.

This is the containment strategy for drift. The extraction into core/ is the
real fix -- five callers of one implementation cannot diverge -- but some
things genuinely stay per-league (feature engineering, the entity model, the
rating update rule), and those are what this suite holds to a common standard.

TWO HAZARDS THIS FILE IS BUILT AROUND
-------------------------------------
1. A conformance suite parameterised over a registry passes trivially when the
   registry is empty. test_registry_matches_the_checked_in_inventory is the
   guard on the guard: the set of built leagues is checked in, so a league
   disappearing from the registry fails by name, and adding one requires
   editing this file on purpose.

2. A harness with no leagues to run against is untested code that will be
   trusted later. So the assertions are exercised here and now against a
   synthetic reference league, and against deliberately broken variants of it,
   proving each assertion can actually fail before any real league arrives.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import registry  # noqa: E402
from coverline.core.distributions import (  # noqa: E402
    BivariatePoissonDistribution,
    NormalMarginDistribution,
)

#: Leagues actually built, checked in so that a disappearance is visible.
#: EMPTY IS CORRECT TODAY -- no league package exists yet. Adding 'nfl' here
#: without a working nfl package will fail the very next test, and building
#: one without adding it here will fail this one.
BUILT_LEAGUES: frozenset[str] = frozenset()


def test_registry_matches_the_checked_in_inventory():
    assert registry.registered() == BUILT_LEAGUES, (
        f"registry has {sorted(registry.registered())}, inventory says "
        f"{sorted(BUILT_LEAGUES)}. If a league was added, add it here too; if "
        "one vanished, find out why before editing this line."
    )


def test_expected_leagues_are_the_five_intended():
    assert registry.EXPECTED_LEAGUES == frozenset({"nfl", "cfb", "mlb", "nhl", "nba"})


def test_unbuilt_leagues_are_reported_rather_than_forgotten():
    missing = registry.missing()
    print(f"\n  leagues declared but not built: {sorted(missing)}")
    assert missing == registry.EXPECTED_LEAGUES - BUILT_LEAGUES


# ------------------------------------------- the shared assertion battery ----

def assert_distribution_conforms(dist, label: str) -> None:
    """Every ScoreDistribution must satisfy these, whatever league made it.

    Kept as a plain function so it can be applied both to real registered
    leagues (once they exist) and to the synthetic ones below.
    """
    rng = np.random.default_rng(7)

    # moments are finite and sane
    assert np.isfinite(dist.margin_mean()), f"{label}: non-finite margin mean"
    assert dist.margin_sd() > 0, f"{label}: non-positive margin sd"
    assert np.isfinite(dist.total_mean()), f"{label}: non-finite total mean"

    # cdf is a cdf
    lo = dist.margin_mean() - 6 * dist.margin_sd()
    hi = dist.margin_mean() + 6 * dist.margin_sd()
    assert dist.margin_cdf(lo) < 0.01, f"{label}: cdf not ~0 in the left tail"
    assert dist.margin_cdf(hi) > 0.99, f"{label}: cdf not ~1 in the right tail"
    grid = np.linspace(lo, hi, 40)
    values = [dist.margin_cdf(x) for x in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), (
        f"{label}: margin cdf is not monotone"
    )

    # discreteness is coherent: cdf steps equal pmf atoms, or there are none
    if dist.is_discrete:
        k = int(round(dist.margin_mean()))
        step = dist.margin_cdf(k) - dist.margin_cdf(k - 1)
        assert step == pytest.approx(dist.margin_pmf(k), abs=1e-9), (
            f"{label}: cdf step does not equal pmf atom -- push prices will be wrong"
        )
        assert dist.margin_pmf(k + 0.5) == 0.0, f"{label}: mass off the integers"
    else:
        assert dist.margin_pmf(0) == 0.0, f"{label}: continuous but claims an atom"

    # it can simulate itself, and the simulation agrees with the analytic mean
    draws = dist.sample(20_000, rng)
    assert draws.shape == (20_000, 2), f"{label}: sample returned the wrong shape"
    sim_margin = (draws[:, 0] - draws[:, 1]).mean()
    tol = 5 * dist.margin_sd() / np.sqrt(20_000)
    assert abs(sim_margin - dist.margin_mean()) < max(tol, 0.05), (
        f"{label}: simulated mean {sim_margin:.3f} disagrees with analytic "
        f"{dist.margin_mean():.3f}"
    )


#: Stand-ins for the two families, so the battery above is exercised today.
REFERENCE = {
    "normal-football": NormalMarginDistribution(-3.0, 13.6, 44.0, 10.0, discrete=True),
    "normal-basketball": NormalMarginDistribution(
        -4.5, 11.5, 225.0, 18.0, discrete=False
    ),
    "poisson-hockey": BivariatePoissonDistribution(3.1, 2.8, 0.3),
    "poisson-baseball": BivariatePoissonDistribution(4.6, 4.2, 0.5),
}


@pytest.mark.parametrize("label", sorted(REFERENCE))
def test_reference_distributions_conform(label):
    assert_distribution_conforms(REFERENCE[label], label)


@pytest.mark.parametrize("league", sorted(BUILT_LEAGUES))
def test_every_built_league_conforms(league):  # pragma: no cover - none built yet
    model = registry.get(league)
    assert model.league == league
    assert model.primary_markets, f"{league} declares no markets"
    dist = model.predict("conformance-probe", "2026-09-21T00:00:00Z")
    assert_distribution_conforms(dist, league)


# ------------------------------------------- prove the battery can fail ----

class _BrokenCdf:
    """A distribution whose cdf steps disagree with its pmf: the exact defect
    that makes push prices wrong while everything else looks fine."""

    is_discrete = True

    def margin_mean(self): return 0.0
    def margin_sd(self): return 10.0
    def margin_cdf(self, x): return float(np.clip((x + 60) / 120, 0, 1))
    def margin_pmf(self, x): return 0.05 if float(x).is_integer() else 0.0
    def total_mean(self): return 44.0
    def total_cdf(self, x): return 0.5
    def total_pmf(self, x): return 0.0
    def sample(self, n, rng): return np.zeros((n, 2))


def test_the_battery_catches_a_cdf_pmf_mismatch():
    with pytest.raises(AssertionError, match="push prices will be wrong"):
        assert_distribution_conforms(_BrokenCdf(), "broken")


def test_the_battery_catches_a_simulator_that_disagrees_with_its_own_maths():
    """A distribution whose sample() drifts from its analytic mean. This is the
    failure mode that produced a real bug once already -- analytic pricing and
    simulated staking silently describing different games."""
    # Shifting BOTH columns leaves the margin intact and would not be caught,
    # which is itself worth knowing: this battery checks the margin, so a
    # total-only drift needs its own assertion. Shift one column.
    class _DriftingMargin(NormalMarginDistribution):
        def sample(self, n, rng):
            out = super().sample(n, rng)
            out[:, 0] += 7.0
            return out

    with pytest.raises(AssertionError, match="disagrees with analytic"):
        assert_distribution_conforms(_DriftingMargin(-3.0, 13.6, 44.0, 10.0), "drift")
