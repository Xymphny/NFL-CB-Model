"""Check the analytic methods against simulation.

The point of these tests is that they do not trust the closed forms. Each
distribution can simulate itself, and a cdf or pmf that disagrees with a large
sample from its own sample() is wrong in one of the two places -- which is the
only way to catch an algebra error in code nobody will re-derive later.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coverline.core.distributions import (  # noqa: E402
    BivariatePoissonDistribution,
    NormalMarginDistribution,
)

N = 400_000
SEED = 20260921


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------- normal ----

def test_normal_margin_cdf_matches_simulation():
    d = NormalMarginDistribution(
        mu_margin=-3.0, sd_margin=13.6, mu_total=44.0, sd_total=10.0
    )
    draws = d.sample(N, _rng())
    margin = draws[:, 0] - draws[:, 1]
    for line in (-10.0, -3.0, 0.0, 7.0, 14.0):
        empirical = float(np.mean(margin <= line))
        assert d.margin_cdf(line) == pytest.approx(empirical, abs=3e-3), line


def test_normal_margin_pmf_matches_simulation():
    d = NormalMarginDistribution(
        mu_margin=-3.0, sd_margin=13.6, mu_total=44.0, sd_total=10.0
    )
    draws = d.sample(N, _rng())
    margin = draws[:, 0] - draws[:, 1]
    for k in (-7, -3, 0, 3, 7):
        empirical = float(np.mean(margin == k))
        assert d.margin_pmf(k) == pytest.approx(empirical, abs=3e-3), k


def test_normal_pmf_is_zero_off_the_integers_and_when_continuous():
    discrete = NormalMarginDistribution(
        mu_margin=0.0, sd_margin=10.0, mu_total=40.0, sd_total=8.0, discrete=True
    )
    assert discrete.margin_pmf(3.5) == 0.0

    continuous = NormalMarginDistribution(
        mu_margin=0.0, sd_margin=10.0, mu_total=220.0, sd_total=18.0, discrete=False
    )
    assert continuous.margin_pmf(3) == 0.0
    assert continuous.is_discrete is False


def test_discrete_cdf_includes_the_atom():
    """P(X <= k) - P(X <= k-1) must equal P(X == k). If it does not, every
    push price computed from this distribution is wrong by the atom."""
    d = NormalMarginDistribution(
        mu_margin=-2.5, sd_margin=13.0, mu_total=45.0, sd_total=10.0
    )
    for k in (-7, -3, 0, 3, 7):
        step = d.margin_cdf(k) - d.margin_cdf(k - 1)
        assert step == pytest.approx(d.margin_pmf(k), abs=1e-12), k


def test_key_number_table_is_absent_and_says_so():
    """The evidence pillar, in test form: shipping without the measured table
    is allowed, shipping while CLAIMING to have it is not."""
    d = NormalMarginDistribution(
        mu_margin=-3.0, sd_margin=13.6, mu_total=44.0, sd_total=10.0
    )
    assert d.has_key_number_correction is False, (
        "a key-number table appeared; if it was measured, record its evidence "
        "and update this test -- if it was guessed, delete it"
    )


def test_key_number_table_overrides_when_supplied():
    d = NormalMarginDistribution(
        mu_margin=-3.0,
        sd_margin=13.6,
        mu_total=44.0,
        sd_total=10.0,
        key_number_mass={3: 0.0975},
    )
    assert d.has_key_number_correction is True
    assert d.margin_pmf(3) == pytest.approx(0.0975)
    assert d.margin_pmf(4) < 0.05  # untouched values still come from the normal


def test_continuous_margin_rejects_a_key_number_table():
    with pytest.raises(ValueError, match="meaningless"):
        NormalMarginDistribution(
            mu_margin=0.0,
            sd_margin=11.0,
            mu_total=220.0,
            sd_total=18.0,
            discrete=False,
            key_number_mass={3: 0.05},
        )


def test_normal_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        NormalMarginDistribution(0.0, -1.0, 44.0, 10.0)
    with pytest.raises(ValueError):
        NormalMarginDistribution(0.0, 13.0, 44.0, 10.0, rho=1.0)


# --------------------------------------------------------------- poisson ----

def test_poisson_margin_matches_simulation():
    d = BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=0.3)
    draws = d.sample(N, _rng())
    margin = draws[:, 0] - draws[:, 1]
    for line in (-2, -1, 0, 1, 2):
        assert d.margin_cdf(line) == pytest.approx(
            float(np.mean(margin <= line)), abs=3e-3
        ), line
        assert d.margin_pmf(line) == pytest.approx(
            float(np.mean(margin == line)), abs=3e-3
        ), line


def test_poisson_total_matches_simulation():
    d = BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=0.3)
    draws = d.sample(N, _rng())
    total = draws[:, 0] + draws[:, 1]
    for line in (4, 5, 6, 7, 8):
        assert d.total_cdf(line) == pytest.approx(
            float(np.mean(total <= line)), abs=3e-3
        ), line
        assert d.total_pmf(line) == pytest.approx(
            float(np.mean(total == line)), abs=3e-3
        ), line


def test_shared_component_cancels_in_the_margin_but_not_the_total():
    """The property the whole class exists for. If this ever fails, the shared
    component is leaking into the margin and NHL/MLB spread prices move when
    only total variance was meant to."""
    a = BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=0.0)
    b = BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=0.9)

    assert a.margin_mean() == pytest.approx(b.margin_mean())
    assert a.margin_sd() == pytest.approx(b.margin_sd())
    for k in (-2, -1, 0, 1, 2):
        assert a.margin_pmf(k) == pytest.approx(b.margin_pmf(k), abs=1e-12), k

    assert b.total_mean() > a.total_mean()
    assert b.total_sd() > a.total_sd()


def test_poisson_total_pmf_is_a_distribution():
    d = BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=0.4)
    assert sum(d.total_pmf(k) for k in range(0, 60)) == pytest.approx(1.0, abs=1e-9)
    assert d.total_pmf(-1) == 0.0
    assert d.total_pmf(3.5) == 0.0


def test_poisson_moments_match_simulation():
    d = BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=0.4)
    draws = d.sample(N, _rng())
    margin = draws[:, 0] - draws[:, 1]
    total = draws[:, 0] + draws[:, 1]
    assert d.margin_mean() == pytest.approx(float(margin.mean()), abs=0.02)
    assert d.margin_sd() == pytest.approx(float(margin.std()), abs=0.02)
    assert d.total_mean() == pytest.approx(float(total.mean()), abs=0.02)
    assert d.total_sd() == pytest.approx(float(total.std()), abs=0.02)


def test_poisson_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        BivariatePoissonDistribution(lam_home=0.0, lam_away=2.8)
    with pytest.raises(ValueError):
        BivariatePoissonDistribution(lam_home=3.1, lam_away=2.8, lam_shared=-0.1)
