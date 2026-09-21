"""A market withheld in the list must also be refused by the object.

THE HALF-CONNECTION THIS CLOSES
NFL, CFB and NBA all withheld totals from `primary_markets`, and all three
went on answering `total_mean()` with a placeholder. A caller iterating
markets was safe. A caller asking the DISTRIBUTION was not -- and the
distribution is what the seam is for, since everything downstream imports core
and talks to a ScoreDistribution rather than to a league.

Two of the three placeholders are measurably wrong by about a third, and in
both cases the contradicting number was already in the repository:

  CFB   sd_total 14.0 against a measured 18.79 over 3,863 games. Its mu_total
        is a constant, so residual and unconditional dispersion are the same
        quantity and the comparison is exact. Nominal 95% covers 87.0%.
  NFL   sd_total 10.0 against the 13.353 RMSE recorded in the very artifact
        that withheld the market, data/totals_validation.json.
  NBA   18.0, simply unmeasured. Which is not better.

None were corrected. A right sd on an ungraded mean is still an ungraded
total, and fixing the visible half makes a failed market look ready.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core.distributions import (  # noqa: E402
    NormalMarginDistribution, UnvalidatedTotal,
)

TOTAL_METHODS = (("total_mean", ()), ("total_sd", ()),
                 ("total_cdf", (50.0,)), ("total_pmf", (50.0,)))


def _unvalidated() -> NormalMarginDistribution:
    return NormalMarginDistribution(-3.0, 13.6, 44.0, 10.0, discrete=True,
                                    total_validated=False)


@pytest.mark.parametrize("method,args", TOTAL_METHODS,
                         ids=[m for m, _ in TOTAL_METHODS])
def test_every_total_method_refuses(method: str, args: tuple) -> None:
    with pytest.raises(UnvalidatedTotal):
        getattr(_unvalidated(), method)(*args)


def test_sampling_a_scoreline_refuses_because_a_scoreline_has_a_total() -> None:
    """The refusal that is easy to forget.

    sample() returns (home, away), which encodes the total as surely as
    total_mean() does. Refusing the accessors and not the sampler would leave
    the placeholder reachable by the longer route.
    """
    rng = np.random.default_rng(1)
    with pytest.raises(UnvalidatedTotal):
        _unvalidated().sample(10, rng)


def test_margin_sampling_still_works() -> None:
    """Refusing the total must not refuse spread staking.

    Correlated slate staking needs draws and never touches the total
    dimension, so sample_margin exists and is exercised here rather than
    assumed.
    """
    rng = np.random.default_rng(2)
    d = _unvalidated()
    draws = d.sample_margin(40_000, rng)
    assert draws.shape == (40_000,)
    assert abs(draws.mean() - d.margin_mean()) < 5 * d.margin_sd() / 200
    assert np.allclose(draws, np.rint(draws)), "discrete margins must be integers"


def test_the_margin_is_untouched_by_the_refusal() -> None:
    """Everything the league DOES model still answers."""
    d = _unvalidated()
    assert d.margin_mean() == pytest.approx(-3.0)
    assert d.margin_sd() > 0
    assert 0.0 < d.margin_cdf(0.0) < 1.0
    assert d.margin_pmf(-3) > 0


def test_a_validated_total_is_unaffected() -> None:
    """The default stays True so a graded total needs no ceremony."""
    d = NormalMarginDistribution(-3.0, 13.6, 44.0, 10.0, discrete=True)
    assert d.total_mean() == pytest.approx(44.0)
    assert d.sample(10, np.random.default_rng(3)).shape == (10, 2)


@pytest.mark.parametrize("league", ["nfl", "cfb", "nba"])
def test_the_market_list_and_the_object_agree(league: str) -> None:
    """The invariant, checked on the real leagues rather than a stand-in.

    Offering a market the object refuses is a crash waiting for a live slate.
    Refusing a market the object could price is dead weight. Either way the
    two must say the same thing.
    """
    dist, markets = _real_league(league)
    try:
        dist.total_mean()
        models_total = True
    except UnvalidatedTotal:
        models_total = False
    assert models_total == ("total" in markets), (
        f"{league}: primary_markets says total={'total' in markets} and the "
        f"distribution says {models_total}"
    )


def _real_league(league: str):
    if league == "nfl":
        from coverline.leagues.nfl import model as m
        mod = m.NFLModel(_Src(m.GameFeatures(rating_diff=0.1,
                                             ngs_present=False)))
    elif league == "cfb":
        from coverline.leagues.cfb import model as m
        src = _Src(m.GameFeatures(rating_diff=0.1, elo_diff=0.0,
                                  elo_present=False))
        mod = m.CFBModel(src)
    else:
        from coverline.leagues.nba import model as m
        mod = m.NBAModel(_Src(m.GameFeatures(0.2, 99.0)),
                         sigma=lambda mu: 14.1666,
                         margin_model=lambda f: 45.0 * f.rating_diff)
    return mod.predict("g", "2026-09-21T00:00:00Z"), mod.primary_markets


class _Src:
    def __init__(self, f):
        self._f = f

    def features(self, game_id, asof):
        return self._f
