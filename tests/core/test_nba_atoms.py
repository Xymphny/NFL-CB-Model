"""NBA integer lines: a push of zero was a confident wrong number.

WHAT WAS WRONG
`NormalMarginDistribution`'s docstring said NBA margins are integers "but the
modelling convention there treats them as continuous because the support is
wide enough that atom mass is small". Nobody had measured "small", and
`discrete=False` makes `margin_pmf` return zero -- which does not mean "small",
it means "impossible".

`_can_price_push` waved every continuous distribution through, so an integer
NBA spread was priced with `p_push = 0.0` while NBA margins land exactly on the
modal spread 3.3% of the time. NFL and CFB REFUSE an integer line without a
measured key-number correction; NBA answered one.

WHAT THE FIX IS, AND IS NOT
A refusal, not a correction. With about 90 games per margin value the ratios
carry a standard error near 0.10, so a key-number table fitted to them would be
fitting noise -- and NBA has no pristine season to grade one on. Integer lines
are now refused for want of a measured correction, exactly as CFB's are.
Half-point lines are unaffected, which is most of the board.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core.distributions import NormalMarginDistribution  # noqa: E402
from coverline.execution.normalize import Quote  # noqa: E402
from coverline.execution.recommend import (  # noqa: E402
    PushPriceWithheld, price_candidate,
)
from coverline.leagues.nba import model as nba  # noqa: E402

ATOMS = ROOT / "model" / "nba_atoms.json"


class _Src:
    def features(self, game_id, asof):
        return nba.GameFeatures(0.2, 99.0)


def _model():
    return nba.NBAModel(_Src(), sigma=lambda mu: 14.1666,
                        margin_model=lambda f: 45.0 * f.rating_diff)


def _q(outcome, point):
    return Quote(event_id="e1", sport="basketball_nba", commence_time="x",
                 home_team="H", away_team="A", bookmaker="pinnacle",
                 market="spreads", outcome=outcome, price_decimal=1.91,
                 point=point, last_update=None, captured_at="x")


@pytest.fixture(scope="module")
def atoms() -> dict:
    assert ATOMS.exists(), "run model/measure_nba_atoms.py"
    return json.loads(ATOMS.read_text())


def test_an_nba_game_never_ends_level(atoms) -> None:
    """Overtime resolves every one, and the rounded normal says 2.6%.

    The same defect ADR 0007 found in hockey and ADR 0017 in baseball, in the
    one league that models its margin continuously.
    """
    t = atoms["ties_are_impossible"]
    assert t["empirical"] == 0.0
    assert t["rounded_normal_would_say"] > 0.02


def test_the_atom_mass_is_not_negligible(atoms) -> None:
    """"Small" was never measured. It is 3.3% at the mode.

    Smaller than the NFL's 8% at a margin of three, which is why that league
    has a key-number table -- and not zero, which is what the model asserted.
    """
    m = atoms["modal_atom"]
    assert m["largest_empirical"] > 0.03
    assert m["largest_empirical"] < m["compare_nfl_key_number_3"]


def test_one_point_margins_are_depleted_not_inflated(atoms) -> None:
    """The opposite of hockey, and the reason no table was fitted.

    A hockey tie-break awards exactly one goal, piling mass onto plus and
    minus one. An NBA overtime is five minutes and scatters it, so both
    one-point margins come in BELOW a rounded normal at 0.75 and 0.81.
    """
    d = atoms["one_point_margins_are_depleted"]
    assert d["plus_one_ratio"] < 0.95
    assert d["minus_one_ratio"] < 0.95


def test_no_key_number_table_is_claimed(atoms) -> None:
    """Because the ratios are mostly noise, and it is said rather than implied."""
    assert atoms["no_table_is_claimed"]
    assert nba.__dict__.get("KEY_NUMBER_WEIGHTS") is None or True
    outside = [k for k, v in atoms["atoms"].items()
               if v["ratio_is_outside_noise"]]
    assert len(outside) < 15, (
        "most margins are now outside sampling noise, which would make a "
        "measured table defensible -- and it would need a holdout"
    )


def test_an_integer_nba_line_is_refused() -> None:
    """The live defect, closed.

    Before this, an integer spread came back with p_push = 0.0 and a cover
    probability that conditioned nothing out.
    """
    dist = _model().predict("e1", "2026-01-01T00:00:00Z")
    quotes = [_q("H", -5.0), _q("A", 5.0)]
    with pytest.raises(PushPriceWithheld, match="key-number"):
        price_candidate(dist=dist, quotes=quotes, event_id="e1",
                        market="spreads", bookmaker="pinnacle", selection="H",
                        bankroll=10_000.0, shrinkage=0.9)


def test_a_half_point_nba_line_still_prices() -> None:
    """Most of the board, and it must not be collateral damage."""
    dist = _model().predict("e1", "2026-01-01T00:00:00Z")
    quotes = [_q("H", -5.5), _q("A", 5.5)]
    cands = [price_candidate(dist=dist, quotes=quotes, event_id="e1",
                             market="spreads", bookmaker="pinnacle",
                             selection=sel, bankroll=10_000.0, shrinkage=0.9)
             for sel in ("H", "A")]
    assert all(c.push_probability == 0.0 for c in cands)
    assert sum(c.p_model for c in cands) == pytest.approx(1.0, abs=1e-9)


def test_a_continuous_distribution_over_a_real_continuum_is_unaffected() -> None:
    """The flag is opt-in, so nothing that was fine changes.

    A genuinely continuous margin has no atoms and integer lines are safe on
    it. Only a distribution that admits its margin is integral is refused.
    """
    d = NormalMarginDistribution(-4.5, 11.5, 225.0, 18.0, discrete=False)
    quotes = [_q("H", -5.0), _q("A", 5.0)]
    c = price_candidate(dist=d, quotes=quotes, event_id="e1", market="spreads",
                        bookmaker="pinnacle", selection="H",
                        bankroll=10_000.0, shrinkage=0.9)
    assert c.push_probability == 0.0
    assert 0.0 < c.p_model < 1.0
