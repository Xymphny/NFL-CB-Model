"""Two vocabularies for markets, and the bridge that had never been built.

WHAT WAS BROKEN
Leagues declare what they price in their own words -- "spread", "moneyline",
"total", "runline", "puck_line". Quotes arrive in the vendor's -- "spreads",
"h2h", "totals". Nothing translated, and nothing ever had to, because
`LeagueModel.primary_markets` had NO PRODUCTION CONSUMER: only tests read it.
`recommend()` took a market string, matched it against the vendor key on a
quote, and never asked the league whether it priced that market.

So the declared list was documentation. Anyone wiring the recommender to a
league's own markets would have matched zero quotes, because "spread" is not
"spreads" -- and anyone passing the vendor key straight through would have
been priced on markets the league withholds.

This is ADR 0011's shape again: a market withheld in a list that nothing
enforces. That one was caught because the distribution refused. This one had
no object to refuse.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core.markets import (  # noqa: E402
    INTERNAL_NAMES, VENDOR_KEY, MarketNotOffered, UnknownMarket,
    require_offered, vendor_key,
)

#: The real lists, so this test fails if a league changes what it offers
#: without anyone thinking about the vendor side.
LEAGUE_MARKETS = {
    "nfl": ("spread", "moneyline"),
    "cfb": ("spread", "moneyline"),
    "nba": ("spread", "moneyline"),
    "mlb": ("moneyline", "runline", "total"),
    "nhl": ("moneyline", "total", "puck_line"),
}


def test_every_declared_market_has_a_vendor_key() -> None:
    """A league cannot offer something the vendor cannot be asked for."""
    for league, markets in LEAGUE_MARKETS.items():
        for m in markets:
            assert m in VENDOR_KEY, (
                f"{league} offers {m!r} and nothing maps it to a vendor key, "
                "so quotes for it can never be found"
            )


def test_the_handicap_is_three_names_for_one_vendor_key() -> None:
    """Many-to-one on purpose, and the distinction is not cosmetic.

    A runline and a puck line are always 1.5; an NFL spread moves. That is
    why the NHL model prices shape and the NFL model prices a mean, and it is
    why the league vocabulary keeps them apart while the vendor does not.
    """
    assert VENDOR_KEY["spread"] == VENDOR_KEY["runline"] == VENDOR_KEY["puck_line"]
    assert set(INTERNAL_NAMES["spreads"]) == {"spread", "runline", "puck_line"}


@pytest.mark.parametrize("league,markets", sorted(LEAGUE_MARKETS.items()))
def test_a_vendor_key_resolves_to_that_league_s_own_name(league, markets) -> None:
    """"spreads" means puck_line in the NHL and spread in the NFL."""
    for vendor in {VENDOR_KEY[m] for m in markets}:
        resolved = require_offered(vendor, markets, league)
        assert resolved in markets
        assert VENDOR_KEY[resolved] == vendor


def test_a_market_a_league_withholds_is_refused() -> None:
    """The control the declared list never had.

    NFL totals graded supported=false and NBA has no total model at all.
    Before this, asking for either returned a priced signal.
    """
    with pytest.raises(MarketNotOffered, match="nfl"):
        require_offered("totals", LEAGUE_MARKETS["nfl"], "nfl")
    with pytest.raises(MarketNotOffered):
        require_offered("total", LEAGUE_MARKETS["nba"], "nba")


def test_the_nhl_handicap_is_refused_without_the_rules_layer() -> None:
    """The specific case this was written for.

    An NHL model with no rules layer offers no handicap at all. Asking for
    "spreads" must refuse rather than fall back to pricing a puck line off a
    distribution that cannot express the shape one needs.
    """
    without = ("moneyline", "total")
    with pytest.raises(MarketNotOffered, match="spreads"):
        require_offered("spreads", without, "nhl")
    assert require_offered("spreads", ("moneyline", "total", "puck_line"),
                           "nhl") == "puck_line"


def test_a_typo_is_a_different_error_from_a_withheld_market() -> None:
    """Because the remedies differ, and a shared exception hides which."""
    with pytest.raises(UnknownMarket):
        vendor_key("spred")
    with pytest.raises(UnknownMarket):
        require_offered("spred", LEAGUE_MARKETS["nfl"], "nfl")


def test_a_vendor_key_passes_through_unchanged() -> None:
    """A narrow tolerance, so this module could land without rewriting every
    call site at once. Tested because an untested tolerance becomes a bug."""
    for vendor in INTERNAL_NAMES:
        assert vendor_key(vendor) == vendor


def test_recommend_refuses_a_market_the_league_does_not_offer() -> None:
    """End to end, on the real entry point.

    primary_markets is optional on recommend() so that existing call sites
    keep working -- a required parameter is how a safety feature gets
    reverted -- but when it is passed it is enforced.
    """
    from coverline.execution import recommend as Rc
    from tests.core.test_recommend import _dist, _market  # reuse the fixtures

    quotes = _market(point=-1.5)
    with pytest.raises(MarketNotOffered):
        Rc.recommend(dist=_dist(mu=+0.3), quotes=quotes, event_id="e1",
                     market="spreads", league="nhl", bankroll=10_000,
                     shrinkage=0.9, primary_markets=("moneyline", "total"))
