"""Oracle-free properties across the path that decides what to bet.

WHY A SEPARATE FILE
Three pricing bugs shipped past a suite that covers these modules heavily.
Every test that missed them had the same shape: price one thing, compare it to
a number worked out the same way the code works it out. That is agreement with
yourself, and it cannot catch an error in the shared reasoning.

The properties here relate two outputs to each other, or an output to an
invariant of the domain. None needs an oracle:

  - devigged probabilities sum to one, whatever the method
  - a shrunk probability lies between the two it was shrunk from
  - a stake never exceeds its cap and never rises when the edge falls
  - a distribution's own pmf reproduces its own mean
  - settlement arithmetic obeys the definition of a bet

Randomised over a fixed seed rather than hand-picked, because the cases that
found the last three bugs were not the ones anybody would have chosen.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coverline.core import pricing as P  # noqa: E402
from coverline.core import staking as S  # noqa: E402
from coverline.core.distributions import (  # noqa: E402
    BivariatePoissonDistribution, NegativeBinomialScoreDistribution,
    NormalMarginDistribution,
)

RNG = np.random.default_rng(20260921)
METHODS = ("multiplicative", "additive", "power", "shin")


def _two_way(n: int = 60):
    """Plausible two-way markets: a fair pair plus a bookmaker's margin.

    The favourite's side is kept below an implied certainty. The first
    version of this generator did not, produced a raw probability above one
    at the extremes, and made devig raise "decimal odds must exceed 1.0" --
    a test bug that read exactly like a devig bug for a minute.
    """
    for _ in range(n):
        hold = float(RNG.uniform(0.005, 0.08))
        # BOTH sides must stay under an implied certainty, not just the one
        # being varied. Clamping only the favourite still produced a raw
        # probability above one on the other side.
        hi = 0.99 / (1.0 + hold)
        p = float(RNG.uniform(1.0 - hi, hi))
        raw = np.array([p, 1.0 - p]) * (1.0 + hold)
        yield [float(1.0 / r) for r in raw]


# -- devigging --------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_devigged_probabilities_sum_to_one(method):
    """The defining property of a devig, checked on every method.

    A method that does not normalise is not devigging, it is rescaling, and
    the difference is invisible when only one side is ever inspected.
    """
    for prices in _two_way():
        fair = P.devig(prices, method)
        assert float(np.sum(fair)) == pytest.approx(1.0, abs=1e-9), method
        assert (fair > 0).all() and (fair < 1).all()


@pytest.mark.parametrize("method", METHODS)
def test_devigging_preserves_the_order_of_the_prices(method):
    """The shorter price must stay the likelier outcome.

    An inversion here would flip which side looks valuable, which is the
    single most expensive thing a devig can get wrong.
    """
    for prices in _two_way():
        fair = P.devig(prices, method)
        assert (fair[0] > fair[1]) == (prices[0] < prices[1]), method


@pytest.mark.parametrize("method", METHODS)
def test_a_market_with_no_hold_is_returned_unchanged(method):
    """Nothing to remove means nothing to move."""
    for p in (0.25, 0.5, 0.735):
        prices = [1.0 / p, 1.0 / (1.0 - p)]
        fair = P.devig(prices, method)
        assert fair[0] == pytest.approx(p, abs=1e-6), method


# -- shrinking --------------------------------------------------------------

def test_a_shrunk_probability_lies_between_its_two_inputs():
    """Shrinkage moves toward the market and never past it, or past itself."""
    for _ in range(300):
        pm = float(RNG.uniform(0.02, 0.98))
        pk = float(RNG.uniform(0.02, 0.98))
        w = float(RNG.uniform(0.0, 1.0))
        out = S.shrink_probability(pk, pm, w)
        lo, hi = min(pk, pm), max(pk, pm)
        assert lo - 1e-12 <= out <= hi + 1e-12


def test_the_two_ends_of_shrinkage_are_the_two_inputs():
    """Weight 0 is the market, weight 1 is the model. No weight is both."""
    for _ in range(50):
        pm, pk = float(RNG.uniform(0.05, 0.95)), float(RNG.uniform(0.05, 0.95))
        assert S.shrink_probability(pk, pm, 0.0) == pytest.approx(pm, abs=1e-12)
        assert S.shrink_probability(pk, pm, 1.0) == pytest.approx(pk, abs=1e-12)


# -- staking ----------------------------------------------------------------

def _plan(**kw):
    base = dict(p_market=0.50, decimal_price=1.95, bankroll=10_000.0,
                shrinkage=0.9)
    base.update(kw)
    return S.size_bet(**base)


def test_a_stake_never_exceeds_its_cap():
    """The last line of defence, and the one a wrong probability runs into."""
    for _ in range(300):
        pm = float(RNG.uniform(0.3, 0.7))
        pk = float(RNG.uniform(0.3, 0.999))
        price = float(RNG.uniform(1.05, 6.0))
        cap = float(RNG.uniform(0.005, 0.05))
        plan = S.size_bet(p_model=pk, p_market=pm, decimal_price=price,
                          bankroll=10_000.0, shrinkage=0.9,
                          max_bankroll_fraction=cap)
        assert plan.bankroll_fraction <= cap + 1e-12
        assert plan.stake <= 10_000.0 * cap + 0.01


def test_agreeing_with_the_market_leaves_only_the_hold_to_bet():
    """A model that agrees with the devigged market has a NEGATIVE edge.

    The first version of this asserted that p_model == p_market means no
    stake, at an arbitrary price. That is wrong and it was my error, not the
    code's: edge is measured against the PRICE, so agreeing with the market
    at a generous price is still an edge. What is true, and what the real
    path guarantees, is that a side's price already carries the hold -- so
    agreement leaves exactly the hold to lose.
    """
    for p in (0.25, 0.5, 0.735):
        hold = 0.045
        prices = [1.0 / (p * (1 + hold)), 1.0 / ((1 - p) * (1 + hold))]
        fair = P.devig(prices, "power")
        plan = S.size_bet(p_model=float(fair[0]), p_market=float(fair[0]),
                          decimal_price=prices[0], bankroll=10_000.0,
                          shrinkage=0.9)
        assert plan.edge_used < 0.0
        assert plan.stake == 0.0
        assert not plan.placed


def test_a_stake_does_not_rise_when_the_edge_falls():
    """Monotonicity in the model probability, at a fixed price and market.

    Not a tautology: the stake passes through shrinking, Kelly, a multiple
    and a cap, and any of them could invert it.
    """
    prev = -1.0
    for pk in np.linspace(0.50, 0.95, 40):
        plan = _plan(p_model=float(pk))
        assert plan.bankroll_fraction >= prev - 1e-12, (
            f"stake fell as the edge rose at p_model={pk:.3f}")
        prev = plan.bankroll_fraction


def test_shrinking_harder_never_increases_the_stake():
    """A smaller weight trusts the model less, so it cannot bet more.

    This is the ordering argument in size_bet's docstring, asserted rather
    than described: shrink first, then Kelly.
    """
    prev = None
    for w in np.linspace(1.0, 0.0, 30):
        plan = _plan(p_model=0.62, shrinkage=float(w))
        if prev is not None:
            assert plan.bankroll_fraction <= prev + 1e-12
        prev = plan.bankroll_fraction
    assert prev == 0.0, "zero weight must leave no edge at all"


# -- distributions ----------------------------------------------------------

DISTS = {
    "normal-keyed": NormalMarginDistribution(
        -3.0, 13.3, 44.0, 10.0, discrete=True,
        key_number_weights={3: 2.2, 7: 1.6}),
    "normal-plain": NormalMarginDistribution(-4.5, 11.5, 225.0, 18.0,
                                             discrete=False),
    "poisson": BivariatePoissonDistribution(3.1, 2.8, 0.3),
    "negbin": NegativeBinomialScoreDistribution(4.6, 4.2, 3.888, 3.2426),
}


@pytest.mark.parametrize("name", sorted(DISTS))
def test_a_distributions_own_pmf_reproduces_its_own_mean(name):
    """Self-consistency between two ways of asking the same question.

    A key-number table that renormalises incorrectly shows up here and
    nowhere else: each atom still looks plausible, and the mean drifts.
    """
    d = DISTS[name]
    if not d.is_discrete:
        pytest.skip("continuous margin has no atoms to sum")
    xs = np.arange(-60, 61)
    mass = np.array([d.margin_pmf(float(x)) for x in xs])
    assert mass.sum() == pytest.approx(1.0, abs=1e-6), "margin pmf does not sum to 1"
    # Tolerance covers the class's own +/-60 support against this wider sum.
    assert float((xs * mass).sum()) == pytest.approx(d.margin_mean(), abs=0.02)


@pytest.mark.parametrize("mu", [-10.0, -3.0, 0.0, 7.0])
def test_key_numbers_move_the_mean_and_the_object_admits_it(mu):
    """The shipped table pulls the mean toward zero, and that is now reported.

    Renormalising a multiplicative reweighting fixes the total mass and does
    NOT fix the first moment, so "the weights are a SHAPE adjustment" is the
    intent and not the arithmetic. Measured on
    data/nfl_key_numbers.json the pull reaches 0.46 points, and margin_mean
    used to return the INPUT mu regardless -- a statement about a
    distribution this object does not represent.

    No price changed: prices come from margin_cdf and margin_pmf.
    """
    import json

    w = {int(k): float(v) for k, v in json.loads(
        (ROOT / "data" / "nfl_key_numbers.json").read_text())["weights"].items()}
    d = NormalMarginDistribution(mu, 13.2979, 44.0, 10.0, discrete=True,
                                 key_number_weights=w)
    xs = np.arange(-80, 81)
    mass = np.array([d.margin_pmf(float(x)) for x in xs])
    assert d.margin_mean() == pytest.approx(float((xs * mass).sum()), abs=0.01)
    # Toward zero, with room for the table's own asymmetry: at a pick'em the
    # measured weights give -0.064 rather than 0, because NFL margins are not
    # symmetric even at a neutral centre.
    assert abs(d.margin_mean()) <= abs(mu) + 0.1, (
        "the key-number table should pull the mean toward zero, not away"
    )
    plain = NormalMarginDistribution(mu, 13.2979, 44.0, 10.0, discrete=True)
    assert plain.margin_mean() == pytest.approx(mu, abs=1e-12), (
        "without weights the reported mean must still be the input"
    )


@pytest.mark.parametrize("name", sorted(DISTS))
def test_the_cdf_and_the_pmf_tell_the_same_story(name):
    """cdf(x) - cdf(x-1) must be the atom at x, or push prices are wrong."""
    d = DISTS[name]
    if not d.is_discrete:
        pytest.skip("continuous margin has no atoms")
    for x in range(-8, 9):
        step = d.margin_cdf(x) - d.margin_cdf(x - 1)
        assert step == pytest.approx(d.margin_pmf(x), abs=1e-9), x


# -- settlement -------------------------------------------------------------

@pytest.mark.parametrize("price", [1.2, 1.91, 2.5, 6.0])
@pytest.mark.parametrize("stake", [10.0, 137.55, 1000.0])
def test_settlement_arithmetic_obeys_the_definition_of_a_bet(price, stake):
    """A win returns stake*(price-1), a loss -stake, a push zero.

    Written out because the definition is the only oracle there is, and
    because a sign error here is invisible until a bankroll curve is wrong.
    """
    win = stake * (price - 1.0)
    lose = -stake
    push = 0.0
    assert win > 0 and lose < 0 and push == 0.0
    # A fair price returns zero expectation, which ties the two sides
    # together rather than checking each against itself.
    p_fair = 1.0 / price
    assert p_fair * win + (1.0 - p_fair) * lose == pytest.approx(0.0, abs=1e-9)


def test_the_published_margin_and_the_priced_margin_are_not_the_same_number():
    """The gap the parity test used to hide, stated as a fact about the board.

    `predict_margin` is the coefficient output and is what a board publishes
    as "the model's line". `margin_mean()` is the mean of the distribution
    those coefficients are turned into, and the key-number table moves it by
    up to half a point.

    Neither is wrong. They are different quantities and were being treated as
    one, which is why the parity test broke when the second one started
    telling the truth. Recorded here so that a board publishing one while
    pricing from the other is a decision somebody made rather than a detail
    nobody noticed.
    """
    import json

    from coverline.leagues.nfl import model as nfl

    w = {int(k): float(v) for k, v in json.loads(
        (ROOT / "data" / "nfl_key_numbers.json").read_text())["weights"].items()}
    gaps = []
    for rating_diff in (-0.25, -0.1, 0.0, 0.1, 0.25):
        f = nfl.GameFeatures(rating_diff=rating_diff, ngs_present=False)
        mu = nfl.predict_margin(f)
        d = NormalMarginDistribution(mu, nfl.MARGIN_SD, 44.0,
                                     nfl.TOTAL_SD_UNVALIDATED, discrete=True,
                                     key_number_weights=w,
                                     total_validated=False)
        gaps.append(abs(d.margin_mean() - mu))
    assert max(gaps) > 0.05, (
        "the two margins have converged; if the key-number table now "
        "preserves the mean, this distinction has gone away and the board "
        "can publish either"
    )
    assert max(gaps) < 1.0, (
        f"the gap has grown to {max(gaps):.2f} points, which is a key number "
        "in itself and too large to leave as an annotation"
    )


# -- closing line value -----------------------------------------------------

def _close(hold=0.045, p=0.55):
    """A two-sided closing market with a bookmaker's margin on it."""
    return [1.0 / (p * (1 + hold)), 1.0 / ((1 - p) * (1 + hold))]


def test_betting_the_posted_close_is_a_losing_bet():
    """The property that proves devigging is actually happening.

    Taking exactly the price the market closed at must show NEGATIVE EV,
    because that price carries the hold. A CLV implementation that forgot to
    devig would report exactly zero here and flatter every bet by the margin
    -- which is what a CLV number means when nobody says which one they
    computed.
    """
    for hold in (0.01, 0.045, 0.08):
        close = _close(hold=hold)
        clv = P.closing_line_value(bet_decimal=close[0], close_decimals=close,
                                   outcome_index=0)
        assert clv.ev_pct < 0.0
        assert clv.naive_ev_pct == pytest.approx(0.0, abs=1e-12)
        assert clv.ev_pct < clv.naive_ev_pct


def test_betting_the_fair_close_is_exactly_break_even():
    """The other end of the same property."""
    for p in (0.2, 0.5, 0.77):
        close = _close(p=p)
        fair = float(P.devig(close, "power")[0])
        clv = P.closing_line_value(bet_decimal=1.0 / fair,
                                   close_decimals=close, outcome_index=0)
        assert clv.ev_pct == pytest.approx(0.0, abs=1e-9)
        assert clv.prob_points == pytest.approx(0.0, abs=1e-9)


def test_a_better_price_is_always_better_clv():
    """Monotonicity in the price taken, at a fixed close."""
    close = _close()
    prev_ev, prev_pts = -9.9, -9.9
    for bet in np.linspace(1.3, 4.0, 40):
        clv = P.closing_line_value(bet_decimal=float(bet),
                                   close_decimals=close, outcome_index=0)
        assert clv.ev_pct >= prev_ev - 1e-12
        assert clv.prob_points >= prev_pts - 1e-12
        prev_ev, prev_pts = clv.ev_pct, clv.prob_points


def test_american_cents_are_not_comparable_and_probability_points_are():
    """Why no mean is reported in cents.

    American odds are discontinuous at even money -- nothing exists between
    -100 and +100 -- and non-linear elsewhere. A move of 2.4 probability
    points that crosses the boundary reads as 210 cents, while 1.1 points
    inside the favourite range reads as 5. Averaging those is meaningless,
    and the trap is that the resulting number looks like a CLV.
    """
    def _row(bet, close_price):
        close = [close_price, 1.0 / (1.0 - 1.0 / close_price)]
        clv = P.closing_line_value(bet_decimal=bet, close_decimals=close,
                                   outcome_index=0)
        return abs(clv.prob_points), abs(clv.price_cents)

    rows = [_row(2.05, 1.952), _row(1.20, 1.18), _row(1.91, 1.87)]
    per_point = [cents / pts for pts, cents in rows]
    assert max(per_point) / min(per_point) > 10, (
        "cents per probability point is no longer wildly non-constant; the "
        "discontinuity warning in ClosingLineValue may no longer apply"
    )

    # And the ranking genuinely inverts, which is the strongest form of it:
    # a deep favourite moving by less than one probability point reads as
    # more cents than a move of two and a half points across even money.
    small_move_many_cents = _row(1.05, 1.04)
    big_move_fewer_cents = _row(2.05, 1.952)
    assert small_move_many_cents[0] < big_move_fewer_cents[0]
    assert small_move_many_cents[1] > big_move_fewer_cents[1]


def test_the_clv_summary_reports_a_scale_that_can_be_averaged():
    """Guarding the summary itself, not just the primitive."""
    import inspect

    from coverline.execution import grade as G

    src = inspect.getsource(G.clv_summary)
    assert "mean_clv_probability_points" in src
    assert "cents" in src, "the reason no cents mean is reported must stay stated"


# -- credits ----------------------------------------------------------------
#
# The budget guard is the only thing between a bug and a month's quota, and
# Thursday is when a paid key arrives. These properties hold it to its own
# claim -- "Nothing was issued" -- over random sequences rather than one
# hand-picked case.


class _CountingTransport:
    """Records every URL it is asked for, and what the API 'charged'."""

    def __init__(self, charge: int | None = None):
        self.urls: list[str] = []
        self.charge = charge

    def get(self, url: str):
        self.urls.append(url)
        headers = {"x-requests-remaining": "400"}
        if self.charge is not None:
            headers["x-requests-last"] = str(self.charge)
        return 200, b"[]", headers


def _client(budget: int, transport, tolerance: int = 0):
    from coverline.execution.odds_client import CreditLedger, OddsAPIClient

    ledger = CreditLedger(budget=budget, drift_tolerance=tolerance)
    return OddsAPIClient(api_key="k", ledger=ledger, transport=transport), ledger


def test_the_cost_model_is_multiplicative_and_monotone():
    """markets x regions, and ten times that for history.

    Asserted as a shape rather than a table, so a formula change shows up
    here rather than in a bill.
    """
    from coverline.execution.odds_client import HISTORICAL_MULTIPLIER as HM
    from coverline.execution.odds_client import OddsAPIClient as C

    for m in range(1, 6):
        for r in range(1, 4):
            markets, regions = ["x"] * m, ["y"] * r
            assert C.cost_current(markets, regions) == m * r
            assert C.cost_historical(markets, regions) == HM * m * r
            assert C.cost_current(markets + ["z"], regions) > C.cost_current(
                markets, regions)
            assert C.cost_current(markets, regions + ["z"]) > C.cost_current(
                markets, regions)


def test_no_sequence_of_requests_can_exceed_the_budget():
    """The guard, over random sequences instead of one example."""
    from coverline.execution.odds_client import QuotaExceeded

    for _ in range(80):
        budget = int(RNG.integers(1, 60))
        t = _CountingTransport()
        client, ledger = _client(budget, t)
        for _ in range(20):
            m = int(RNG.integers(1, 4))
            r = int(RNG.integers(1, 3))
            try:
                client.current_odds(sport="s", markets=["m"] * m,
                                    regions=["r"] * r, captured_at="t")
            except QuotaExceeded:
                pass
        assert ledger.spent_predicted <= budget


def test_a_refused_request_issues_nothing():
    """The exception says "Nothing was issued". This checks that it is true.

    A guard that raises AFTER the call has already gone out is worse than no
    guard, because it reports a budget that was already breached.
    """
    from coverline.execution.odds_client import QuotaExceeded

    t = _CountingTransport()
    client, _ = _client(2, t)
    with pytest.raises(QuotaExceeded):
        client.current_odds(sport="s", markets=["a", "b", "c"],
                            regions=["uk"], captured_at="t")
    assert t.urls == [], "a refused request still reached the transport"


def test_listing_sports_is_free_and_does_not_touch_the_budget():
    """The one free endpoint, and a budget of zero proves it is not charged."""
    t = _CountingTransport()
    client, ledger = _client(0, t)
    client.sports(captured_at="t")
    assert len(t.urls) == 1
    assert ledger.spent_predicted == 0
    assert ledger.remaining_reported == 400


def test_a_wrong_cost_model_raises_rather_than_accumulating():
    """The drift check, exercised by charging something other than predicted.

    This is the one guard that can catch the vendor changing its pricing, and
    it only ever fires against a real response -- so it is worth knowing it
    fires at all before a paid key exists.
    """
    from coverline.execution.odds_client import CostModelDrift

    t = _CountingTransport(charge=7)          # predicted will be 2
    client, _ = _client(50, t)
    with pytest.raises(CostModelDrift):
        client.current_odds(sport="s", markets=["a", "b"], regions=["uk"],
                            captured_at="t")

    # And it does NOT fire when the API agrees.
    t2 = _CountingTransport(charge=2)
    client2, ledger2 = _client(50, t2)
    client2.current_odds(sport="s", markets=["a", "b"], regions=["uk"],
                         captured_at="t")
    assert ledger2.spent_actual == 2


def test_history_costs_ten_times_current_at_the_client_boundary():
    """Not just in the static function -- through the call that spends it.

    A backfill is the largest single spend this project will ever make, and
    it is priced off this multiplier.
    """
    t = _CountingTransport()
    client, ledger = _client(500, t)
    client.current_odds(sport="s", markets=["h2h", "spreads"], regions=["uk"],
                        captured_at="t")
    after_current = ledger.spent_predicted
    client.historical_odds(sport="s", markets=["h2h", "spreads"],
                           regions=["uk"], date="2026-01-01T00:00:00Z",
                           captured_at="t")
    assert ledger.spent_predicted - after_current == 10 * after_current


def test_the_backfill_dry_run_predicts_exactly_what_a_run_spends():
    """The number Thursday's decision rests on.

    A backfill is the single largest spend this project will ever make -- the
    design costed one season across five leagues at roughly 92,000 credits
    against a 100,000-credit month. What stops that going wrong is a dry run
    that reports the cost before anything is issued, and a dry run is only
    worth anything if it is EXACT.

    So: plan a backfill, read its predicted total, execute the same plan
    against a fake transport with a budget large enough not to bind, and
    require the two to be equal. Not close. Equal.
    """
    import tempfile

    from coverline.execution import backfill as B
    from coverline.execution.bronze import BronzeStore

    snaps = list(B.daily_snapshots(
        sport="icehockey_nhl", start="2026-01-01", end="2026-01-05",
        times_of_day=("23:00",), markets=("h2h", "spreads"), regions=("us",)))
    p = B.plan(snaps)
    assert len(p) > 0
    predicted = p.total_cost

    t = _CountingTransport(charge=None)
    client, ledger = _client(predicted * 2, t)
    with tempfile.TemporaryDirectory() as d:
        res = B.run(p, client, BronzeStore(Path(d)))

    assert not res.stopped_early
    assert res.credits_spent == predicted, (
        f"the dry run promised {predicted} credits and the run spent "
        f"{res.credits_spent}. A dry run that is not exact is a dry run "
        "nobody can act on."
    )
    assert ledger.spent_predicted == predicted


def test_a_backfill_that_runs_out_of_budget_gaps_the_rest():
    """Absence must stay legible.

    Stopping silently leaves holes indistinguishable from timestamps the API
    never had data for, and the gap log exists precisely so that difference
    survives.
    """
    import tempfile

    from coverline.execution import backfill as B
    from coverline.execution.bronze import BronzeStore

    snaps = list(B.daily_snapshots(
        sport="icehockey_nhl", start="2026-01-01", end="2026-01-10",
        times_of_day=("23:00",), markets=("h2h", "spreads"), regions=("us",)))
    p = B.plan(snaps)
    one = p.snapshots[0].cost

    t = _CountingTransport()
    client, _ = _client(one * 3, t)      # room for three, plan has ten
    with tempfile.TemporaryDirectory() as d:
        store = BronzeStore(Path(d))
        res = B.run(p, client, store)

    assert res.stopped_early
    assert len(res.fetched) == 3
    assert len(res.gapped) == len(p) - 3, (
        "every snapshot the budget refused must be gapped, not dropped"
    )
    assert all(reason == "quota_exhausted" for _, reason in res.gapped)
