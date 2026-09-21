"""The point-in-time guard that core/interfaces.py has been citing all along.

WHY THIS FILE EXISTS
`LeagueModel.predict`'s docstring says:

    `asof` is an ISO-8601 timestamp and is not decorative: it is the
    point-in-time contract. An implementation that reads any datum whose
    availability timestamp is later than `asof` is leaking, and the two-run
    guard test in tests/core/test_point_in_time.py is designed to catch
    exactly that.

That file did not exist. The most important contract in the seam was cited by
name, in the most widely read module in the repository, and enforced by
nothing.

WHAT A GUARD AT THIS LEVEL CAN AND CANNOT DO
A model forwards `asof` to its feature source, so the leak itself lives in the
source, not in the model. What is checkable here is the part the model is
responsible for, and both halves matter:

  1. `asof` REACHES the source, unchanged. A model that drops it, rounds it,
     or substitutes "now" has broken the contract before any source has a
     chance to honour it.
  2. `predict` is a PURE FUNCTION of (game_id, asof). Same inputs, same
     distribution, in any order, however many times. A model that accumulated
     state between calls -- a rating updated in place, a cache keyed on the
     game and not the timestamp -- would answer differently depending on what
     was predicted before it, which is leakage by a slower route.

Property 2 is the one with teeth, and it is oracle-free: it compares the
model to itself rather than to a number somebody worked out the same way.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EARLY = "2026-01-01T00:00:00Z"
LATE = "2026-09-21T18:00:00Z"


class RecordingSource:
    """A feature source that remembers what it was asked, and when.

    Returns features that DEPEND on asof, so a model ignoring the timestamp
    produces identical distributions for different points in time and is
    caught rather than flattered.
    """

    def __init__(self, make, scale_by_asof=True):
        self._make = make
        self._scale = scale_by_asof
        self.calls: list[tuple[str, str]] = []

    def features(self, game_id: str, asof: str):
        self.calls.append((game_id, asof))
        factor = 1.0 if (not self._scale or asof == EARLY) else 2.0
        return self._make(factor)


def _nfl():
    from coverline.leagues.nfl import model as m
    return m.NFLModel, (lambda f: m.GameFeatures(rating_diff=0.08 * f,
                                                 ngs_present=False)), {}


def _cfb():
    from coverline.leagues.cfb import model as m
    return m.CFBModel, (lambda f: m.GameFeatures(rating_diff=0.05 * f,
                                                 elo_present=False)), {}


def _mlb():
    from coverline.leagues.mlb import model as m
    return m.MLBModel, (lambda f: m.GameFeatures(exp_home=4.4 * f,
                                                 exp_away=4.1)), {}


def _nhl():
    from coverline.leagues.nhl import model as m
    return m.NHLModel, (lambda f: m.GameFeatures(lam_home=3.1 * f,
                                                 lam_away=2.9)), {}


def _nba():
    from coverline.leagues.nba import model as m
    return (m.NBAModel, (lambda f: m.GameFeatures(0.2 * f, 99.0)),
            {"sigma": lambda mu: 14.1666,
             "margin_model": lambda g: 45.0 * g.rating_diff})


BUILDERS = {"nfl": _nfl, "cfb": _cfb, "mlb": _mlb, "nhl": _nhl, "nba": _nba}


def _model(league: str, scale_by_asof: bool = True):
    cls, make, kwargs = BUILDERS[league]()
    src = RecordingSource(make, scale_by_asof=scale_by_asof)
    return cls(src, **kwargs), src


def _fingerprint(dist) -> tuple:
    """Enough of a distribution to notice it changed, without needing totals.

    Totals are refused by three leagues, so this reads the margin only --
    which is the quantity every league models and prices from.
    """
    return (round(dist.margin_mean(), 10), round(dist.margin_sd(), 10),
            tuple(round(dist.margin_cdf(x), 10) for x in (-7, -3, 0, 3, 7)))


@pytest.mark.parametrize("league", sorted(BUILDERS))
def test_the_asof_reaches_the_feature_source_unchanged(league):
    """Dropped, rounded, or replaced with "now" all fail here."""
    model, src = _model(league)
    model.predict("g1", LATE)
    assert src.calls == [("g1", LATE)], (
        f"{league}: the source was asked {src.calls}, not [('g1', {LATE!r})]"
    )


def check_purity(build) -> None:
    """The assertion itself, so it can be aimed at a leaky model too.

    A guard that has only ever been pointed at code that passes is a guard
    nobody has tested. `build()` returns a fresh predictor each call, because
    order independence needs two independent runs to compare.
    """
    a = build()
    early_first = _fingerprint(a.predict("g1", EARLY))
    late_after = _fingerprint(a.predict("g1", LATE))

    b = build()
    late_first = _fingerprint(b.predict("g1", LATE))
    early_after = _fingerprint(b.predict("g1", EARLY))

    assert early_first == early_after, (
        f"predicting at {EARLY} gave a different answer depending on whether "
        "a later timestamp had been predicted first")
    assert late_after == late_first, (
        f"predicting at {LATE} gave a different answer depending on what came "
        "before it")
    assert _fingerprint(a.predict("g1", EARLY)) == early_first, (
        "repeating a prediction changed it, so predict has side effects")


@pytest.mark.parametrize("league", sorted(BUILDERS))
def test_predict_is_a_pure_function_of_game_and_asof(league):
    """Same inputs, same answer, in any order and however many times.

    A model that accumulated state between calls -- a rating updated in
    place, a cache keyed on the game and not the timestamp -- would answer
    differently depending on what was predicted before it. That is leakage
    by a slower route.
    """
    check_purity(lambda: _model(league)[0])


@pytest.mark.parametrize("league", sorted(BUILDERS))
def test_a_different_asof_actually_changes_the_answer(league):
    """The other half: a model that ignores asof passes purity trivially.

    The source here returns different features at the two timestamps, so a
    model that forwarded the call but discarded the result -- or that cached
    on game_id alone -- would produce identical distributions and be caught.
    """
    model, _ = _model(league)
    assert _fingerprint(model.predict("g1", EARLY)) != _fingerprint(
        model.predict("g1", LATE)), (
        f"{league}: the distribution is identical at two very different "
        "points in time, so asof is reaching the source and being ignored"
    )


class _LeakyCache:
    """Memoises on game_id alone -- the easiest version of the mistake.

    It looks like a performance improvement and it silently answers a later
    question with an earlier answer, or the reverse, depending on which call
    happened to come first.
    """

    def __init__(self, inner):
        self._inner = inner
        self._seen: dict[str, object] = {}

    def predict(self, game_id, asof):
        if game_id not in self._seen:
            self._seen[game_id] = self._inner.predict(game_id, asof)
        return self._seen[game_id]


class _AccumulatingModel:
    """Updates a rating in place on every prediction.

    A walk-forward that forgot it was inside predict() looks exactly like
    this, and unlike the cache it fails the REPEAT check rather than the
    order check.
    """

    def __init__(self, inner):
        self._inner = inner
        self._drift = 0.0

    def predict(self, game_id, asof):
        self._drift += 0.25
        d = self._inner.predict(game_id, asof)
        object.__setattr__(d, "mu_margin",
                           getattr(d, "mu_margin", 0.0) + self._drift)
        return d


@pytest.mark.parametrize("league", ["nfl", "cfb", "nba"])
def test_the_guard_rejects_a_model_that_caches_on_the_game_alone(league):
    """Break it deliberately, the way every guard in this repo is checked.

    The SAME assertion the real leagues pass is aimed at a leaky wrapper and
    must fail. Without this the purity test could be vacuous and nobody
    would know.
    """
    with pytest.raises(AssertionError, match="depending on"):
        check_purity(lambda: _LeakyCache(_model(league)[0]))


@pytest.mark.parametrize("league", ["nfl", "cfb"])
def test_the_guard_rejects_a_model_that_accumulates_state(league):
    """The other failure mode: order is fine, repetition is not."""
    with pytest.raises(AssertionError, match="side effects|depending on"):
        check_purity(lambda: _AccumulatingModel(_model(league)[0]))


def test_every_league_is_covered_here():
    """A league added without a point-in-time guard is a league unguarded.

    Reads the same sets the conformance battery uses, so a sixth league
    cannot be declared implemented without also being guarded here.
    """
    from tests.core.test_conformance import (
        BUILT_LEAGUES, IMPLEMENTED_LEAGUES, STRUCTURAL_ONLY_LEAGUES,
    )

    known = (set(IMPLEMENTED_LEAGUES) | set(BUILT_LEAGUES)
             | set(STRUCTURAL_ONLY_LEAGUES))
    assert known, "no league sets found"
    missing = sorted(known - set(BUILDERS))
    assert not missing, (
        f"leagues with no point-in-time guard: {missing}. The contract in "
        "core/interfaces.py applies to every league, not the ones that "
        "happened to be written first."
    )
