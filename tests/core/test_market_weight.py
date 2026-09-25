"""The market-relative staking weight: estimator, artifact, and its use.

The estimator is checked on data where the right answer is known, because a
weight that always came back zero would look exactly like the NFL result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core import market_weight as MW  # noqa: E402


def _synthetic(true_w, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    pk = rng.uniform(0.3, 0.7, n)
    pm = np.clip(pk + rng.normal(0, 0.06, n), 0.02, 0.98)
    q = np.clip(true_w * pm + (1 - true_w) * pk, 0.01, 0.99)
    return pm, pk, (rng.uniform(size=n) < q).astype(int)


@pytest.mark.parametrize("true_w", [0.0, 0.5, 1.0, -0.5])
def test_the_estimator_recovers_a_known_weight(true_w):
    r = MW.estimate(*_synthetic(true_w))
    assert abs(r.w_hat - true_w) < 3 * r.se
    assert r.se < 0.2


def test_a_model_with_real_information_gets_a_positive_stake():
    r = MW.estimate(*_synthetic(1.0))
    assert 0.5 < r.staking_weight <= 1.0
    assert r.staking_weight == pytest.approx(r.w_hat - MW.Z_ONE_SIDED * r.se)


@pytest.mark.parametrize("true_w", [0.0, -0.5])
def test_noise_or_worse_stakes_nothing(true_w):
    assert MW.estimate(*_synthetic(true_w)).staking_weight == 0.0


def test_the_maximum_is_the_maximum():
    pm, pk, y = _synthetic(0.4, seed=3)
    r = MW.estimate(pm, pk, y)

    def ll(w):
        q = pk + w * (pm - pk)
        return np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    assert ll(r.w_hat) >= max(ll(r.w_hat - 0.01), ll(r.w_hat + 0.01))


def test_bad_input_is_refused():
    pm, pk, y = _synthetic(0.0, n=100)
    with pytest.raises(ValueError, match="30"):
        MW.estimate(pm[:10], pk[:10], y[:10])
    with pytest.raises(ValueError, match="pushes"):
        MW.estimate(pm, pk, np.where(y == 1, 0.5, 0))
    with pytest.raises(ValueError, match="inside"):
        MW.estimate(np.where(pm > 0.5, 1.0, pm), pk, y)


# ----------------------------------------------------------- the artifact --

def test_a_league_with_no_grade_stakes_nothing(tmp_path):
    art = json.loads(MW.WEIGHTS_PATH.read_text())
    art["leagues"].pop("nba")                  # every league has a grade since 2026-09-25
    p = tmp_path / "w.json"
    p.write_text(json.dumps(art))
    assert MW.staking_weight("nba", p) == (0.0, None)
    assert MW.staking_weight("nfl", tmp_path / "absent.json") == (0.0, None)


def test_a_hand_edited_weight_is_refused(tmp_path):
    art = json.loads(MW.WEIGHTS_PATH.read_text())
    art["leagues"]["nfl"]["staking_weight"] = 0.9
    p = tmp_path / "w.json"
    p.write_text(json.dumps(art))
    with pytest.raises(ValueError, match="lower bound"):
        MW.staking_weight("nfl", p)


def test_the_committed_weights_reproduce_from_source():
    """The artifact is regenerated here and compared, so a weight nobody can
    rebuild cannot be what sizes a bet."""
    from model import grade_market_weight as G
    art = json.loads(MW.WEIGHTS_PATH.read_text())
    fresh_all = {}
    for name in ("nfl", "cfb", "mlb", "nhl", "nba"):
        hist = None
        got = G.historical_rows(name)             # kept rows for the ESPN leagues
        if got is not None and len(got[0]) >= 100:
            rows, meta = got
            hist = {**meta, **G.grade(rows)}
        g = G.choose(name, hist, G.ledger_rows(G.LEDGER_DIR, name))
        if g is not None:
            fresh_all[name] = g
    assert set(art["leagues"]) == set(fresh_all), (
        "the leagues with a grade have changed -- most likely a league's paper "
        "ledger has passed the grading floor. That is the moment ADR 0024 is "
        "for: run `scripts/settle_ledger.py --grade`, read the result, and "
        "commit it deliberately.")
    for name, fresh in fresh_all.items():
        assert fresh["source"] == art["leagues"][name]["source"], name
        for k in ("n", "w_hat", "se", "staking_weight"):
            assert fresh[k] == pytest.approx(art["leagues"][name][k], abs=1e-6), (name, k)


def test_football_has_no_measured_edge_over_the_close():
    """The finding this module exists for, pinned so a change to it is a
    deliberate event. If a refit moves either league off zero, this fails and
    the ADR gets revisited -- it does not quietly start staking."""
    for league in ("nfl", "cfb"):
        w, grade = MW.staking_weight(league)
        assert grade is not None and w == 0.0, league
        assert grade["contamination"].startswith("UPPER BOUND")


# ------------------------------------------------------ the paper ledger --

def _paper_ledger(tmp_path, n_games, true_w, seed=1):
    """A ledger as the runner and settler would write it: both sides of a
    spread at two books per game, settled."""
    from coverline.execution.ledger import BetLedger, Outcome, Signal
    rng = np.random.default_rng(seed)
    led = BetLedger(tmp_path / "ledger")
    for i in range(n_games):
        pk = rng.uniform(0.4, 0.6)
        pm = float(np.clip(pk + rng.normal(0, 0.08), 0.05, 0.95))
        home_wins = rng.uniform() < true_w * pm + (1 - true_w) * pk
        for book, jitter in (("pinnacle", 0.0), ("fanduel", 0.01)):
            for side, p_m, p_k in (("home", pm, pk + jitter), ("away", 1 - pm, 1 - pk - jitter)):
                sid = f"g{i}|{book}|{side}"
                led.record(Signal(signal_id=sid, at=f"2026-10-{1 + i % 28:02d}T12:00:00Z",
                                  league="nba", event_id=f"g{i}", market="spreads",
                                  selection=side, line=-1.5 if side == "home" else 1.5,
                                  p_model=p_m, p_market=p_k, p_used=p_k, shrinkage=0.0,
                                  edge_claimed=0.0, edge_used=0.0,
                                  disposition="not_placed",
                                  not_placed_reason="below_threshold",
                                  game_id=f"g{i}", side=side, book=book,
                                  price_decimal=1.91))
                won = home_wins if side == "home" else not home_wins
                led.record_outcome(Outcome(sid, "t", "win" if won else "loss", 0, 0))
    return tmp_path / "ledger"


def test_the_ledger_counts_one_row_per_game(tmp_path):
    from model import grade_market_weight as G
    rows = G.ledger_rows(_paper_ledger(tmp_path, 40, 0.0), "nba")
    assert len(rows) == 40                 # not 160: two books x two sides


def test_a_ledger_that_shows_real_information_earns_a_stake(tmp_path):
    from model import grade_market_weight as G
    rows = G.ledger_rows(_paper_ledger(tmp_path, 1500, 1.0), "nba")
    g = G.choose("nba", None, rows)
    assert g["source"] == "ledger" and g["staking_weight"] > 0.3


def test_a_small_ledger_does_not_replace_or_create_a_grade(tmp_path):
    from model import grade_market_weight as G
    rows = G.ledger_rows(_paper_ledger(tmp_path, 100, 1.0), "nba")
    assert G.choose("nba", None, rows) is None
    hist = {"n": 900, "w_hat": 0.0, "se": 0.1, "staking_weight": 0.0}
    assert G.choose("nba", hist, rows)["source"] == "historical"



def test_the_kept_espn_rows_are_what_the_live_models_say():
    """The NHL, NBA and MLB grades are computed from kept per-game rows (a
    full replay takes minutes). Re-price a sample of them through the same
    loaders' live paths: a row that no longer reproduces means the model or
    its inputs changed and the grade must be rebuilt."""
    import numpy as np
    from model import grade_market_weight as G
    for name in G.ESPN_LEAGUES:
        f = G.ROWS_DIR / f"{name}_rows.parquet"
        if not f.exists():
            continue
        kept = pd.read_parquet(f)
        sample = kept.sample(n=min(8, len(kept)), random_state=0)
        closes = G.espn_closes(name)
        closes = closes[closes.espn_id.isin(sample.espn_id)]
        seasons = sorted(closes.season.unique())
        saved = G.ESPN_SEASONS[name]
        try:
            G.ESPN_SEASONS[name] = (min(seasons), max(seasons))
            orig = G.espn_closes
            G.espn_closes = lambda lg, _c=closes: _c
            fresh, _ = G.LEAGUES[name]()
        finally:
            G.espn_closes = orig
            G.ESPN_SEASONS[name] = saved
        both = sample.merge(fresh, on="espn_id", suffixes=("", "_fresh"))
        assert len(both) == len(sample), f"{name}: kept rows no longer price"
        assert np.allclose(both.p_model, both.p_model_fresh, atol=1e-9), name
        assert np.allclose(both.p_market, both.p_market_fresh, atol=1e-12), name
