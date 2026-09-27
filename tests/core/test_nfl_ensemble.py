"""The full ensemble on the live board (accuracy plan 1a), offline.

model/nfl_ensemble_parity.py runs the gate with the network and writes
data/nfl_ensemble_parity.json. This file checks, with no network:

  - the source sets ngs_present per game from what was fetched, degrades to
    the rating-only vector when a feed fails, and never lets Elo see a result
    from the priced game's day or later;
  - the de-bias port equals the legacy function on every committed week-3
    board, and the board's market margin is the legacy median;
  - the board uses the ensemble only while the gate artifact says passed,
    and the tier holds Pedro has not decided on are off.
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from coverline.execution import board as B  # noqa: E402
from coverline.execution.normalize import Quote  # noqa: E402
from coverline.leagues.nfl import ensemble as EN  # noqa: E402
from coverline.leagues.nfl.model import GameFeatures, NFLModel, predict_margin  # noqa: E402

SCHEDULE = ROOT / "tests" / "fixtures" / "nfl_2026_week03_parity_schedule.csv"
PARITY = ROOT / "data" / "nfl_ensemble_parity.json"
ASOF = "2026-09-27T12:00:00Z"
NGS_COLS = ("team_cpoe", "team_avg_separation", "team_yac_over_expected", "team_ryoe")


@pytest.fixture(scope="module")
def schedule():
    return pd.read_csv(SCHEDULE)


def _ngs(teams, value=lambda i: float(i)):
    return pd.DataFrame({c: [value(i) for i, _ in enumerate(teams)] for c in NGS_COLS},
                        index=list(teams))


def _history(schedule):
    """Two finished week-2 results and GB/ATL's week-3 result."""
    rows = [
        {"season": 2026, "week": 2, "gameday": "2026-09-20", "home_team": "GB",
         "away_team": "DEN", "home_score": 30, "away_score": 10},
        {"season": 2026, "week": 2, "gameday": "2026-09-20", "home_team": "LA",
         "away_team": "ATL", "home_score": 20, "away_score": 17},
        {"season": 2026, "week": 3, "gameday": "2026-09-24", "home_team": "GB",
         "away_team": "ATL", "home_score": 3, "away_score": 40},
    ]
    return pd.DataFrame(rows)


def _source(schedule, ngs=None, history=None, ngs_raises=False):
    def ngs_loader(s, w):
        if ngs_raises:
            raise ConnectionError("404 from nflverse")
        return ngs if ngs is not None else _ngs(sorted(set(schedule.home_team)
                                                       | set(schedule.away_team)))
    return EN.EnsembleWeekSource.load(
        2026, 3, schedule, ratings_dir=ROOT / "data" / "ratings",
        ngs_loader=ngs_loader,
        history_loader=lambda s: history if history is not None else _history(schedule))


def test_full_ensemble_when_both_feeds_have_both_teams(schedule):
    src = _source(schedule)
    f, cset, why = src.resolve("2026-W03-DEN-LA", ASOF)
    assert (cset, why, f.ngs_present) == (EN.FULL, None, True)
    assert f.elo_diff != 0.0


def test_a_team_missing_from_ngs_prices_that_game_rating_only(schedule):
    teams = sorted((set(schedule.home_team) | set(schedule.away_team)) - {"LA"})
    src = _source(schedule, ngs=_ngs(teams))
    f, cset, why = src.resolve("2026-W03-DEN-LA", ASOF)
    assert (cset, f.ngs_present) == (EN.RATING_ONLY, False)
    assert "LA" in why
    assert src.resolve("2026-W03-GB-ATL", ASOF)[1] == EN.FULL


def test_ngs_outage_degrades_every_game_and_says_so(schedule):
    src = _source(schedule, ngs_raises=True)
    assert src.status["ngs"] is False and "404" in src.status["ngs_error"]
    for gid in src.game_ids():
        f, cset, why = src.resolve(gid, ASOF)
        assert (cset, f.ngs_present, why) == (EN.RATING_ONLY, False, "NGS feed unavailable")


def test_rating_only_features_are_exactly_the_old_sources(schedule):
    """Degrading must land on the board as it was, not on a variant of it."""
    src = _source(schedule, ngs_raises=True)
    for gid in src.game_ids():
        assert src.features(gid, ASOF) == src.base.features(gid, ASOF)


def test_elo_never_sees_the_priced_games_own_result(schedule):
    """GB/ATL's week-3 result is in the history; pricing that game must not
    use it, and the game-day cut is what stops it."""
    src = _source(schedule)
    elo = src._elo("2026-09-24", ASOF)
    blind = EN.elo_before(_history(schedule).iloc[:2], "2026-09-24")
    assert elo == blind


def test_elo_is_cut_at_asof_too(schedule):
    src = _source(schedule)
    assert src._elo("2026-09-28", "2026-09-20T09:00:00Z") == {}


def test_provenance_carries_the_boards_feature_value_shape(schedule):
    from test_board_provenance import REQUIRED_FIELDS
    p = _source(schedule).provenance("2026-W03-DEN-LA", ASOF)
    assert set(p["feature_values"]) == REQUIRED_FIELDS
    assert p["coefficient_set"] == EN.FULL


# ---------------------------------------------------------- de-bias port ----

def _week3_boards():
    return sorted(glob.glob(str(ROOT / "data" / "divergence" / "2026-week-03-*.json")))


def test_slate_debias_equals_the_legacy_function_on_every_week3_board():
    from deploy.odds_watch_job import inseason_offsets
    boards = _week3_boards()
    assert len(boards) >= 10
    for p in boards:
        b = json.loads(Path(p).read_text())
        pairs = []
        for d in b["divergences"]:
            if d.get("line_status") == "closed":
                continue
            fv = d["feature_values"]
            f = GameFeatures(**{k: fv[k] for k in (
                "rating_diff", "rest_diff", "cpoe_diff", "separation_diff",
                "yac_oe_diff", "ryoe_diff", "is_neutral_site")},
                elo_diff=fv["elo_diff"] or 0.0,
                ngs_present=d["coefficient_set"] == EN.FULL)
            pairs.append((predict_margin(f), d["market_spread"]))
        legacy, _ = inseason_offsets([{"spread_gap": m - k} for m, k in pairs])
        core, _ = B.slate_debias(pairs)
        assert core == pytest.approx(legacy, abs=1e-12), Path(p).name
        assert core == pytest.approx(b["debias_offsets"][0], abs=1e-9), Path(p).name


def test_slate_debias_falls_back_to_the_prior_then_to_zero():
    few = [(1.0, 3.0)] * 7
    assert B.slate_debias(few, prior=-0.8)[0] == -0.8
    assert B.slate_debias(few)[0] == 0.0
    assert B.slate_debias(few + [(float("nan"), 1.0)], prior=None)[0] == 0.0
    assert B.slate_debias([(0.0, 2.0)] * 8)[0] == 2.0


def _q(outcome, point, book="b1"):
    return Quote("e1", "americanfootball_nfl", "2026-09-27T17:00:00Z", "Denver Broncos",
                 "Los Angeles Rams", book, "spreads", outcome, 1.91, point, None,
                 "2026-09-27T12:00:00Z")


def test_market_margin_is_the_median_home_spread_home_positive():
    qs = [_q("Denver Broncos", -2.5, "a"), _q("Denver Broncos", -3.0, "b"),
          _q("Denver Broncos", -3.5, "c"), _q("Los Angeles Rams", 3.0, "a")]
    assert B.market_home_margin(qs) == 3.0
    assert B.market_home_margin([_q("Los Angeles Rams", 3.0)]) is None


def test_the_offset_is_added_by_the_model_the_board_builds(schedule):
    src = _source(schedule)
    base = NFLModel(src).predict("2026-W03-DEN-LA", ASOF).mu_margin
    shifted = NFLModel(src, margin_offset=-0.33).predict("2026-W03-DEN-LA", ASOF).mu_margin
    assert shifted - base == pytest.approx(-0.33)


# ------------------------------------------------------ the gate, wired ----

def test_the_parity_artifact_passed_within_tolerance():
    art = json.loads(PARITY.read_text())
    assert art["passed"] is True and art["tolerance_points"] == 0.05
    assert art["A_recorded"]["rows"] >= 160
    assert art["A_recorded"]["max_abs_diff"] <= 0.05
    lv = art["B_fetched"]["legacy_vocabulary"]
    assert lv["passed"] and len(lv["boards"]) == 2
    for b in lv["boards"].values():
        assert not b["coefficient_set_mismatches"]
        for g in b["games"]:
            assert all(abs(v) < 1e-6 for v in g["term_diffs"].values()), g["game"]


def test_the_only_vector_change_under_the_core_vocabulary_is_the_rams():
    art = json.loads(PARITY.read_text())
    for b in art["B_fetched"]["core_vocabulary"]["boards"].values():
        assert all("LA" in g.split("@") for g in b["coefficient_set_mismatches"])


def test_the_board_uses_the_ensemble_only_while_the_gate_passes(tmp_path):
    import export_board as X
    assert X.nfl_ensemble_enabled() is True
    assert X.nfl_ensemble_enabled(tmp_path / "missing.json") is False
    (tmp_path / "p.json").write_text(json.dumps({"passed": False}))
    assert X.nfl_ensemble_enabled(tmp_path / "p.json") is False


def test_the_holds_pedro_has_not_decided_are_off():
    import export_board as X
    assert X.HOLD_NFL_TIERS_ON_RATING_ONLY is False
    assert X.HOLD_CFB_TIERS_EARLY is False


def test_a_tier_hold_reduces_and_never_removes():
    import export_board as X
    e = {"markets": {"spread": {"status": "priced", "tier": "play", "stake_fraction": 0.01,
                                "p_model": 0.6},
                     "moneyline": {"status": "refused", "tier": None}}}
    X.apply_tier_hold(e, "rating_only", "why")
    m = e["markets"]["spread"]
    assert (m["tier"], m["stake_fraction"], m["cap"]["rule"], m["p_model"]) == \
        ("coin_flip", 0.0, "rating_only", 0.6)
    assert e["markets"]["moneyline"] == {"status": "refused", "tier": None}


def test_the_banner_is_the_briefs_wording():
    import export_board as X
    assert "takes the underdog about 80% of the time" in X.NFL_RATING_ONLY_BANNER
    assert "went 47% on those picks" in X.NFL_RATING_ONLY_BANNER
