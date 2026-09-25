"""The dashboard's data: tiers, the board, the record and the gates.

The rule the whole revamp rests on is that the site computes nothing -- it
renders what these files say. So these tests hold the files to what the core
computes: every number on a card must be the one the operator command would
produce for the same game.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import export_board as X  # noqa: E402
import export_record as XR  # noqa: E402
from coverline.core import pricing as P  # noqa: E402
from coverline.core import tiers as T  # noqa: E402
from coverline.core.distributions import NormalMarginDistribution  # noqa: E402
from coverline.execution import board as B  # noqa: E402
from coverline.execution import recommend as Rc  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402

SDV = ROOT / "data" / "raw" / "sportsdataverse"
BANDS = {"coin_flip": 0.02, "lean": 0.05, "play": 0.08}


# ------------------------------------------------------------------ tiers --

@pytest.mark.parametrize("edge,want", [(0.0, "no_edge"), (0.019, "no_edge"),
                                       (0.02, "coin_flip"), (-0.06, "lean"),
                                       (0.08, "play"), (0.3, "play")])
def test_a_tier_is_the_band_the_edge_falls_in(edge, want):
    assert T.tier(edge, BANDS) == want


def test_every_league_has_ordered_bands_and_says_where_they_came_from():
    art = json.loads(T.THRESHOLDS_PATH.read_text())
    assert set(art["leagues"]) == set(X.LEAGUES)
    for league in X.LEAGUES:
        b = T.thresholds(league)
        assert b["provisional"] is True                     # nothing has 150 games yet
        assert b["source"] in ("backtest", "ledger") or b["source"].startswith("borrowed:")


def test_the_bands_reproduce_from_source_and_record_that_they_are_not_edge():
    from model import derive_tier_thresholds as D
    art = json.loads(T.THRESHOLDS_PATH.read_text())
    # MLB's inputs grow daily (odds watch + in-season cron), so the bands are
    # rebuilt from the cut-off the artifact recorded, not from today's data --
    # otherwise this failed on every cron update. Refreshing the bands is a
    # deliberate `derive_tier_thresholds.py` run, not a side effect of time.
    assert art["leagues"]["mlb"].get("through"), "the MLB bands record no cut-off"
    fresh = D.derive(mlb_through=art["leagues"]["mlb"]["through"])
    for league, row in art["leagues"].items():
        for k in ("coin_flip", "lean", "play", "source", "n"):
            assert fresh["leagues"][league][k] == row[k], (league, k)
    # The finding ADR 0025 rests on: no band is distinguishable from
    # break-even. (An earlier draft said none CLEARED it; the NFL and CFB top
    # bands sit 0.15 and 0.07 SE above it, and this test is what caught that.)
    for league in ("nfl", "cfb", "mlb"):
        for band in art["leagues"][league]["by_band"].values():
            p, n = band["preferred_side_hit_rate"], band["n"]
            z = (p - 0.5238) / np.sqrt(p * (1 - p) / n)
            assert z < 2.0, (league, band)


# ---------------------------------------------------------- one market ----

def _quotes(books, market="spreads", home_pt=-3.5, event="e1"):
    """books: {name: (home_price, away_price[, home_point])}"""
    ev = {"id": event, "sport_key": "basketball_nba", "commence_time": "2030-01-15T23:30:00Z",
          "home_team": "Boston Celtics", "away_team": "New York Knicks", "bookmakers": []}
    for b, spec in books.items():
        hp, ap = spec[0], spec[1]
        pt = spec[2] if len(spec) > 2 else home_pt
        outs = ([{"name": "Boston Celtics", "price": hp, "point": pt},
                 {"name": "New York Knicks", "price": ap, "point": -pt}] if market == "spreads"
                else [{"name": "Boston Celtics", "price": hp},
                      {"name": "New York Knicks", "price": ap}])
        ev["bookmakers"].append({"key": b, "markets": [{"key": market, "outcomes": outs}]})
    return normalize([ev], captured_at="x")


def _dist(mu=6.0):
    return NormalMarginDistribution(mu_margin=mu, sd_margin=14.0, mu_total=225.0, sd_total=18.0,
                                    integral_margin=False, total_validated=True, discrete=False)


def test_the_board_prices_the_consensus_line_with_the_median_market():
    q = _quotes({"a": (1.91, 1.91), "b": (1.95, 1.87), "c": (1.80, 2.02),
                 "odd": (1.91, 1.91, -5.5)})               # one book off the consensus
    v = B.price_market(_dist(), q, "spread", weight=0.0, bands=BANDS)
    assert v.status == "priced" and v.line == -3.5 and v.books == 3
    home = [float(P.devig([h, a], "power")[0]) for h, a in ((1.91, 1.91), (1.95, 1.87), (1.80, 2.02))]
    p_home, _ = Rc.cover_probability(_dist(), -3.5)
    assert v.side == "home"
    assert v.p_market == pytest.approx(sorted(home)[1])
    assert v.p_model == pytest.approx(p_home)
    assert v.edge == pytest.approx(v.p_model - v.p_market)
    assert v.best_book == "b" and v.best_price_decimal == 1.95
    assert v.tier == T.tier(v.edge, BANDS)


def test_the_preferred_side_can_be_the_away_side():
    v = B.price_market(_dist(mu=-2.0), _quotes({"a": (1.91, 1.91)}), "spread",
                       weight=0.0, bands=BANDS)
    assert v.side == "away" and v.line == 3.5 and v.selection == "New York Knicks"
    assert v.p_model > v.p_market


def test_an_ungraded_league_is_unsized_and_says_so():
    v = B.price_market(_dist(), _quotes({"a": (1.91, 1.91)}), "spread", weight=0.0, bands=BANDS)
    assert v.stake_fraction == 0.0
    assert v.unsized_reason == "Unsized — not yet graded against the market"


def test_a_graded_league_gets_the_stake_the_staking_engine_sizes():
    q = _quotes({"a": (1.91, 1.91)})
    v = B.price_market(_dist(), q, "spread", weight=0.5, bands=BANDS)
    c = Rc.price_candidate(dist=_dist(), quotes=q, event_id="e1", market="spreads",
                           bookmaker="a", selection="Boston Celtics", bankroll=1.0,
                           shrinkage=0.5, kelly_multiple=B.KELLY,
                           max_bankroll_fraction=B.MAX_FRACTION, min_edge=0.0)
    assert v.stake_fraction == pytest.approx(c.plan.bankroll_fraction) and v.stake_fraction > 0
    assert v.unsized_reason is None


def test_an_integer_nba_line_is_refused_not_priced():
    d = NormalMarginDistribution(mu_margin=6.0, sd_margin=14.0, mu_total=225.0, sd_total=18.0,
                                 integral_margin=True, total_validated=False, discrete=False)
    v = B.price_market(d, _quotes({"a": (1.91, 1.91)}, home_pt=-4.0), "spread",
                       weight=0.0, bands=BANDS)
    assert v.status == "refused" and "ADR 0022" in v.refusal


def test_the_open_line_is_carried_for_movement():
    v = B.price_market(_dist(), _quotes({"a": (1.91, 1.91)}), "spread", weight=0.0,
                       bands=BANDS, open_quotes=_quotes({"a": (1.91, 1.91)}, home_pt=-2.5),
                       open_at="2030-01-14T12:00:00Z")
    assert v.open_line == -2.5 and v.line == -3.5 and v.open_at == "2030-01-14T12:00:00Z"


# ------------------------------------------------------------- a board ----

@pytest.fixture()
def nba_night(tmp_path, monkeypatch):
    """Committed 2021-2022, 2023 with its last two games moved to a future
    night, a bronze store with an open and a later snapshot of them."""
    from coverline.leagues.nba import live
    for y in (2021, 2022):
        shutil.copy(SDV / f"nba_{y}.parquet", tmp_path / f"nba_{y}.parquet")
    d = pd.read_parquet(SDV / "nba_2023.parquet")
    std = d[(d.season_type == 2) & (d.type_abbreviation == "STD") & ~d.neutral_site]
    rows = std.sort_values("date").iloc[-2:]
    tips = ("2030-01-15T23:30:00Z", "2030-01-16T00:30:00Z")
    for tip, (_, r) in zip(tips, rows.iterrows()):
        m = d.id == r.id
        d.loc[m, "date"] = tip.replace(":00Z", "Z")      # ESPN's format
        d.loc[m, ["home_score", "away_score"]] = 0
        d.loc[m, "status_type_completed"] = False
        d.loc[m, "status_type_name"] = "STATUS_SCHEDULED"
    d.to_parquet(tmp_path / "nba_2023.parquet")
    src = live.NBALiveSource.load(2023, data=str(tmp_path / "nba_{year}.parquet"))

    def ev(tip, r, line):
        return {"id": f"ev{r.id}", "sport_key": "basketball_nba", "commence_time": tip,
                "home_team": r.home_display_name, "away_team": r.away_display_name,
                "bookmakers": [{"key": b, "markets": [{"key": "spreads", "outcomes": [
                    {"name": r.home_display_name, "price": 1.91, "point": line},
                    {"name": r.away_display_name, "price": 1.91, "point": -line}]}]}
                    for b in ("pinnacle", "fanduel")]}
    store = BronzeStore(tmp_path / "bronze")
    for at, line in (("2030-01-14T12:00:00Z", -1.5), ("2030-01-15T22:00:00Z", -2.5)):
        store.write_snapshot(sport="basketball_nba", captured_at=at, cost=1, source_url="u",
                             payload=[ev(t, r, line) for t, (_, r) in zip(tips, rows.iterrows())])
    R = X._runner()
    import dataclasses
    R.LEAGUES["nba"] = dataclasses.replace(
        R.LEAGUES["nba"], loader=lambda day: (live.build_model(src), src, src.slate(day)))
    return X.build("nba", store, R, day="2030-01-15"), src, rows


def test_a_board_is_the_cores_numbers_for_every_game(nba_night):
    board, src, rows = nba_night
    assert board["slate"] == {"kind": "date", "date": "2030-01-15"}
    assert board["grade"]["weight"] == 0.0
    assert board["tiers"]["provisional"] is True
    assert len(board["games"]) == 2 and not board["refusals"]
    from coverline.leagues.nba import live
    model = live.build_model(src)
    for g in board["games"]:
        m = g["markets"]["spread"]
        assert m["status"] == "priced" and m["books"] == 2
        dist = model.predict(g["game_id"], "now")
        p_home, _ = Rc.cover_probability(dist, -2.5)
        want = p_home if m["side"] == "home" else 1 - p_home
        assert m["p_model"] == pytest.approx(want, abs=1e-5)
        assert m["stake_fraction"] == 0.0 and m["unsized_reason"].startswith("Unsized")
        assert m["open_line"] in (-1.5, 1.5) and m["tier"] in T.TIERS
        assert g["model"]["p_home_win"] is not None
        assert "home_rest_days" in g["context"] and g["context"]["team_level_model"] is True
        assert g["start"].endswith("Z")
    assert board["teams"] and "rating" in board["teams"][0]


def test_a_board_is_valid_json_with_no_nan(nba_night, tmp_path):
    board, _, _ = nba_night
    text = json.dumps(X._clean(board), allow_nan=False)
    assert "NaN" not in text


def test_a_league_that_cannot_be_priced_still_gets_a_board_that_says_why(tmp_path):
    # A CFB season with no schedule pulled: refused, with the command.
    b = X.build("cfb", BronzeStore(tmp_path / "empty"), X._runner(), day="2098-09-20")
    assert b["games"] == [] and "cfb_espn.py --seasons 2098-2098" in b["refusals"][0]["reason"]
    assert X.summarise(b)["state"] == "refused"
    assert json.dumps(X._clean(b), allow_nan=False)


def test_no_odds_yet_still_shows_the_models_view(tmp_path, monkeypatch, nba_night):
    board, src, rows = nba_night
    import dataclasses
    from coverline.leagues.nba import live
    R = X._runner()
    R.LEAGUES["nba"] = dataclasses.replace(
        R.LEAGUES["nba"], loader=lambda day: (live.build_model(src), src, src.slate(day)))
    b = X.build("nba", BronzeStore(tmp_path / "empty"), R, day="2030-01-15")
    assert "No odds captured yet" in b["refusals"][0]["reason"]
    assert len(b["games"]) == 2 and all(g["model"]["p_home_win"] for g in b["games"])
    assert all(g["markets"] == {} for g in b["games"])


# ------------------------------------------------------- record, gates ----

def test_the_record_counts_one_settled_game_per_event(tmp_path):
    from tests.core.test_market_weight import _paper_ledger
    led = _paper_ledger(tmp_path, 40, 1.0)
    rec = XR.record(led)
    r = rec["leagues"]["nba"]
    assert 0 < r["settled_games"] <= 40 and rec["floor"] == 150
    assert r["progress_to_floor"] == pytest.approx(r["settled_games"] / 150, abs=1e-4)
    assert r["preferred_side_wins"] <= r["settled_games"]
    assert rec["leagues"]["nfl"]["settled_games"] == 0


def test_the_gates_are_read_from_the_artifacts_not_written_by_hand():
    g = XR.gates()
    for row in g["fitted"]:
        art = json.loads((ROOT / row["file"]).read_text())
        assert row["passed"] == bool(art["holdout_grade"]["supported"])
        assert row["t"] == art["holdout_grade"]["t"]
    ids = {d["id"] for d in g["decisions"]}
    on_disk = {p.name[:4] for p in (ROOT / "docs" / "decisions").glob("[0-9]*.md")} - {"0000"}
    assert ids == on_disk
    assert {m["league"] for m in g["market"]} == set(X.LEAGUES)


# ------------------------------------------------ the regime cap, status ----

def _entry(side="home", tier="play", stake=0.01):
    return {"home": "ATL", "away": "TB", "context": {}, "markets": {"spread": {
        "status": "priced", "side": side, "tier": tier, "stake_fraction": stake}}}


def test_a_flag_backing_a_first_year_staff_is_capped_not_removed():
    regimes = {"ATL": {"tier": 1, "coach": "Stefanski"}}
    e = _entry()
    X.apply_regime_cap(e, "spread", regimes, week=3)
    m = e["markets"]["spread"]
    assert m["tier"] == "lean" and m["stake_fraction"] == 0.005
    assert f"went {X.regime_evidence()} in backtests" in m["cap"]["reason"]
    assert e["context"]["regime"]["home"]["coach"] == "Stefanski"


def test_fading_a_regime_team_and_late_season_flags_are_untouched():
    regimes = {"ATL": {"tier": 1, "coach": "Stefanski"}}
    e = _entry(side="away")
    X.apply_regime_cap(e, "spread", regimes, week=3)
    assert e["markets"]["spread"]["tier"] == "play" and "cap" not in e["markets"]["spread"]
    e = _entry()
    X.apply_regime_cap(e, "spread", regimes, week=9)
    assert e["markets"]["spread"]["tier"] == "play" and "regime" not in e["context"]


def test_the_regime_number_is_the_artifacts():
    art = json.loads((ROOT / "model" / "coach_regime_results.json").read_text())
    g = next(g for g in art["grades"]
             if g["label"] == "model BACKED the regime team" and g["min_edge"] == 2.5)
    assert X.regime_evidence() == f"{g['wins']}/{g['n_graded']}"


@pytest.mark.parametrize("games,refusals,want", [
    ([], [{"scope": "league", "reason": "r"}], "refused"),
    ([], [], "idle"),
    (["priced"], [], "up"),
    (["priced", "none"], [], "degraded"),
    (["none"], [{"scope": "league", "reason": "no odds"}], "degraded"),
])
def test_the_circuit_state_comes_from_the_export(games, refusals, want):
    b = {"games": [{"headline": "spread", "markets": (
            {"spread": {"status": "priced", "tier": "lean"}} if g == "priced" else {})} for g in games],
         "refusals": refusals}
    s = X.summarise(b)
    assert s["state"] == want
    assert s["priced"] == games.count("priced")


def test_an_idle_league_says_when_it_resumes():
    for league in ("nba", "nhl"):
        b = json.loads((ROOT / "data" / "site" / f"board_{league}.json").read_text())
        if b["status"]["state"] == "idle":
            assert b.get("next_slate") and b["next_slate"]["games"] > 0
