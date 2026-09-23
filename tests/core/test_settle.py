"""Paper trades are graded like real ones: a close, an outcome, a CLV.

Every league paper-trades (ADR 0024), and the grade that could change that is
built from these rows. A paper row that cannot be closed or settled is a row
that can never earn a stake -- which is exactly what the ledger used to
produce.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.core import pricing as P  # noqa: E402
from coverline.core.distributions import NormalMarginDistribution  # noqa: E402
from coverline.execution import grade as G  # noqa: E402
from coverline.execution import recommend as Rc  # noqa: E402
from coverline.execution import settle as S  # noqa: E402
from coverline.execution.bronze import BronzeStore  # noqa: E402
from coverline.execution.ledger import BetLedger, Outcome, Signal  # noqa: E402
from coverline.execution.normalize import normalize  # noqa: E402

KICKOFF = "2026-10-21T23:30:00Z"
J110 = P.american_to_decimal(-110)


def _payload(point=-3.5, home_price=J110, away_price=J110):
    return [{"id": "e1", "sport_key": "basketball_nba", "home_team": "Boston Celtics",
             "away_team": "New York Knicks", "commence_time": KICKOFF,
             "bookmakers": [{"key": "pinnacle", "markets": [
                 {"key": "spreads", "outcomes": [
                     {"name": "Boston Celtics", "price": home_price, "point": point},
                     {"name": "New York Knicks", "price": away_price, "point": -point}]},
                 {"key": "totals", "outcomes": [
                     {"name": "Over", "price": J110, "point": 224.5},
                     {"name": "Under", "price": J110, "point": 224.5}]}]}]}]


def _paper(led, market="spread", mu=2.0):
    dist = NormalMarginDistribution(mu_margin=mu, sd_margin=14.0, mu_total=225.0,
                                    sd_total=18.0, integral_margin=False,
                                    total_validated=True, discrete=False)
    quotes = normalize(_payload(), captured_at="2026-10-21T15:00:00Z")
    return Rc.recommend(dist=dist, quotes=quotes, event_id="e1", market=market,
                        league="nba", bankroll=1000, shrinkage=0.0, ledger=led,
                        game_id="401", kelly_multiple=0.25,
                        max_bankroll_fraction=0.02, min_edge=0.0)


@pytest.fixture
def led(tmp_path):
    return BetLedger(tmp_path / "ledger")


def test_a_paper_signal_carries_everything_grading_needs(led):
    sigs = _paper(led)
    assert len(sigs) == 2 and not any(s.placed for s in sigs)
    by_side = {s.side: s for s in sigs}
    assert set(by_side) == {"home", "away"}
    for s in sigs:
        assert s.game_id == "401" and s.commence_time == KICKOFF
        assert s.book == "pinnacle" and s.price_decimal == pytest.approx(J110)
        assert s.stake is None
    assert by_side["home"].line == -3.5 and by_side["away"].line == 3.5


@pytest.mark.parametrize("side,line,home,away,want", [
    ("home", -3.5, 110, 104, "win"),    # won by 6
    ("home", -3.5, 106, 104, "loss"),   # won by 2
    ("away", 3.5, 106, 104, "win"),
    ("home", -3.0, 105, 102, "push"),
    ("home", None, 3, 3, "push"),       # moneyline tie
    ("away", None, 2, 3, "win"),
    ("over", 224.5, 120, 110, "win"),
    ("under", 224.5, 120, 110, "loss"),
])
def test_results_follow_the_side_and_its_own_line(side, line, home, away, want):
    sig = Signal(signal_id="x", at="t", league="nba", event_id="e", market="m",
                 selection="s", line=line, p_model=0.5, p_market=0.5, p_used=0.5,
                 shrinkage=0.0, edge_claimed=0.0, edge_used=0.0,
                 disposition="not_placed", not_placed_reason="below_threshold",
                 side=side)
    assert S.result_for(sig, home, away) == want


def test_settle_writes_one_outcome_per_signal_and_never_twice(led):
    _paper(led)
    rep = S.settle(led, {"401": (110, 104)}, "nba")
    assert len(rep.settled) == 2
    res = {led_s.side: o.result for o in led.outcomes()
           for led_s in led.signals() if led_s.signal_id == o.signal_id}
    assert res == {"home": "win", "away": "loss"}
    again = S.settle(led, {"401": (110, 104)}, "nba")
    assert again.settled == [] and again.already == 2
    with pytest.raises(ValueError, match="already has an outcome"):
        led.record_outcome(Outcome(led.signals()[0].signal_id, "t", "win", 1, 0))


def test_a_game_not_final_is_pending_and_an_old_row_is_unsettleable(led):
    _paper(led)
    led.record(Signal(signal_id="old", at="t", league="nba", event_id="e1",
                      market="spreads", selection="Boston Celtics", line=-3.5,
                      p_model=0.5, p_market=0.5, p_used=0.5, shrinkage=0.0,
                      edge_claimed=0.0, edge_used=0.0, disposition="not_placed",
                      not_placed_reason="below_threshold"))
    rep = S.settle(led, {}, "nba")
    assert rep.pending == 2 and rep.unsettleable == 1 and rep.settled == []


def test_paper_signals_get_a_close_and_a_paper_clv(led, tmp_path):
    _paper(led, mu=6.0)              # model likes the home side at -3.5
    store = BronzeStore(tmp_path / "bronze")
    store.write_snapshot(sport="basketball_nba", captured_at="2026-10-21T23:20:00Z",
                         payload=_payload(point=-4.5), cost=1, source_url="u")
    rep = G.grade(led, store, sport="basketball_nba")
    assert len(rep.graded) == 2               # kickoff taken from the signal
    paper = G.clv_summary(led, paper=True)
    assert paper["graded_valid"] == 1         # only the side the model preferred
    # Price CLV is measured at the bet's own outcome; a moved line is reported
    # separately as line points. Home -3.5 against a -4.5 close is one point
    # better than the market finished.
    assert paper["mean_line_points"] == pytest.approx(1.0)
    assert G.clv_summary(led)["graded_valid"] == 0    # nothing placed


def test_paper_rows_can_be_excluded_from_grading(led, tmp_path):
    _paper(led)
    store = BronzeStore(tmp_path / "bronze")
    store.write_snapshot(sport="basketball_nba", captured_at="2026-10-21T23:20:00Z",
                         payload=_payload(), cost=1, source_url="u")
    assert G.grade(led, store, sport="basketball_nba", include_paper=False).graded == []


def test_finals_are_keyed_the_way_the_live_slates_key_games():
    from coverline.leagues.nba import live as nba
    src = nba.NBALiveSource.load(2023)
    done = src.schedule[src.schedule.completed].index[:40]
    assert all(g in S.nba_finals() for g in done)
    h = S.nhl_finals()
    assert all(str(g) in h for g in __import__("pandas").read_parquet(
        ROOT / "data" / "raw" / "nhl" / "nhl_2025.parquet").game_id[:40])
    m = S.mlb_finals()
    assert "SFN202603250" in m


def test_an_ambiguous_doubleheader_key_is_never_settled(tmp_path):
    import pandas as pd
    model = tmp_path / "model"
    model.mkdir()
    cols = ["season", "game_key", "date", "game_number", "home_team", "away_team",
            "home_score", "away_score", "park", "home_sp", "away_sp"]
    pd.DataFrame([[2026, "ATL202606170", "2026-06-17", 0, "ATL", "SFN", 3, 2, "P", "a", "b"],
                  [2026, "ATL202606170", "2026-06-17", 0, "ATL", "SFN", 1, 5, "P", "c", "d"],
                  [2026, "NYA202606170", "2026-06-17", 0, "NYA", "BOS", 4, 1, "P", "e", "f"]],
                 columns=cols).to_csv(model / "mlb_schedule_current.csv", index=False)
    f = S.mlb_finals(tmp_path)
    assert "ATL202606170" not in f and f["NYA202606170"] == (4, 1)


def test_the_vendor_keys_match_the_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "recommend_slate", ROOT / "scripts" / "recommend_slate.py")
    R = importlib.util.module_from_spec(spec)
    sys.modules["recommend_slate"] = R
    spec.loader.exec_module(R)
    assert S.VENDOR == {k: v.vendor_sport for k, v in R.LEAGUES.items()}


def test_the_settle_command_closes_and_settles_end_to_end(tmp_path, monkeypatch):
    import importlib.util
    led = BetLedger(tmp_path / "ledger")
    _paper(led, mu=6.0)
    store = BronzeStore(tmp_path / "bronze")
    store.write_snapshot(sport="basketball_nba", captured_at="2026-10-21T23:20:00Z",
                         payload=_payload(point=-4.5), cost=1, source_url="u")
    monkeypatch.setitem(S.FINALS, "nba", lambda: {"401": (110, 104)})
    spec = importlib.util.spec_from_file_location(
        "settle_ledger", ROOT / "scripts" / "settle_ledger.py")
    cmd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd)
    args = ["--league", "nba", "--ledger", str(tmp_path / "ledger"),
            "--bronze", str(tmp_path / "bronze")]
    assert cmd.main(args) == 0
    assert len(led.closes()) == 2 and len(led.outcomes()) == 2
    assert cmd.main(args) == 0                     # idempotent
    assert len(led.closes()) == 2 and len(led.outcomes()) == 2

