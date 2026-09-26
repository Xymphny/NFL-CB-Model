"""The board's per-game context (dashboard v2 items 1, 5, 6, 11).

What these pin: context never moves a price; every item says whether the
model reads it, and that answer comes from the model's inputs; a manual
override beats every feed; a missing source is "no report", not healthy.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import board_context as C  # noqa: E402
import export_board as X  # noqa: E402
from tests.core.test_board_export import make_nba_night  # noqa: E402


# ----------------------------------------------------------- in_price ------

def test_in_price_is_the_models_inputs_and_matches_the_brief():
    ip = C.in_price()
    assert ip["nfl"]["rest"] is True, "rest_diff is in the live rating-only vector"
    assert ip["nfl"]["weather"] is False, "wind reaches only the withheld totals model"
    assert ip["mlb"]["venue"] is True, "expected runs carry the park"
    assert ip["mlb"]["weather"] is False
    for league in ("cfb", "nba", "nhl"):
        assert not any(ip[league].values()), league
    for league in ip:
        assert ip[league]["injuries"] is False and ip[league]["key_player"] is False
        assert ip[league]["travel"] is False


def test_nfl_rest_follows_the_vector_not_a_belief(monkeypatch):
    from coverline.leagues.nfl import model as nfl
    monkeypatch.setitem(nfl.MARGIN_COEFFICIENTS_V1_RATING_ONLY, "rest_diff", 0.0)
    assert C.in_price()["nfl"]["rest"] is False


# ------------------------------------------------------------ injuries -----

def test_nfl_injuries_sort_worst_first_then_key_positions():
    rows = [{"player": "S1", "position": "S", "status": "Out"},
            {"player": "Q1", "position": "QB", "status": "Questionable"},
            {"player": "W1", "position": "WR", "status": "Out"},
            {"player": "T1", "position": "T", "status": "Out"},
            {"player": "D1", "position": "DE", "status": "Doubtful"}]
    got = [r["player"] for r in C.order_injuries("nfl", rows)]
    assert got == ["W1", "T1", "S1", "D1", "Q1"]


def test_cards_show_four_on_a_play_three_elsewhere_and_counts_on_a_coin_flip():
    rows = [{"player": f"P{i}", "position": "WR", "status": "Out"} for i in range(6)]
    play = C.injury_display(rows, "play")
    assert len(play["shown"]) == 4 and play["more"] == 2
    lean = C.injury_display(rows, "lean")
    assert len(lean["shown"]) == 3 and lean["more"] == 3
    flip = C.injury_display(rows, "coin_flip")
    assert flip["shown"] == [] and flip["counts"] == {"Out": 6}


def test_a_missing_report_is_unknown_not_healthy():
    assert C.injury_display(None, "lean")["status"] == "no report"
    assert C.team_injuries(None, "BOS", "nba") is None
    assert C.team_injuries({"NYK": []}, "BOS", "nba") == []            # on the report, nobody
    assert C.team_injuries({"Ohio State": []}, "Toledo", "cfb") is None  # college: unknown


def test_statuses_are_normalised():
    assert C.normalise_status("60-Day-IL") == "60-Day IL"
    assert C.normalise_status("Suspension") == "Suspended"
    assert C.normalise_status("Active") is None


# --------------------------------------------------------- key players -----

def test_an_override_beats_every_feed(tmp_path, monkeypatch):
    f = tmp_path / "o.json"
    f.write_text('{"nfl": {"ATL": {"alert": "Rush starts per team report"}}, "nba": {}}')
    monkeypatch.setattr(C, "OVERRIDES", f)
    feed = {"ATL": {"text": "Tagovailoa started last game", "source": "nflverse games.csv"},
            "NO": {"text": "Carr listed Out", "source": "nflverse injury report"}}
    got = C.key_players("nfl", "NO", "ATL", feed)
    assert {k["team"]: (k["text"], k["source"]) for k in got} == {
        "ATL": ("Rush starts per team report", "manual override"),
        "NO": ("Carr listed Out", "nflverse injury report")}
    assert all(k["role"] == "QB1" for k in got)


def test_an_override_starter_silences_the_last_start_signal(monkeypatch, tmp_path):
    """The known feed error: games.csv listed Tagovailoa as ATL's week-2
    starter when reports had Rush starting. With an override naming the
    starter, the last-start alert must not fire."""
    from deploy import qb_status as Q
    f = tmp_path / "o.json"
    f.write_text('{"nfl": {"ATL": {"starter": "Cooper Rush"}}}')
    monkeypatch.setattr(Q, "OVERRIDES_PATH", str(f))
    monkeypatch.setattr(Q, "get_projected_starters", lambda season: {"ATL": "Cooper Rush"})
    import deploy.game_context as GC
    monkeypatch.setattr(GC, "fetch_espn_injuries", lambda: {})      # never the live feed
    games = pd.DataFrame([
        {"season": 2026, "week": w, "home_team": "ATL", "away_team": "X", "home_score": 20,
         "away_score": 10, "home_qb_name": qb, "away_qb_name": "Q"}
        for w, qb in ((1, "Cooper Rush"), (2, "Tua Tagovailoa"))])
    empty = pd.DataFrame(columns=["team", "full_name", "position", "week", "report_status"])
    assert "ATL" not in Q.get_qb_alerts_detailed(2026, 3, games=games, injuries=empty)
    monkeypatch.setattr(Q, "OVERRIDES_PATH", str(tmp_path / "none.json"))
    got = Q.get_qb_alerts_detailed(2026, 3, games=games, injuries=empty)
    assert got["ATL"]["source"].startswith("nflverse games.csv"), "the source is shown"


def test_nba_starters_out_are_flagged_and_bench_players_are_not():
    players = {"teams": [{"team": "BOS", "games_in_window": 10,
                         "rotation": [{"player": "Star", "starts": 10},
                                      {"player": "Bench", "starts": 0}],
                         "injuries": [{"player": "Star", "status": "Out"},
                                      {"player": "Bench", "status": "Out"}]}]}
    assert C.nba_starter_alerts(players) == {
        "BOS": {"text": "Star Out", "source": "ESPN injury report"}}
    assert C.nba_starter_alerts(None) == {}


# ------------------------------------------------------ price unchanged ----

def _strip(board):
    return [(g["game_id"], g.get("model"), g.get("markets")) for g in board["games"]]


def test_context_never_moves_a_price_and_the_strip_follows_the_alert(tmp_path, monkeypatch):
    """Build the same board with a key-player alert and injuries, then with
    none: every model and market field identical, the strip present only
    when the alert is."""
    board_plain, _, _ = make_nba_night(tmp_path / "a")
    home = board_plain["games"][0]["home"]
    monkeypatch.setattr(C, "_nba_players", lambda: {"teams": [
        {"team": home, "games_in_window": 10, "rotation": [{"player": "Star", "starts": 10}],
         "injuries": [{"player": "Star", "status": "Out", "position": "G"}]}]})
    board_ctx, _, _ = make_nba_night(tmp_path / "b")
    assert _strip(board_ctx) == _strip(board_plain)
    g = next(x for x in board_ctx["games"] if x["home"] == home)
    assert g["context"]["key_player"] == [
        {"team": home, "role": "starter", "text": "Star Out", "source": "ESPN injury report"}]
    assert g["context"]["injuries"]["home"]["shown"][0]["player"] == "Star"
    plain = next(x for x in board_plain["games"] if x["home"] == home)
    assert plain["context"]["key_player"] == []
    assert plain["context"]["weather"]["status"] == "indoors"
    assert plain["context"]["venue"]["in_price"] is False


def test_a_failing_source_never_blocks_a_board():
    def boom():
        raise RuntimeError("feed down")
    assert X._context_bundle(boom) == {}


def test_nfl_context_from_the_schedule_row(monkeypatch, tmp_path):
    f = tmp_path / "o.json"
    f.write_text("{}")
    monkeypatch.setattr(C, "OVERRIDES", f)
    wk = pd.DataFrame([
        {"home_team": "DET", "away_team": "NYJ", "stadium": "Ford Field", "roof": "dome",
         "surface": "fieldturf", "location": "Home", "home_rest": 7, "away_rest": 10},
        {"home_team": "DAL", "away_team": "BAL", "stadium": "Somewhere", "roof": "outdoors",
         "surface": "grass", "location": "Neutral", "home_rest": 7, "away_rest": 7}])
    got = C.nfl(wk, ["g1", "g2"], 2026, 3, {"g1": "2030-01-01T18:00:00Z",
                                            "g2": "2030-01-01T18:00:00Z"})
    dome, neutral = got["g1"], got["g2"]
    assert dome["weather"]["status"] == "indoors" and dome["venue"]["name"] == "Ford Field"
    assert dome["rest"] == {"home_days": 7, "away_days": 10, "in_price": True,
                            "note": "The rest difference is one of the model's inputs."}
    assert dome["travel"]["away_miles"] > 400 and dome["travel"]["in_price"] is False
    assert neutral["travel"]["away_miles"] is None, "a neutral site's distance is unknown"
    assert neutral["weather"]["status"] == "no report"
    assert dome["injuries"]["status"] == "no report"                  # offline: unknown


def test_miles_are_great_circle():
    assert 2400 < C.miles((40.713, -74.006), (34.052, -118.244)) < 2500
    assert C.miles(None, (1, 1)) is None
