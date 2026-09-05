"""
Committed regression tests backing the docstring claims in
deploy/game_context.py, deploy/espn_extras.py, deploy/qb_status.py and
deploy/cfb_odds_watch.py. Run: python3 tests/test_parsers.py
(plain asserts, no pytest dependency; network-free -- every payload is
a captured real response shape from the live verification sessions).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.game_context import parse_open_meteo, parse_espn_injuries
from deploy.espn_extras import parse_fpi_summary
from deploy.cfb_odds_watch import map_odds_names_to_ratings
from deploy.qb_status import _norm_name


def test_open_meteo():
    payload = {"hourly": {
        "time": ["2026-09-13T12:00", "2026-09-13T13:00", "2026-09-13T14:00"],
        "temperature_2m": [61.0, 63.5, 65.1],
        "wind_speed_10m": [12.0, 17.5, 19.0],
        "precipitation_probability": [10, 35, 40]}}
    wx = parse_open_meteo(payload, "2026-09-13T13:00")
    assert wx == {"temp_f": 63.5, "wind_mph": 17.5, "precip_prob": 35, "forecast_hour": "2026-09-13T13:00"}
    assert parse_open_meteo({}, "2026-09-13T13:00") is None


def test_espn_injuries():
    payload = {"injuries": [
        {"displayName": "Atlanta Falcons", "injuries": [
            {"status": "Active", "athlete": {"displayName": "Jack Strand", "position": {"abbreviation": "QB"}}},
            {"status": "Questionable", "athlete": {"displayName": "Michael Penix Jr.", "position": {"abbreviation": "QB"}}},
            {"status": "Injured Reserve", "athlete": {"displayName": "Some Lineman", "position": {"abbreviation": "G"}}},
            {"status": "Suspension", "athlete": {"displayName": "Some CB", "position": {"abbreviation": "CB"}}}]},
        {"displayName": "Unknown Team XFL", "injuries": []}]}
    rep = parse_espn_injuries(payload, {"Atlanta Falcons": "ATL"})
    assert set(rep) == {"ATL"} and len(rep["ATL"]) == 3         # Active dropped, unknown team skipped
    assert rep["ATL"][0]["status"] == "Out"                      # IR/Susp -> Out, worst-first
    assert any(p["player"] == "Michael Penix Jr." and p["status"] == "Questionable" for p in rep["ATL"])


def test_fpi():
    s = {"predictor": {"homeTeam": {"gameProjection": "61.1"}, "awayTeam": {"gameProjection": "38.6"}}}
    assert parse_fpi_summary(s, "SEA", "NE") == {"home": "SEA", "away": "NE", "fpi_home_prob": 0.611}
    assert parse_fpi_summary({"predictor": {}}, "SEA", "NE") is None
    r = parse_fpi_summary(s, "WSH", "LAR")
    assert r["home"] == "WAS" and r["away"] == "LA"              # ESPN->nflverse abbr bridge


def test_cfb_name_mapper():
    teams = ["Utah", "Georgia", "Georgia Tech", "Georgia State", "Tennessee",
             "Texas A&M", "Hawai'i", "Miami", "Miami (OH)", "App State", "NC State"]
    odds = [{"home_team": h, "away_team": a} for h, a in [
        ("Utah Tech Trailblazers", "Utah Utes"),
        ("Tennessee Tech Golden Eagles", "Tennessee Volunteers"),
        ("Georgia State Panthers", "Georgia Bulldogs"),
        ("Hawaii Rainbow Warriors", "Texas A&M Aggies"),
        ("Miami (OH) RedHawks", "Miami Hurricanes"),
        ("Appalachian State Mountaineers", "North Carolina State Wolfpack")]]
    m, u = map_odds_names_to_ratings(odds, teams)
    assert "Utah Tech Trailblazers" in u and "Tennessee Tech Golden Eagles" in u
    assert m["Utah Utes"] == "Utah" and m["Georgia State Panthers"] == "Georgia State"
    assert m["Hawaii Rainbow Warriors"] == "Hawai'i" and m["Miami (OH) RedHawks"] == "Miami (OH)"
    assert m["Appalachian State Mountaineers"] == "App State"
    assert m["North Carolina State Wolfpack"] == "NC State"


def test_name_norm():
    assert _norm_name("Michael Penix Jr.") == _norm_name("Michael Penix")
    assert _norm_name("Robert Griffin III") == _norm_name("Robert Griffin")
    assert _norm_name("Odell Beckham Jr") != _norm_name("Odell Beck")




def test_depth_chart_schemas():
    """Network-free schema regression for _projected_starters, covering
    the depth_team/depth_position confusion caught by external audit."""
    import pandas as pd
    import deploy.qb_status as qs
    orig = pd.read_parquet
    def fake(url, *a, **k):
        if "2025plus" in url:
            return pd.DataFrame([
                {"team": "ATL", "player_name": "Tua Tagovailoa", "pos_abb": "QB", "pos_rank": 1, "dt": "2026-09-04"},
                {"team": "ATL", "player_name": "Michael Penix", "pos_abb": "QB", "pos_rank": 2, "dt": "2026-09-04"}])
        return pd.DataFrame([
            {"club_code": "KC", "full_name": "Patrick Mahomes", "position": "QB", "depth_position": "QB", "depth_team": "1", "week": 10},
            {"club_code": "KC", "full_name": "Backup Guy", "position": "QB", "depth_position": "QB", "depth_team": "2", "week": 10}])
    pd.read_parquet = fake
    old_url = qs.DEPTH_CHARTS_URL
    try:
        qs.DEPTH_CHARTS_URL = "x2025plus{season}"
        assert qs._projected_starters(2026) == {"ATL": "Tua Tagovailoa"}
        qs.DEPTH_CHARTS_URL = "legacy{season}"
        assert qs._projected_starters(2024) == {"KC": "Patrick Mahomes"}
    finally:
        pd.read_parquet = orig
        qs.DEPTH_CHARTS_URL = old_url


def test_cfb_scoreboard_finals():
    """Captured ESPN CFB scoreboard shape -> completed finals only."""
    from deploy.generate_cfb_performance import parse_scoreboard_finals
    payload = {"events": [
        {"competitions": [{"status": {"type": {"completed": True}}, "competitors": [
            {"homeAway": "home", "score": "31", "team": {"displayName": "Alabama Crimson Tide"}},
            {"homeAway": "away", "score": "17", "team": {"displayName": "East Carolina Pirates"}}]}]},
        {"competitions": [{"status": {"type": {"completed": False}}, "competitors": [
            {"homeAway": "home", "score": "7", "team": {"displayName": "Oregon Ducks"}},
            {"homeAway": "away", "score": "3", "team": {"displayName": "Utah Utes"}}]}]},
    ]}
    rows = parse_scoreboard_finals(payload)
    assert rows == [("Alabama Crimson Tide", "East Carolina Pirates", 31, 17)]


def test_cfb_grading_mirrors_board():
    """Week-1 play-sized edge grades at LEAN stakes; CLV vs frozen close."""
    from deploy.generate_cfb_performance import grade_week
    snaps = [
        {"divergences": [{"home_team": "Alabama", "away_team": "East Carolina",
                          "market_spread": 29.0, "spread_gap": 13.0}]},   # earliest: entry
        {"divergences": [{"home_team": "Alabama", "away_team": "East Carolina",
                          "market_spread": 31.0, "spread_gap": 11.0, "line_status": "closed"}]},  # latest: close
    ]
    plays = grade_week(1, snaps, {("Alabama", "East Carolina"): (45, 10)})
    assert len(plays) == 1
    p = plays[0]
    assert p["tier"] == "lean"                 # week 1 cap: 13-pt edge still lean
    assert p["result"] == "win"                # ALA -29, won by 35
    assert abs(p["units"] - 0.5 * 100 / 110) < 1e-3  # grader rounds to 3dp
    assert p["clv"] == 2.0                     # entered -29, closed -31, home side: +2 CLV
    plays5 = grade_week(6, snaps, {("Alabama", "East Carolina"): (45, 10)})
    assert plays5[0]["tier"] == "play"         # week 6: same edge is a Play



if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name}: OK")
    print("all parser regression tests passed")
