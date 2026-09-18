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


def test_finished_games_stay_on_board():
    """A game the odds feed dropped (FINAL) is carried frozen; a game
    absent with a future kickoff (postponed) is not."""
    import deploy.odds_watch_job as ow
    now = "2026-09-13T23:00:00Z"
    odds = [{"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos",
             "commence_time": "2026-09-14T00:20:00Z"}]  # only the late game remains in the feed
    snaps = [("2026-09-13T12:00:00Z", [
        {"home_team": "DET", "away_team": "MIN", "home_name": "Detroit Lions", "away_name": "Minnesota Vikings",
         "market_spread": 3.5, "kickoff": "2026-09-13T17:00:00Z", "line_status": "open"},
        {"home_team": "KC", "away_team": "DEN", "home_name": "Kansas City Chiefs", "away_name": "Denver Broncos",
         "market_spread": 7.0, "kickoff": "2026-09-14T00:20:00Z", "line_status": "open"},
        {"home_team": "BUF", "away_team": "MIA", "home_name": "Buffalo Bills", "away_name": "Miami Dolphins",
         "market_spread": 6.0, "kickoff": "2026-09-20T17:00:00Z", "line_status": "open"},  # postponed-style: future
    ])]
    pregame, carried = ow.split_started_and_carry(odds, snaps, now)
    keys = {(c["home_team"], c["away_team"]) for c in carried}
    assert ("DET", "MIN") in keys              # finished + dropped -> carried frozen
    assert ("BUF", "MIA") not in keys          # future kickoff -> not carried
    assert all(c["line_status"] == "closed" for c in carried)
    assert [g["home_team"] for g in pregame] == ["Kansas City Chiefs"]


def test_inseason_debias():
    """The in-season constant-bias correction (live catch 2026-09-17:
    mean gap -4.06, 15/16 negative, nine one-directional Plays)."""
    from deploy.odds_watch_job import inseason_offsets, load_prior_debias
    import json, os, tempfile
    probe = [{"spread_gap": g, "total_gap": None} for g in
             (-4.5, -3.9, -4.2, -5.0, -3.4, -4.8, -4.1, -2.9, -6.0, 1.2)]
    s_off, t_off = inseason_offsets(probe)
    assert 3.9 <= s_off <= 4.6 and t_off == 0.0        # median-robust: the +1.2 outlier survives
    adj = [p["spread_gap"] + s_off for p in probe]
    import statistics
    assert abs(statistics.median(adj)) < 0.01           # median re-centered
    assert max(adj) > 5.0                               # large real edge NOT shrunk (slope stays 1)
    assert inseason_offsets(probe[:5]) == (0.0, 0.0)    # small slate defers to fallback
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "divergence"))
    json.dump({"debias_offsets": [4.1, -1.0]}, open(os.path.join(d, "divergence", "2026-week-02-x.json"), "w"))
    assert load_prior_debias(d, "divergence", 2026, 2) == (4.1, -1.0)
    assert load_prior_debias(d, "divergence", 2026, 3) is None


def test_cfb_slate_week():
    """Week label derives from the slate's own kickoffs, not the
    ratings file (a stalled ratings step froze the label at week 2
    while the board priced the week-3 slate, 2026-09-17)."""
    from deploy.cfb_odds_watch import slate_week
    assert slate_week(["2026-08-29T16:00:00Z"], 2026) == 0
    assert slate_week(["2026-09-05T16:00:00Z"], 2026) == 1
    assert slate_week(["2026-09-12T16:00:00Z"], 2026) == 2
    assert slate_week(["2026-09-18T00:00:00Z", "2026-09-19T16:00:00Z", "2026-09-19T20:00:00Z"], 2026) == 3
    assert slate_week([], 2026) is None


def test_props_called_with_module_key():
    """fetch_week_props must receive the module's ODDS_API_KEY -- the
    original call passed an undefined name, and the soft-fail wrapper
    turned the NameError into a silent weekly '[props] skipped' (no
    props file ever written; caught 2026-09-17)."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "deploy", "odds_watch_job.py")).read()
    assert "fetch_week_props(ODDS_API_KEY" in src


def test_regime_layer():
    """Coach-regime layer (evidence: model/coach_regime_experiment.py --
    backed-regime early flags 0/9 ATS 2016-2023; faded-regime flags at
    baseline; internal promotions behave like stable teams)."""
    from deploy.odds_watch_job import apply_regime_layer, load_regime_map
    regimes = {"NYG": {"tier": 2, "coach": "J.Harbaugh"}, "ARI": {"tier": 1, "coach": "LaFleur"}}
    rows = [
        {"home_team": "NYG", "away_team": "DAL", "spread_gap": 4.5, "line_status": "open"},   # backs NYG -> cap
        {"home_team": "LA", "away_team": "ARI", "spread_gap": 3.0, "line_status": "open"},    # backs LA (fades ARI) -> chip only
        {"home_team": "KC", "away_team": "DEN", "spread_gap": 5.0, "line_status": "open"},    # stable game -> nothing
        {"home_team": "NYG", "away_team": "PHI", "spread_gap": -4.0, "line_status": "closed"},# closed -> untouched
    ]
    apply_regime_layer(rows, regimes, week=2)
    assert rows[0].get("tier_cap") == "lean" and "0/9" in rows[0]["tier_cap_reason"]
    assert rows[1].get("tier_cap") is None and rows[1]["regime"]["away"]["coach"] == "LaFleur"
    assert "regime" not in rows[2] and "tier_cap" not in rows[2]
    assert rows[3].get("tier_cap") is None                    # frozen rows never capped
    rows2 = [{"home_team": "NYG", "away_team": "DAL", "spread_gap": 4.5, "line_status": "open"}]
    apply_regime_layer(rows2, regimes, week=5)
    assert rows2[0].get("tier_cap") is None                   # cap window is weeks 1-4
    rows3 = [{"home_team": "NYG", "away_team": "DAL", "spread_gap": 4.5, "line_status": "open"}]
    apply_regime_layer(rows3, regimes, week=7)
    assert "regime" not in rows3[0]                           # chip window ends week 6
    # Internal promotions excluded at load time.
    import json, tempfile, os
    d = tempfile.mkdtemp()
    json.dump({"changes": {"2026": {"BUF": [0, "Brady"], "NYG": [2, "J.Harbaugh"]}}},
              open(os.path.join(d, "coach_changes.json"), "w"))
    m = load_regime_map(2026, d)
    assert "BUF" not in m and m["NYG"]["tier"] == 2


def test_grader_honors_regime_cap():
    """A board-capped spread Play grades at Lean stakes -- the record
    grades what the board showed, never a shadow book."""
    from deploy.generate_performance import grade_divergence, STAKES
    d = {"home_team": "NYG", "away_team": "DAL", "spread_gap": 5.0, "market_spread": -3.0,
         "tier_cap": "lean", "total_gap": None}
    graded = grade_divergence(d, 2, {(2, "NYG", "DAL"): {"home_score": 27, "away_score": 20, "spread_line": -3.0}})
    assert len(graded) == 1 and graded[0]["tier"] == "lean"
    assert abs(graded[0]["units"]) in (0.0, round(STAKES["lean"] * 100 / 110, 3), STAKES["lean"])


def test_props_consensus_edge():
    """Model-free prop edges: fair value = de-vigged multi-book
    consensus; edge = best price against it. No player projections
    anywhere -- the market is the model."""
    from deploy.odds_watch_job import consensus_edge
    books = {"DK": {"line": 220.5, "over": -110, "under": -110},
             "FD": {"line": 220.5, "over": -112, "under": -108},
             "MGM": {"line": 220.5, "over": -108, "under": -112},
             "Soft": {"line": 220.5, "over": 105, "under": -135},
             "Off": {"line": 224.5, "over": -110, "under": -110}}
    r = consensus_edge(books)
    assert r["edge"]["side"] == "over" and r["edge"]["book"] == "Soft" and r["edge"]["ev_pct"] > 2
    assert r["off_market"][0]["book"] == "Off" and r["off_market"][0]["vs_consensus"] == 4.0
    thin = consensus_edge({"DK": {"line": 60.5, "over": -110, "under": -110},
                           "FD": {"line": 60.5, "over": -115, "under": -105}})
    assert thin["edge"] is None                      # 2 books never make an edge claim
    bal = consensus_edge({b: {"line": 100.5, "over": -110, "under": -110} for b in "ABC"})
    assert bal["edge"]["ev_pct"] < 0                 # balanced market: EV is vig-negative
    td = consensus_edge({"DK": {"yes": -175}, "FD": {"yes": -180}, "S": {"yes": -120}}, yes_market=True)
    assert td["edge"]["side"] == "yes" and "conservative" in td["edge"]["basis"]


def test_props_format_refetch():
    """A format-1 props file (pre-consensus schema) triggers ONE
    refetch; a current-format file still dedupes."""
    import json, os, tempfile
    import deploy.odds_watch_job as ow
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "props"))
    path = os.path.join(d, "props", "2026-week-02.json")
    json.dump({"season": 2026, "week": 2, "games": {}}, open(path, "w"))   # format 1 (absent)
    calls = []
    orig = ow.requests.get
    class R:
        def raise_for_status(self): pass
        def json(self): return {"bookmakers": []}
    ow.requests.get = lambda *a, **k: (calls.append(1), R())[1]
    try:
        ow.fetch_week_props("k", 2026, 2, d, [{"id": "x"}])
        assert calls, "old-format file must trigger a refetch"
        json.dump({"season": 2026, "week": 2, "format": ow.PROPS_FORMAT, "games": {}}, open(path, "w"))
        calls.clear()
        assert ow.fetch_week_props("k", 2026, 2, d, [{"id": "x"}]) is None and not calls
    finally:
        ow.requests.get = orig



if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name}: OK")
    print("all parser regression tests passed")
