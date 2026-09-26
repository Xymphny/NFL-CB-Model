"""Team logos: keyed by model codes through the trusted tables, never guessed."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from model.ingest import logos as L  # noqa: E402


def _team(display, location, lid, dark=True):
    logos = [{"href": f"https://x/500/{lid}.png", "rel": ["full", "default"]}]
    if dark:
        logos.append({"href": f"https://x/500-dark/{lid}.png", "rel": ["full", "dark"]})
    return {"team": {"displayName": display, "location": location, "logos": logos}}


def _payload(*teams):
    return {"sports": [{"leagues": [{"teams": list(teams)}]}]}


def test_the_dark_variant_is_the_one_the_site_uses():
    got, _ = L.build("nba", _payload(_team("Boston Celtics", "Boston", "bos")))
    assert got == {"BOS": {"dark": "https://x/500-dark/bos.png", "light": "https://x/500/bos.png"}}
    got, _ = L.build("nba", _payload(_team("Boston Celtics", "Boston", "bos", dark=False)))
    assert got["BOS"]["dark"] == "https://x/500/bos.png"


def test_a_shared_college_location_takes_the_table_school_never_list_order():
    got, _ = L.build("cfb", _payload(_team("Troy Vikings", "Troy", "v"), _team("Troy Trojans", "Troy", "t"),
                                     _team("Roosevelt Lakers", "Roosevelt", "a"),
                                     _team("Roosevelt Lakers", "Roosevelt", "b")))
    assert got["Troy"]["light"] == "https://x/500/t.png"
    assert "Roosevelt" not in got, "ambiguous and unknown to the table: no logo, not a guess"


def test_an_unknown_team_is_reported_not_guessed():
    got, unplaced = L.build("nhl", _payload(_team("Seattle Metropolitans", "Seattle", "sm")))
    assert got == {} and unplaced == ["Seattle Metropolitans"]


def test_the_committed_logos_cover_every_team_the_models_price():
    art = json.loads((ROOT / "data" / "logos.json").read_text())
    for league, n in {"nfl": 32, "nba": 30, "nhl": 32, "mlb": 30}.items():
        assert len(art[league]) == n, league
    assert "LA" in art["nfl"] and "NYA" in art["mlb"] and "UTAH" in art["nba"] and "TBL" in art["nhl"]


def test_the_board_carries_each_teams_logo(tmp_path):
    from tests.core.test_board_export import make_nba_night
    board, _, _ = make_nba_night(tmp_path)
    g = board["games"][0]
    assert g["home_logo"].startswith("https://") and "500-dark" in g["home_logo"]
