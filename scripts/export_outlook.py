#!/usr/bin/env python3
"""NFL season outlook for the Teams tab (dashboard v2 item 12). Display only.

Writes outlook_nfl.json into the site directory: each team's playoff probability from
model.season_simulation.simulate_season_with_playoffs, run on the newest
ratings snapshot -- played games as they happened, the rest simulated with
the rating-only vector live prices use -- n=2000 at the simulator's fixed
seed. Runs in the weekly job after each ratings snapshot (82 seconds
measured locally, 2026-09-26), never on page load.

PLAYOFF ODDS ONLY. The simulator's division tally counts only the #1 seed's
division, a proxy, so no division odds are published.

KNOWN AND NOT CHANGED HERE: the simulator draws residual noise with SD 13.0
(RESIDUAL_GAME_STD in model/season_simulation.py) where the model's measured
MARGIN_SD is 13.2979. Aligning it is a decision for its owner, recorded in
the dashboard v2 report; the output says which value it used.

    python scripts/export_outlook.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SITE = ROOT / "data" / "site"
N_SIMULATIONS = 2000


def inputs(season: int, schedule: pd.DataFrame | None = None, root: Path = ROOT):
    """(played, remaining, ratings, uncertainty, snapshot, path) for the
    newest snapshot of `season`."""
    from export_teams import latest_ratings
    snap, path = latest_ratings(season, root)
    if snap is None:
        raise FileNotFoundError(f"no {season} ratings snapshot")
    if schedule is None:
        from coverline.leagues.nfl.live import load_schedule
        schedule = load_schedule([season])
    s = schedule[schedule.game_type == "REG"].copy()
    s["div_game"] = s.get("div_game", 0).fillna(0).astype(bool)
    played = s[s.home_score.notna()][["home_team", "away_team", "home_score", "away_score",
                                      "div_game"]]
    rem = s[s.home_score.isna()].copy()
    rem["rest_diff"] = (rem.home_rest - rem.away_rest).fillna(0.0)
    rem["is_neutral_site"] = rem.location.eq("Neutral")
    r = pd.DataFrame(snap["ratings"]).set_index("team")
    return played, rem, r[["total_rating"]], r[["rating_std"]], snap, path


def build(season: int | None = None, n: int = N_SIMULATIONS, schedule=None, root: Path = ROOT) -> dict:
    from coverline.leagues.nfl.model import MARGIN_COEFFICIENTS_V1_RATING_ONLY, MARGIN_SD
    from model import season_simulation as S
    season = season or date.today().year
    played, rem, ratings, unc, snap, path = inputs(season, schedule, root)
    res = S.simulate_season_with_playoffs(rem, played, ratings, unc,
                                          dict(MARGIN_COEFFICIENTS_V1_RATING_ONLY), n_simulations=n)
    week = snap.get("week")
    return {
        "league": "nfl", "season": season, "ratings_version": path.name,
        "ratings_computed_at": snap.get("computed_at"), "ratings_week": week,
        "n": n, "seed": 42,
        "label": f"{n:,} simulated seasons from week-{week} ratings. QB changes aren't in it.",
        "residual_sd_used": 13.0, "model_margin_sd": MARGIN_SD,
        "playoff_pct": {t: round(float(p), 4) for t, p in res.playoff_pct.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(SITE))
    ap.add_argument("--n", type=int, default=N_SIMULATIONS)
    a = ap.parse_args(argv)
    art = build(n=a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "outlook_nfl.json").write_text(json.dumps(art, indent=1) + "\n")
    top = sorted(art["playoff_pct"].items(), key=lambda kv: -kv[1])[:3]
    print(f"outlook_nfl.json: {len(art['playoff_pct'])} teams, n={art['n']}, top {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
