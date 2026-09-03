"""
Real ESPN Pass Rush/Run Stop/Pass Block/Run Block Win Rate metrics --
the closest free, real answer found to the "PFF-style charting" gap
identified early in this project and reconfirmed by the failed sack-
based pass-rusher VAR test (model/injuries_and_var.py's own docstring
documents that failure). Unlike sacks (situational, not skill-
isolating), these are genuine per-play, assignment-level win/loss
determinations computed by ESPN Analytics from real NFL Next Gen Stats
player tracking data -- exactly the category of signal that test was
missing.

CONFIRMED REAL, not guessed: verified directly against ESPN's actual
published page (espn.com/nfl/story/.../2025-nfl-win-rates-...),
cross-checked against a known open-source scraper (espnscrapeR's
scrape_espn_win_rate(), which has independently scraped this same
data since 2020) that confirms the format has been stable for years.

HONEST SCOPE LIMIT, found while building this: unlike the FPI or odds
pages (stable URLs, live data), this win-rate leaderboard is published
as a NEW dated story article each year with a unique numeric ID --
there's no parameterized URL to construct season over season. The
current URL below is real and verified for the 2025 season recap
(published Jan 2026); a future season will need its own URL found via
search, same as the personnel-changes/coaching-changes files that
also needed real, manually-verified 2026-specific data.
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd

KNOWN_WIN_RATE_URLS = {
    2025: "https://www.espn.com/nfl/story/_/id/46138675/2025-nfl-win-rates-top-teams-players-rankings-pass-run-block",
}


def fetch_team_win_rates(url):
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    html = resp.text

    tables = pd.read_html(html)
    team_table = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("prwr" in c for c in cols) and len(t) >= 30:
            team_table = t
            break

    if team_table is None:
        raise ValueError("Could not find the team win-rate summary table -- ESPN may have changed page structure")

    def extract_pct(cell):
        match = re.search(r"(\d+)%", str(cell))
        return int(match.group(1)) if match else None

    result = pd.DataFrame({
        "team_name": team_table.iloc[:, 0],
        "prwr": team_table.iloc[:, 1].apply(extract_pct),
        "rswr": team_table.iloc[:, 2].apply(extract_pct),
        "pbwr": team_table.iloc[:, 3].apply(extract_pct),
        "rbwr": team_table.iloc[:, 4].apply(extract_pct),
    })
    return result


if __name__ == "__main__":
    print("Testing parser against the real, verified 2025 season page...")
    df = fetch_team_win_rates(KNOWN_WIN_RATE_URLS[2025])
    print(f"\n{len(df)} teams parsed")
    print("\nBest pass rush win rate:")
    print(df.sort_values("prwr", ascending=False).head(5))
