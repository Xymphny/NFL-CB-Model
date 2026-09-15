"""
Historical MLB closing lines from the free Sportsbook Reviews archive
-> model/mlb_lines_cache.csv.

NOT runnable from the build sandbox (site egress blocked there) --
run on Render or any open machine, the CFBD pattern. The PARSER is
unit-tested in tests/test_parsers.py against the archive's documented
row shape; the first live download verifies like every external feed.

Archive format (one ROW PER TEAM, visitor then home):
Date(mmdd), Rot, VH, Team, Pitcher, 1st..9th innings, Final,
Open ML, Close ML, RunLine+odds, OpenOU+odds, CloseOU+odds.
Doubleheaders appear as repeated date+teams pairs in rot order; we
number them in file order to match retrosplits game_number.
"""

import io
import os
import urllib.request

import pandas as pd

SEASON_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb%20odds%20{season}.xlsx"

TEAM_MAP = {  # SBR names -> retrosheet keys (extend on first live run's unmatched report)
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Boston": "BOS",
    "CUBS": "CHN", "ChiCubs": "CHN", "ChiSox": "CHA", "WhiteSox": "CHA",
    "Cincinnati": "CIN", "Cleveland": "CLE", "Colorado": "COL", "Detroit": "DET",
    "Houston": "HOU", "KansasCity": "KCA", "LAAngels": "ANA", "LADodgers": "LAN",
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN", "NYMets": "NYN",
    "NYYankees": "NYA", "Oakland": "OAK", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "SanDiego": "SDN", "SanFrancisco": "SFN", "Seattle": "SEA", "StLouis": "SLN",
    "TampaBay": "TBA", "Texas": "TEX", "Toronto": "TOR", "Washington": "WAS",
}


def parse_season_frame(df, season):
    """Pair consecutive V/H rows into games. Pure function, unit-tested."""
    cols = {c.strip().lower(): c for c in df.columns}
    get = lambda row, name: row[cols[name]] if name in cols else None
    games, seen_count = [], {}
    rows = df.to_dict("records")
    i = 0
    while i + 1 < len(rows):
        v, h = rows[i], rows[i + 1]
        if str(get(v, "vh")).strip().upper() not in ("V", "N") or str(get(h, "vh")).strip().upper() != "H":
            i += 1
            continue
        date_raw = int(get(v, "date"))
        month, day = date_raw // 100, date_raw % 100
        away = TEAM_MAP.get(str(get(v, "team")).strip())
        home = TEAM_MAP.get(str(get(h, "team")).strip())
        key = (month, day, away, home)
        n = seen_count.get(key, 0)
        seen_count[key] = n + 1
        def ml(x):
            try:
                val = float(x)
                return val if abs(val) >= 100 else None
            except (TypeError, ValueError):
                return None
        games.append({
            "season": season, "date": f"{season}-{month:02d}-{day:02d}",
            "game_number_in_day": n,          # 0-based; matches retrosplits ordering
            "away_team": away, "home_team": home,
            "away_pitcher_name": str(get(v, "pitcher") or "").strip(),
            "home_pitcher_name": str(get(h, "pitcher") or "").strip(),
            "away_final": get(v, "final"), "home_final": get(h, "final"),
            "away_ml_open": ml(get(v, "open")), "away_ml_close": ml(get(v, "close")),
            "home_ml_open": ml(get(h, "open")), "home_ml_close": ml(get(h, "close")),
            "total_close": None,
        })
        i += 2
    out = pd.DataFrame(games)
    return out[out["away_team"].notna() & out["home_team"].notna()]


def build(first, last, out_path="model/mlb_lines_cache.csv"):
    frames, unmatched = [], set()
    for season in range(first, last + 1):
        url = SEASON_URL.format(season=season)
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "coverline-mlb"}), timeout=60).read()
            raw = pd.read_excel(io.BytesIO(data))
            frame = parse_season_frame(raw, season)
            for name in set(raw[raw.columns[3]].astype(str)) if len(raw.columns) > 3 else set():
                if name.strip() not in TEAM_MAP:
                    unmatched.add(name.strip())
            frames.append(frame)
            print(f"[mlb_odds] {season}: {len(frame)} games parsed")
        except Exception as e:
            print(f"[mlb_odds] {season} soft-fail: {e}")
    if not frames:
        print("[mlb_odds] nothing fetched (blocked environment?) -- run on Render")
        return None
    allf = pd.concat(frames)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    allf.to_csv(out_path, index=False)
    print(f"[mlb_odds] wrote {out_path}: {len(allf)} games; unmatched team names: {sorted(unmatched)[:10]}")
    return out_path


if __name__ == "__main__":
    build(int(os.environ.get("MLB_ODDS_FIRST", 2015)), int(os.environ.get("MLB_ODDS_LAST", 2021)))
