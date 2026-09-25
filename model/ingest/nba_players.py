#!/usr/bin/env python3
"""NBA player availability and minutes, for the Players tab. NOT a model input.

WHY THIS EXISTS WHEN THE MODEL IGNORES IT
The NBA model rates teams. The published work that beats this market is built
from players and projected minutes, and the lines move on injury news hours
before tip. The model cannot see any of that, so the dashboard shows it beside
the model's read and the owner weighs it. Nothing here is read by a pricing
path, and a test pins that.

WHAT IT WRITES
data/raw/nba/injuries_current.json
    ESPN's league injury report, normalised: {team code: [rows]}, worst first.
    Written only when its CONTENT changes (ESPN's own per-row dates, never the
    fetch time), so an unchanged report makes no commit. History is in git.
data/raw/nba/player_box_{season}.parquet
    One row per player per completed regular-season game: minutes, starter,
    did-not-play. Fetched INCREMENTALLY -- only finals not yet cached, capped
    per run -- so a normal day costs one summary call per game played the
    night before. Retained as raw data: it is what a future player-minutes
    layer would be fitted and graded on.

Team codes are the fitted codes (NY, GS, SA, UTAH, ...), the same ones
data/site/board_nba.json uses, via coverline.leagues.nba.teams.

    python model/ingest/nba_players.py --season 2027
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RAW = ROOT / "data" / "raw" / "nba"
SDV = ROOT / "data" / "raw" / "sportsdataverse"
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}"
#: No User-Agent override -- see model/ingest/cfb_espn.py.
HEADERS: dict[str, str] = {}
#: Summaries fetched per run. A season backfill spreads over a few runs rather
#: than hammering an ungated endpoint; a normal night is ~15 games.
MAX_GAMES_PER_RUN = 120
SLEEP_S = 0.2

#: ESPN status -> the official NBA vocabulary, worst first.
STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Probable": 3, "Day-To-Day": 4}
ESPN_STATUS = {"Out": "Out", "Doubtful": "Doubtful", "Questionable": "Questionable",
               "Probable": "Probable", "Day-To-Day": "Day-To-Day", "Suspension": "Out"}

BOX_COLUMNS = ["season", "game_id", "date", "team", "athlete_id", "player", "position",
               "starter", "minutes", "did_not_play"]


def _get(url: str, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed: {url}") from last


def name_to_code() -> dict[str, str]:
    from coverline.leagues.nba.teams import NBA_NAMES
    return dict(NBA_NAMES)


# ------------------------------------------------------------- injuries ----

def parse_injuries(payload: dict, names: dict[str, str]) -> dict[str, list[dict]]:
    """ESPN league injury report -> {team code: rows}, worst status first.

    Pure. A team whose name the table does not know is dropped by name in the
    returned `_unmatched` list rather than silently: a new spelling is exactly
    the failure the team tables exist to catch.
    """
    out: dict[str, list[dict]] = {}
    unmatched: list[str] = []
    for team in payload.get("injuries", []):
        code = names.get(team.get("displayName", ""))
        if not code:
            unmatched.append(team.get("displayName", "?"))
            continue
        rows = []
        for inj in team.get("injuries", []):
            status = ESPN_STATUS.get(inj.get("status", ""))
            athlete = inj.get("athlete") or {}
            if not status or not athlete.get("displayName"):
                continue
            det = inj.get("details") or {}
            rows.append({
                "player": athlete["displayName"],
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
                "status": status,
                "injury": " ".join(x for x in (det.get("side"), det.get("type"))
                                   if x and x != "Not Specified") or None,
                "return_date": det.get("returnDate"),
                "note": inj.get("shortComment"),
                "reported": inj.get("date"),
            })
        rows.sort(key=lambda r: (STATUS_ORDER.get(r["status"], 9), r["player"]))
        out[code] = rows
    out = dict(sorted(out.items()))
    if unmatched:
        out["_unmatched"] = sorted(unmatched)            # type: ignore[assignment]
    return out


def write_if_changed(path: Path, obj) -> bool:
    text = json.dumps(obj, indent=1, sort_keys=False) + "\n"
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


# -------------------------------------------------------------- minutes ----

def _minutes(v) -> int | None:
    try:
        return int(str(v).split(":")[0])
    except (TypeError, ValueError):
        return None


def parse_box(summary: dict, game: dict, names: dict[str, str]) -> list[dict]:
    """One game summary -> player rows. Pure.

    `game` carries season, game_id and date from the schedule file, so rows
    are keyed exactly as the board keys games. Team codes come from the
    schedule's home/away codes matched on ESPN's team id when present, else
    on the display name.
    """
    rows = []
    for side in (summary.get("boxscore") or {}).get("players", []):
        t = side.get("team") or {}
        code = names.get(t.get("displayName", "")) or t.get("abbreviation")
        for block in side.get("statistics", [])[:1]:
            labels = block.get("labels") or []
            i_min = labels.index("MIN") if "MIN" in labels else None
            for a in block.get("athletes", []):
                ath = a.get("athlete") or {}
                stats = a.get("stats") or []
                dnp = bool(a.get("didNotPlay")) or not stats
                mins = _minutes(stats[i_min]) if (i_min is not None and not dnp and len(stats) > i_min) else None
                rows.append({
                    "season": int(game["season"]), "game_id": str(game["game_id"]),
                    "date": str(game["date"]), "team": code,
                    "athlete_id": str(ath.get("id", "")), "player": ath.get("displayName"),
                    "position": (ath.get("position") or {}).get("abbreviation", ""),
                    "starter": bool(a.get("starter")),
                    "minutes": mins if mins is not None else (0 if dnp else None),
                    "did_not_play": dnp,
                })
    return rows


def completed_games(season: int, sdv: Path = SDV) -> pd.DataFrame:
    f = sdv / f"nba_{season}.parquet"
    if not f.exists():
        return pd.DataFrame(columns=["season", "game_id", "date"])
    d = pd.read_parquet(f)
    d = d[(d.season_type == 2) & (d.type_abbreviation == "STD")
          & d.status_type_completed.fillna(False).astype(bool)]
    return pd.DataFrame({"season": d.season.astype(int), "game_id": d.id.astype(str),
                         "date": d.date.astype(str)}).sort_values("date")


def update_box(season: int, raw: Path = RAW, sdv: Path = SDV,
               fetch=None, limit: int = MAX_GAMES_PER_RUN) -> tuple[int, int]:
    """Fetch summaries for finals not yet cached. Returns (added, remaining)."""
    live = fetch is None
    fetch = fetch or (lambda gid: _get(SUMMARY_URL.format(gid=gid)))
    names = name_to_code()
    path = raw / f"player_box_{season}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=BOX_COLUMNS)
    done = set(have.game_id.astype(str))
    todo = [g for g in completed_games(season, sdv).to_dict("records") if g["game_id"] not in done]
    new_rows: list[dict] = []
    for g in todo[:limit]:
        new_rows += parse_box(fetch(g["game_id"]), g, names)
        if live:
            time.sleep(SLEEP_S)
    if new_rows:
        out = pd.concat([have, pd.DataFrame(new_rows, columns=BOX_COLUMNS)], ignore_index=True)
        out = out.sort_values(["date", "game_id", "team", "starter", "player"],
                              ascending=[True, True, True, False, True]).reset_index(drop=True)
        raw.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return len(todo[:limit]), max(0, len(todo) - limit)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True, help="the year the season ends")
    ap.add_argument("--limit", type=int, default=MAX_GAMES_PER_RUN)
    a = ap.parse_args(argv)

    report = parse_injuries(_get(INJURIES_URL), name_to_code())
    changed = write_if_changed(RAW / "injuries_current.json", report)
    n_inj = sum(len(v) for k, v in report.items() if not k.startswith("_"))
    print(f"injuries: {n_inj} players on {len([k for k in report if not k.startswith('_')])} "
          f"teams ({'updated' if changed else 'unchanged'})")
    if report.get("_unmatched"):
        print(f"  UNMATCHED team names (add to coverline/leagues/nba/teams.py): {report['_unmatched']}")

    added, remaining = update_box(a.season, limit=a.limit)
    print(f"box scores {a.season}: {added} game(s) added, {remaining} still to fetch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
