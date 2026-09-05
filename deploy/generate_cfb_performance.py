"""
CFB self-grading -> data/cfb_performance.json, the CFB Track record's
data source (the tab is already league-scoped and 404s to placeholders
until this file exists).

Mirrors deploy/generate_performance.py's contract exactly (summarize()
is imported from it, so both leagues' KPIs are computed by the same
code) with the CFB-specific differences:

- Thresholds: play >= 5, lean >= 3 (backtest-derived; see
  model/cfb_backtest_2023_results.json).
- Weeks 1-4 grade at LEAN stakes regardless of edge size -- the board
  caps early-season verdicts at Lean per the held-out evidence, and
  the record must grade what the board actually showed, never a
  shadow-book of what raw edges implied.
- Spreads only (no CFB totals model).
- Entry line: earliest snapshot per week (same philosophy as NFL).
  Closing line: LAST snapshot's row per game -- with the kickoff
  freeze in place, that row is the frozen pregame close by
  construction.
- Finals: ESPN's college scoreboard by date (dates harvested from the
  snapshots' own kickoff fields; FBS group). Names bridge through the
  same suffix-strip mapper the odds watch uses. NOT testable live
  from the build sandbox (ESPN blocked there) -- parser is
  unit-tested against a captured response shape in
  tests/test_parsers.py, and everything soft-fails: a week grades
  only when its finals resolve.
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from deploy.generate_performance import summarize, STAKES, WIN_PAYOUT
from deploy.cfb_odds_watch import map_odds_names_to_ratings

PLAY_GAP = 5.0
LEAN_GAP = 3.0
EARLY_WEEK_MAX = 4
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=300&dates={date}"


def load_snapshots_by_week(data_dir, season):
    """{week: [snapshots earliest->latest]} for week >= 1."""
    by_week = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "cfb_divergence", f"{season}-week-*.json"))):
        m = re.search(rf"{season}-week-(\d+)", os.path.basename(path))
        if not m or int(m.group(1)) < 1:
            continue
        with open(path) as f:
            by_week[int(m.group(1))].append(json.load(f))
    return by_week


def parse_scoreboard_finals(payload):
    """[(espn_display_home, espn_display_away, home_score, away_score)]
    for COMPLETED games only. Pure parser, unit-tested."""
    out = []
    for ev in payload.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not ((comp.get("status") or {}).get("type") or {}).get("completed"):
            continue
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        home, away = sides.get("home") or {}, sides.get("away") or {}
        try:
            out.append(((home.get("team") or {}).get("displayName"),
                        (away.get("team") or {}).get("displayName"),
                        int(home.get("score")), int(away.get("score"))))
        except (TypeError, ValueError):
            continue
    return out


def fetch_finals_for_week(snapshots, team_names, timeout=20):
    """{(home_ratings_name, away_ratings_name): (home_score, away_score)}
    by querying ESPN for each kickoff date present in the snapshots."""
    dates = set()
    for snap in snapshots:
        for d in snap.get("divergences", []):
            ko = d.get("kickoff")
            if ko:
                dates.add(ko[:10].replace("-", ""))
    finals = {}
    for date in sorted(dates):
        try:
            payload = requests.get(SCOREBOARD_URL.format(date=date), timeout=timeout).json()
        except Exception as e:
            print(f"[cfb_performance] scoreboard soft-fail {date}: {e}")
            continue
        rows = parse_scoreboard_finals(payload)
        pseudo = [{"home_team": h, "away_team": a} for h, a, _, _ in rows]
        mapping, _ = map_odds_names_to_ratings(pseudo, team_names)
        for h, a, hs, as_ in rows:
            mh, ma = mapping.get(h), mapping.get(a)
            if mh and ma:
                finals[(mh, ma)] = (hs, as_)
    return finals


def grade_week(week, snapshots, finals):
    earliest, latest = snapshots[0], snapshots[-1]
    close_by_game = {(d["home_team"], d["away_team"]): d.get("market_spread")
                     for d in latest.get("divergences", [])}
    plays = []
    for d in earliest.get("divergences", []):
        gap = d.get("spread_gap")
        if gap is None:
            continue
        edge = abs(gap)
        if edge < LEAN_GAP:
            continue
        tier = "lean" if week <= EARLY_WEEK_MAX else ("play" if edge >= PLAY_GAP else "lean")

        key = (d["home_team"], d["away_team"])
        if key not in finals:
            continue
        hs, as_ = finals[key]
        actual_margin = hs - as_
        line = d["market_spread"]
        picked_home = gap > 0
        if actual_margin == line:
            result, units = "push", 0.0
        elif (actual_margin > line) == picked_home:
            result, units = "win", STAKES[tier] * WIN_PAYOUT
        else:
            result, units = "loss", -STAKES[tier]
        close = close_by_game.get(key)
        clv = None if close is None else round(close - line if picked_home else line - close, 2)
        side = d["home_team"] if picked_home else d["away_team"]
        side_line = -line if picked_home else line
        plays.append({
            "week": week, "market": "spread", "tier": tier,
            "label": f"{side} {side_line:+g}",
            "line": line, "close": close, "clv": clv,
            "result": result, "units": round(units, 3),
            "model_margin": round(line + gap, 2), "market_spread": line,
            "actual_margin": actual_margin,
        })
    return plays


def generate(data_dir, season):
    by_week = load_snapshots_by_week(data_dir, season)
    if not by_week:
        print("[cfb_performance] no CFB snapshots yet; nothing to grade")
        return None
    # Team-name universe for the finals mapper: every team on any board.
    team_names = sorted({d[t] for snaps in by_week.values() for s in snaps
                         for d in s.get("divergences", []) for t in ("home_team", "away_team")})
    plays = []
    for week in sorted(by_week):
        finals = fetch_finals_for_week(by_week[week], team_names)
        graded = grade_week(week, by_week[week], finals)
        print(f"[cfb_performance] week {week}: {len(graded)} plays graded "
              f"({len(finals)} finals resolved)")
        plays.extend(graded)
    if not plays:
        print("[cfb_performance] no completed graded plays yet; not writing a file")
        return None
    perf = summarize(plays)
    perf["plays"] = sorted(plays, key=lambda p: (-p["week"],))
    perf["league"] = "CFB"
    perf["season"] = season
    out_path = os.path.join(data_dir, "cfb_performance.json")
    with open(out_path, "w") as f:
        json.dump(perf, f, indent=2)
    n = perf["ats_wins"] + perf["ats_losses"]
    print(f"[cfb_performance] wrote {out_path}: {perf['ats_wins']}-{perf['ats_losses']} "
          f"({n} graded), {perf['units']:+.2f}u, avg CLV {perf['avg_clv']}")
    return out_path


if __name__ == "__main__":
    from datetime import datetime
    generate(os.environ.get("REPO_DATA_PATH", "./data"),
             int(os.environ.get("SEASON", datetime.now().year)))
