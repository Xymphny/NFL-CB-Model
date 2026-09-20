"""Grade the projection engine's watch-mode prop opinions.

WHY THIS EXISTS. The engine publishes roughly 380 probability claims
a week onto the live board -- by volume, the largest claim surface in
the system -- and until 2026-09-20 nothing graded any of them, while
the board's own copy told readers they were graded. That is the third
pillar ("every claim is graded publicly, wins and losses alike")
failing at the place it matters most.

It is also the engine's own on-ramp rule: watch-mode opinions earn
verdict authority "through live graded evidence, or not at all". That
rule was unreachable as written, because no machinery existed to
produce the evidence.

WHAT IT DOES. Joins each week's published prop opinions to nflverse
box scores and scores them with the same instruments the held-out
gate used -- log-loss, Brier, and bucket calibration -- split by
market and by engine version, so a September chip is never pooled
with an October one from different code.

WHAT IT REFUSES TO DO. It does not grade the market's EV rows: those
are book-vs-book dispersion, not a claim this project made. It does
not grade an opinion whose game has not finished. It does not
silently drop a player it cannot resolve -- unmatched names are
counted and published, because a grader that quietly discards its
misses flatters itself.

Output: data/prop_grades/{season}-week-NN.json, plus a rolling
data/prop_grades/summary.json the site can read.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_DATA_PATH = os.environ.get("REPO_DATA_PATH", "./data")
BUCKETS = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60), (0.60, 1.01))


def _logloss_brier(rows):
    import math
    if not rows:
        return None, None
    ll = 0.0
    br = 0.0
    for p, y in rows:
        p = min(max(p, 1e-9), 1 - 1e-9)
        ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        br += (p - y) ** 2
    return round(ll / len(rows), 5), round(br / len(rows), 5)


def load_actuals(season, week):
    """Per-player actuals for one week from nflverse box scores."""
    import pandas as pd
    from model.player_projection import STATS_URL, _norm_name

    df = pd.read_parquet(STATS_URL.format(season=season),
                         columns=["player_display_name", "team", "season", "week", "season_type",
                                  "rushing_yards", "receiving_yards", "passing_yards",
                                  "rushing_tds", "receiving_tds"])
    df = df[(df["season_type"] == "REG") & (df["week"] == week)].fillna(0)
    out = {}
    for r in df.to_dict("records"):
        out[_norm_name(r["player_display_name"])] = {
            "rush_yds": float(r["rushing_yards"]),
            "rec_yds": float(r["receiving_yards"]),
            "pass_yds": float(r["passing_yards"]),
            "scored": float(r["rushing_tds"] + r["receiving_tds"]) > 0,
        }
    return out


def grade_week(season, week, data_dir=REPO_DATA_PATH):
    from model.player_projection import _norm_name

    props_path = os.path.join(data_dir, "props", f"{season}-week-{week:02d}.json")
    if not os.path.exists(props_path):
        print(f"[grade_props] no props file for {season} week {week}")
        return None
    payload = json.load(open(props_path))
    actuals = load_actuals(season, week)
    if not actuals:
        # "No box scores yet" is not "no player matched". Writing a
        # 0-graded report here would look like a grading result when
        # it is an absence of data -- the same confusion the board's
        # empty state was fixed for.
        print(f"[grade_props] {season} week {week} has no published box scores yet; "
              f"nothing graded (this is not a miss, the week is unfinished)")
        return None

    graded = []
    unmatched = []
    for game, g in payload.get("games", {}).items():
        for mk_key, players in g.get("markets", {}).items():
            for player, row in players.items():
                # A chip that was withdrawn mid-week was still published,
                # so it is graded from its frozen copy.
                op = row.get("model") or row.get("model_frozen")
                if not op:
                    continue
                a = actuals.get(_norm_name(player))
                if a is None:
                    unmatched.append(player)
                    continue
                if op.get("kind") == "score":
                    p, y = op.get("p_score"), 1.0 if a["scored"] else 0.0
                    line = None
                elif op.get("kind") == "yards":
                    line = row.get("line")
                    if line is None:
                        continue
                    p = op.get("p_over")
                    actual = a.get(op.get("market"))
                    if actual is None:
                        continue
                    if actual == line:
                        continue                       # push: no claim resolved
                    y = 1.0 if actual > line else 0.0
                else:
                    continue
                if p is None:
                    continue
                graded.append({
                    "game": game, "player": player, "market": op.get("market"),
                    "engine": op.get("engine"), "rz": op.get("rz"),
                    "line": line, "claimed": p, "hit": y,
                    "withdrawn": bool(row.get("model_frozen") and not row.get("model")),
                })

    by_key = defaultdict(list)
    for r in graded:
        by_key[(r["engine"], r["market"])].append((r["claimed"], r["hit"]))

    report = {
        "season": season, "week": week,
        "engine_version_at_publish": payload.get("engine_version"),
        "n_graded": len(graded),
        "n_unmatched_players": len(unmatched),
        "unmatched_sample": sorted(set(unmatched))[:20],
        "note": ("Watch-mode grading. These are the engine's own probability claims scored "
                 "against results -- NOT the market EV rows, which are book-vs-book dispersion "
                 "rather than a claim this project made. Opinions withdrawn mid-week are graded "
                 "from their frozen copy: publishing a claim and then deleting it does not "
                 "unpublish it."),
        "by_market": [],
        "buckets": [],
        "plays": graded,
    }
    for (engine, market), rows in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        ll, br = _logloss_brier(rows)
        report["by_market"].append({
            "engine": engine, "market": market, "n": len(rows),
            "mean_claimed": round(sum(p for p, _ in rows) / len(rows), 4),
            "actual_rate": round(sum(y for _, y in rows) / len(rows), 4),
            "log_loss": ll, "brier": br,
        })
    for lo, hi in BUCKETS:
        rows = [(r["claimed"], r["hit"]) for r in graded if lo <= r["claimed"] < hi]
        if len(rows) >= 10:
            report["buckets"].append({
                "lo": lo, "hi": hi, "n": len(rows),
                "claimed": round(sum(p for p, _ in rows) / len(rows), 4),
                "actual": round(sum(y for _, y in rows) / len(rows), 4),
            })

    # A REPORT OF NOTHING IS NOT A RESULT (2026-09-20). load_actuals
    # returning rows is not the same as the week being over: run
    # mid-week, it returns the players from whichever games have
    # finished, which defeats the "no box scores yet" guard above. On
    # 2026-09-20, week 2 had one final of sixteen, its players were not
    # on the props board, and this graded 0 of 386 and still wrote a
    # file and returned its path -- which every caller reads as
    # success. Write nothing and say which of the two it was.
    if not graded:
        share = len(graded) / (len(graded) + len(unmatched)) if unmatched else 0.0
        print(f"[grade_props] {season} week {week}: 0 of "
              f"{len(graded) + len(unmatched)} published opinions matched a box "
              f"score ({share:.0%}). Nothing written. This is an unfinished "
              f"week if the games have not been played, and a real miss if "
              f"they have -- the caller decides with the slate in hand.")
        return None

    out_dir = os.path.join(data_dir, "prop_grades")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{season}-week-{week:02d}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[grade_props] {out_path}: {len(graded)} claims graded, "
          f"{len(unmatched)} players unmatched")
    return out_path


def rebuild_summary(season, data_dir=REPO_DATA_PATH):
    """Roll every graded week into one file the site can read."""
    import glob
    rows = []
    weeks = []
    for path in sorted(glob.glob(os.path.join(data_dir, "prop_grades", f"{season}-week-*.json"))):
        rep = json.load(open(path))
        weeks.append(rep["week"])
        rows.extend((p["claimed"], p["hit"], p.get("engine"), p.get("market")) for p in rep["plays"])
    if not rows:
        return None
    by_key = defaultdict(list)
    for p, y, engine, market in rows:
        by_key[(engine, market)].append((p, y))
    summary = {
        "season": season, "weeks_graded": sorted(weeks), "n_claims": len(rows),
        "note": ("Live graded record of the projection engine's watch-mode opinions. The engine "
                 "earns verdict authority here or not at all."),
        "by_market": [],
        "buckets": [],
    }
    for (engine, market), rs in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        ll, br = _logloss_brier(rs)
        summary["by_market"].append({
            "engine": engine, "market": market, "n": len(rs),
            "mean_claimed": round(sum(p for p, _ in rs) / len(rs), 4),
            "actual_rate": round(sum(y for _, y in rs) / len(rs), 4),
            "log_loss": ll, "brier": br,
        })
    for lo, hi in BUCKETS:
        rs = [(p, y) for p, y, _, _ in rows if lo <= p < hi]
        if len(rs) >= 10:
            summary["buckets"].append({
                "lo": lo, "hi": hi, "n": len(rs),
                "claimed": round(sum(p for p, _ in rs) / len(rs), 4),
                "actual": round(sum(y for _, y in rs) / len(rs), 4),
            })
    out_path = os.path.join(data_dir, "prop_grades", "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[grade_props] summary: {len(rows)} claims across weeks {sorted(weeks)}")
    return out_path


def main():
    season = int(os.environ.get("SEASON", 0)) or None
    week = int(os.environ.get("WEEK", 0)) or None
    if season is None or week is None:
        from ingest.nfl_schedules import get_next_upcoming_week
        from datetime import datetime, timezone
        season = season or datetime.now(timezone.utc).year
        # Grade the week that just finished.
        week = week or max(1, (get_next_upcoming_week(season) or 2) - 1)
    paths = [p for p in (grade_week(season, week), rebuild_summary(season)) if p]
    if os.environ.get("GIT_REPO_URL"):
        from deploy.git_utils import git_commit_and_push
        for p in paths:
            git_commit_and_push(p, commit_message=f"Prop grades: {season} week {week}")


if __name__ == "__main__":
    main()
