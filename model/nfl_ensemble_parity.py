#!/usr/bin/env python3
"""Parity gate: the core's full ensemble against the legacy board, week 3.

    python3 model/nfl_ensemble_parity.py            # network: NGS, schedule

Writes data/nfl_ensemble_parity.json. scripts/export_board.py prices NFL on
the full ensemble only while that file says `passed: true`; otherwise it
keeps the rating-only path and shows the banner (accuracy plan, 1a).

THE GATE (docs/briefs/2026-09-27-accuracy-plan.md)
For every week-3 game in data/divergence/2026-week-03-*.json the core's
margin matches the legacy margin to within 0.05 points: same feature_values,
same coefficient set, same de-bias. Then the same on the two newest snapshots
with the features FETCHED by the core rather than read from the board.

  A. recorded   Every row of every week-3 board: the recorded feature_values
                through the recorded coefficient set (nfl.model.predict_margin)
                plus the slate offset execution/board.slate_debias measures
                from that board's own open rows. Tests the port of the
                arithmetic and of the de-bias.
  B. fetched    The two newest boards: EnsembleWeekSource fetches NGS and Elo
                and reads the ratings version current at the board's
                computed_at. Every term is compared, not only the sum, so a
                failure names the term that differs.

WHY B RUNS ONLY ON THE NEWEST BOARDS
nflverse revises its NGS release in place. Today's release reproduces the
Sunday boards' recorded NGS values exactly and the Wednesday boards' only
for 13 of 16 games (GB/ATL, DET/NYJ, CHI/PHI moved between releases). That
is data vintage, not code, and A covers those boards with the values they
actually used.

THE RAMS (ADR 0004)
Legacy prices DEN/LA rating-only because its NGS join misses LAR; the core
normalises. B runs twice: `legacy_vocabulary` (NGS keyed as legacy keys it)
must match every game, and `core_vocabulary` (what the board will publish)
reports the Rams game's vector change and the offset shift it causes on
every other game. The gate is the first; the second is the announced
publishing change ADR 0004 names as its revisit trigger.
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coverline.execution.board import slate_debias  # noqa: E402
from coverline.leagues.nfl import ensemble as EN  # noqa: E402
from coverline.leagues.nfl.model import GameFeatures, predict_margin  # noqa: E402

OUT = ROOT / "data" / "nfl_ensemble_parity.json"
TOLERANCE = 0.05
SEASON, WEEK = 2026, 3
TERMS = ("rating_diff", "rest_diff", "cpoe_diff", "separation_diff",
         "yac_oe_diff", "ryoe_diff", "elo_diff", "is_neutral_site")


def boards() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(
        str(ROOT / "data" / "divergence" / f"{SEASON}-week-{WEEK:02d}-*.json")))


def recorded_features(d: dict) -> GameFeatures:
    fv = d["feature_values"]
    return GameFeatures(
        rating_diff=fv["rating_diff"], rest_diff=fv["rest_diff"],
        cpoe_diff=fv["cpoe_diff"], separation_diff=fv["separation_diff"],
        yac_oe_diff=fv["yac_oe_diff"], ryoe_diff=fv["ryoe_diff"],
        elo_diff=fv["elo_diff"] or 0.0, is_neutral_site=fv["is_neutral_site"],
        ngs_present=d["coefficient_set"] == EN.FULL)


def _key(d):
    return (d["home_team"], d["away_team"], d["market_spread"], d["spread_gap"])


def check_recorded(paths: list[Path]) -> dict:
    """A: every row, every week-3 board."""
    offsets, rows, worst = {}, [], 0.0
    for p in paths:
        b = json.loads(p.read_text())
        open_ = [d for d in b["divergences"] if d.get("line_status") != "closed"]
        pairs = [(predict_margin(recorded_features(d)), d["market_spread"]) for d in open_]
        off, how = slate_debias(pairs)
        offsets[p.name] = {"recorded": b["debias_offsets"][0], "core": off, "how": how,
                           "diff": off - b["debias_offsets"][0]}
        for d in b["divergences"]:
            # A closed row is carried verbatim from the last board before
            # kickoff, and carries that board's offset.
            o = off
            if d.get("line_status") == "closed":
                src = next(q for q in reversed(paths) if q.name < p.name and any(
                    _key(x) == _key(d) and x.get("line_status") != "closed"
                    for x in json.loads(q.read_text())["divergences"]))
                o = offsets[src.name]["core"]
            core = predict_margin(recorded_features(d)) + o
            legacy = d["market_spread"] + d["spread_gap"]
            diff = core - legacy
            worst = max(worst, abs(diff))
            rows.append({"board": p.name, "game": f"{d['away_team']}@{d['home_team']}",
                         "coefficient_set": d["coefficient_set"],
                         "line_status": d.get("line_status"),
                         "legacy": round(legacy, 6), "core": round(core, 6),
                         "diff": round(diff, 9)})
    return {"rows": len(rows), "boards": len(paths), "max_abs_diff": worst,
            "passed": worst <= TOLERANCE, "offsets": offsets,
            "failures": [r for r in rows if abs(r["diff"]) > TOLERANCE]}


def _legacy_vocabulary(frame: pd.DataFrame) -> pd.DataFrame:
    """The NGS frame as model/prediction joins it: Rams as LAR, so absent."""
    return frame.rename(index={"LA": "LAR"})


def check_fetched(paths: list[Path], schedule: pd.DataFrame, ngs: pd.DataFrame,
                  history: pd.DataFrame) -> dict:
    """B: the two newest boards, features fetched by the core."""
    out = {}
    for vocab in ("legacy_vocabulary", "core_vocabulary"):
        frame = _legacy_vocabulary(ngs) if vocab == "legacy_vocabulary" else ngs
        src = EN.EnsembleWeekSource.load(
            SEASON, WEEK, schedule, ngs_loader=lambda s, w: frame,
            history_loader=lambda s: history)
        per_board = {}
        for p in paths:
            b = json.loads(p.read_text())
            asof = b["computed_at"]
            open_ = [d for d in b["divergences"] if d.get("line_status") != "closed"]
            games = []
            for d in open_:
                gid = f"{SEASON}-W{WEEK:02d}-{d['home_team']}-{d['away_team']}"
                f, cset, reason = src.resolve(gid, asof)
                games.append((d, f, cset, reason))
            off, how = slate_debias([(predict_margin(f), d["market_spread"])
                                     for d, f, _, _ in games])
            rows, worst = [], 0.0
            for d, f, cset, reason in games:
                rec = recorded_features(d)
                terms = {t: float(getattr(f, t)) - float(getattr(rec, t)) for t in TERMS
                         if t != "elo_diff" or (f.ngs_present and rec.ngs_present)}
                core = predict_margin(f) + off
                legacy = d["market_spread"] + d["spread_gap"]
                diff = core - legacy
                worst = max(worst, abs(diff))
                rows.append({
                    "game": f"{d['away_team']}@{d['home_team']}",
                    "coefficient_set": {"legacy": d["coefficient_set"], "core": cset},
                    "core_reason": reason,
                    "term_diffs": {k: round(v, 9) for k, v in terms.items()},
                    "legacy": round(legacy, 6), "core": round(core, 6),
                    "diff": round(diff, 9)})
            per_board[p.name] = {
                "asof": asof, "offset": {"recorded": b["debias_offsets"][0], "core": off,
                                         "how": how},
                "max_abs_diff": worst,
                "coefficient_set_mismatches": [r["game"] for r in rows
                                               if r["coefficient_set"]["legacy"]
                                               != r["coefficient_set"]["core"]],
                "failures": [r for r in rows if abs(r["diff"]) > TOLERANCE],
                "games": rows}
        out[vocab] = {"boards": per_board,
                      "passed": all(v["max_abs_diff"] <= TOLERANCE
                                    and not v["coefficient_set_mismatches"]
                                    for v in per_board.values())}
    return out


def main() -> int:
    from coverline.leagues.nfl.live import load_schedule
    paths = boards()
    a = check_recorded(paths)
    schedule = load_schedule([SEASON])
    ngs = EN._default_ngs(SEASON, WEEK)
    history = EN._default_history(SEASON)
    b = check_fetched(paths[-2:], schedule, ngs, history)
    core = b["core_vocabulary"]["boards"]
    rams = {name: {"games": [g for g in v["games"]
                             if g["coefficient_set"]["legacy"] != g["coefficient_set"]["core"]],
                   "offset_shift": v["offset"]["core"] - v["offset"]["recorded"],
                   "max_abs_diff_other_games": max(
                       (abs(g["diff"]) for g in v["games"]
                        if g["coefficient_set"]["legacy"] == g["coefficient_set"]["core"]),
                       default=0.0)}
            for name, v in core.items()}
    passed = a["passed"] and b["legacy_vocabulary"]["passed"]
    art = {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": SEASON, "week": WEEK, "tolerance_points": TOLERANCE,
        "passed": passed,
        "gate": ("A (recorded values, every week-3 board) and B under the legacy NGS "
                 "vocabulary (fetched values, two newest boards) both within tolerance, "
                 "with the same coefficient set on every game"),
        "A_recorded": a,
        "B_fetched": b,
        "adr_0004_rams": {
            "note": ("What the board publishes under the core's NGS vocabulary: the Rams "
                     "game moves to the full ensemble, and the slate median it feeds moves "
                     "every other game by the offset shift. Announced, not gated (ADR 0004)."),
            "boards": rams},
        "script": "model/nfl_ensemble_parity.py",
    }
    OUT.write_text(json.dumps(art, indent=1, default=float) + "\n")
    print(f"A recorded: {a['rows']} rows on {a['boards']} boards, max |diff| "
          f"{a['max_abs_diff']:.2e}, offsets max |diff| "
          f"{max(abs(v['diff']) for v in a['offsets'].values()):.2e}")
    for vocab, v in b.items():
        for name, r in v["boards"].items():
            print(f"B {vocab:18s} {name}: max |diff| {r['max_abs_diff']:.4f}, "
                  f"offset core {r['offset']['core']:+.4f} vs recorded "
                  f"{r['offset']['recorded']:+.4f}, vector mismatches "
                  f"{r['coefficient_set_mismatches']}")
    print(f"GATE {'PASSED' if passed else 'FAILED'} -> {OUT.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
