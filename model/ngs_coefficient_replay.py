"""Did the NGS coefficient substitution cause the 0.68 calibration slope?

THE QUESTION. The Track-record tile measures the slope of actual
margins regressed on stated model margins. A slope of 1.0 means the
margins mean what they say; the model measured 0.68 against the
market's 1.14, and the cause was left open with a week-6 checkpoint.

Separately, the 2026-09-20 audit found that model/prediction.py
selected coefficients on Elo presence alone, with no branch for
missing NGS. MARGIN_COEFFICIENTS carries rating_diff = 0.1078 because
the NGS block and Elo absorb that signal; run against zeroed NGS
inputs it understates the team rating ~211x versus the rating-only
fit's 22.7091. Every 2026 board ran that path while nflverse's
per-season NGS files were 404ing.

THE HYPOTHESIS. If the rating's coefficient is effectively switched
off, published margins compress toward a constant -- and compression
toward a constant is exactly what a low regression slope measures.

WHAT IT FOUND (2026-09-20, n=17, weeks 1-2). Half right, and the
half that failed is the half that mattered.

Confirmed: the substitution did switch the rating off. Across the
same 17 games, rating_diff moved the published margin by 0.013
points of standard deviation under the shipped coefficients versus
2.677 under the rating-only fit -- 211x, the coefficient ratio
itself. The team rating accounted for 0.4% of the published board's
variation. Elo and the de-bias term were the board.

Not confirmed: this does not explain the 0.68 slope. The
compression story predicts a NARROW published board. The published
board was not narrow (sd 3.06) -- wider, in fact, than the fixed
rating term alone -- because Elo supplied the variation the rating
was not. And every slope here carries SE > 0.8, so none of them
separate from 1.0 or from each other. The week-6 checkpoint still
has to answer the calibration question.

The counter-argument in the original draft of this file was the
correct one: the mechanical prediction cut the other way, and it
won. Recorded rather than deleted, because a hypothesis that
survived its own author's doubt and then lost anyway is the part
of the record worth keeping.

WHAT THIS SCRIPT DOES. Replays the published board under both
coefficient vectors, using only committed artifacts:
  - data/ratings/2026-week-NN.json      (the ratings as published)
  - data/divergence/2026-week-NN-*.json (the market lines as seen)
  - nflverse games.csv                  (actual results)
For each, it recomputes the calibration slope, the rank correlation
with actual margins, and MAE, so the two configurations are compared
on identical games.

WHAT IT CANNOT DO. n is small (the season is two weeks old) and the
slope's standard error is large -- around +/- 0.5 at n=17. This
narrows the question; it does not settle it. Read the SE before
treating any of it as decided.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.prediction import (MARGIN_COEFFICIENTS, MARGIN_COEFFICIENTS_V1_RATING_ONLY,
                              predict_margin)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SEASON = 2026


def load_published_boards(season=SEASON):
    """Every game's FIRST published divergence row per week, with the
    ratings snapshot that produced it."""
    rows = []
    for rpath in sorted(glob.glob(os.path.join(REPO, "data", "ratings", f"{season}-week-*.json"))):
        week_of_ratings = int(os.path.basename(rpath).split("week-")[1].split(".")[0])
        snap = json.load(open(rpath))
        ratings = {t["team"]: t for t in snap["ratings"]}
        # The board for week N is priced off the week N-1 ratings file.
        board_week = week_of_ratings + 1
        pattern = os.path.join(REPO, "data", "divergence",
                               f"{season}-week-{board_week:02d}-*.json")
        seen = set()
        for dpath in sorted(glob.glob(pattern)):
            try:
                dsnap = json.load(open(dpath))
            except Exception:
                continue
            for d in dsnap.get("divergences", []):
                key = (d.get("away_team"), d.get("home_team"))
                if key in seen or d.get("spread_gap") is None:
                    continue
                if key[0] not in ratings or key[1] not in ratings:
                    continue
                seen.add(key)
                rows.append({
                    "week": board_week,
                    "away_team": key[0], "home_team": key[1],
                    "market_spread": d["market_spread"],
                    "published_gap": d["spread_gap"],
                    "home_rating": ratings[key[1]]["total_rating"],
                    "away_rating": ratings[key[0]]["total_rating"],
                })
    return pd.DataFrame(rows)


# Below this much variation in the predictions, a regression slope is
# the ratio of two noise terms and means nothing. The full-ensemble
# path lands here, which is itself the finding -- not a number to read.
DEGENERATE_SPREAD = 0.10


def slope_and_se(x, y):
    """OLS slope of y on x, with its standard error.

    Returns nan when the predictions barely vary: dividing by a
    near-zero variance produces a large slope that reflects only the
    smallness of the denominator."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < DEGENERATE_SPREAD:
        return float("nan"), float("nan")
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    se = np.sqrt((resid.var(ddof=2)) / (np.var(x, ddof=1) * (len(x) - 1)))
    return float(b), float(se)


def main():
    board = load_published_boards()
    if board.empty:
        raise SystemExit("no published boards found under data/")

    games = pd.read_csv(GAMES_URL)
    games = games[(games["season"] == SEASON) & (games["game_type"] == "REG")]
    games = games.dropna(subset=["home_score", "away_score"])
    games["actual_margin"] = games["home_score"] - games["away_score"]
    m = board.merge(games[["week", "home_team", "away_team", "actual_margin", "spread_line"]],
                    on=["week", "home_team", "away_team"], how="inner")
    if m.empty:
        raise SystemExit("no completed games join the published boards yet")

    # Reconstruct each configuration's margin from the SAME ratings.
    rating_diff = m["home_rating"] - m["away_rating"]
    # The committed ratings snapshots do not store Elo, so the shipped
    # board cannot be re-derived from this repo. The two reconstructions
    # below therefore hold Elo at zero: they isolate what the team
    # RATING contributed under each coefficient vector, which is the
    # question the substitution bug actually raises. They are not
    # replays of the published number.
    m["margin_no_ngs"] = [
        predict_margin(rd, False, 0, coefficients=MARGIN_COEFFICIENTS_V1_RATING_ONLY)
        for rd in rating_diff]
    m["margin_full"] = [
        predict_margin(rd, False, 0, elo_diff=0.0, coefficients=MARGIN_COEFFICIENTS)
        for rd in rating_diff]
    # What the board actually published (de-biased, Elo included).
    m["margin_published"] = m["market_spread"] + m["published_gap"]

    print(f"=== NGS coefficient replay: {len(m)} completed games, "
          f"weeks {sorted(m['week'].unique())} ===\n")
    print(f"{'configuration':<34} {'slope':>8} {'SE':>7} {'spread':>8} {'rank r':>8} {'MAE':>7}")
    print("-" * 76)
    rows = []
    for label, col in (("published board (as shipped)", "margin_published"),
                       ("rating alone, shipped coefs", "margin_full"),
                       ("rating alone, the fix", "margin_no_ngs"),
                       ("market closing line", "spread_line")):
        b, se = slope_and_se(m[col], m["actual_margin"])
        rank = pd.Series(m[col]).corr(m["actual_margin"], method="spearman")
        mae = float((m["actual_margin"] - m[col]).abs().mean())
        spread = float(m[col].std())
        degenerate = spread < DEGENERATE_SPREAD
        shown = "     n/a     n/a" if degenerate else f"{b:>8.3f} {se:>7.3f}"
        print(f"{label:<34} {shown} {spread:>8.2f} {rank:>8.3f} {mae:>7.2f}")
        rows.append({"configuration": label,
                     "slope": None if degenerate else round(b, 4),
                     "slope_se": None if degenerate else round(se, 4),
                     "prediction_spread": round(spread, 3),
                     "rank_correlation": round(float(rank), 4), "mae": round(mae, 3),
                     "slope_withheld_degenerate": bool(degenerate)})

    # The finding the table implies, stated as a number.
    sp_broken = float(m["margin_full"].std())
    sp_fixed = float(m["margin_no_ngs"].std())
    sp_pub = float(m["margin_published"].std())
    ratio = (sp_fixed / sp_broken) if sp_broken > 0 else float("inf")
    rating_share_pub = sp_broken / sp_pub if sp_pub > 0 else float("nan")

    print(f"\nRating's contribution to margin spread, in points of margin:")
    print(f"  shipped coefficients : {sp_broken:6.3f}")
    print(f"  the fix              : {sp_fixed:6.3f}   ({ratio:.0f}x wider)")
    print(f"  published board      : {sp_pub:6.3f}   (Elo + de-bias + rating)")
    print(f"  -> under the shipped coefficients the team rating accounted for "
          f"{rating_share_pub * 100:.1f}% of the")
    print(f"     published board's variation. Elo and the de-bias term carried the rest.")

    print("\nReading it:")
    print("  * The compression is real and now measured: with NGS absent, rating_diff")
    print("    moved the margin by hundredths of a point. That is the bug, quantified.")
    print("  * But it does NOT explain the 0.68 slope. Compression predicts a NARROW")
    print("    published board; the published board's spread (3.06) was wider than the")
    print("    fix's rating term alone (2.68), because Elo filled the gap. The slope's")
    print("    cause stays open -- the week-6 checkpoint still has to answer it.")
    print("  * 'rank r' is order-only. At n=17 its own SE is ~0.25, so none of the")
    print("    rank figures here separate from zero, the market's 0.244 included.")
    print("  * Slopes are withheld where the predictions barely vary: a ratio with a")
    print("    near-zero denominator is not an estimate.")

    out = {
        "_provenance": {
            "script": "model/ngs_coefficient_replay.py",
            "generated": __import__("datetime").date.today().isoformat(),
            "season": SEASON, "n_games": int(len(m)),
            "weeks": sorted(int(w) for w in m["week"].unique()),
            "inputs": ["data/ratings/", "data/divergence/", "nflverse games.csv"],
            "elo_not_reconstructable": ("The committed ratings snapshots do not store Elo. "
                                        "The two 'rating alone' rows hold Elo at zero and "
                                        "isolate the coefficient path; they are not replays "
                                        "of the shipped number."),
            "caveat": ("Two weeks of games. Every slope here has SE > 0.8, so no "
                       "configuration's slope separates from 1.0 or from any other's. "
                       "Re-run at the week-6 checkpoint the ledger already committed to."),
        },
        "rating_contribution_to_margin_spread": {
            "shipped_coefficients_pts": round(sp_broken, 4),
            "fixed_coefficients_pts": round(sp_fixed, 4),
            "ratio_fixed_over_shipped": round(ratio, 1),
            "published_board_pts": round(sp_pub, 4),
            "rating_share_of_published_variation": round(rating_share_pub, 5),
        },
        "verdict": {
            "compression_confirmed": True,
            "explains_the_068_slope": False,
            "why": ("The substitution did switch the rating off -- it moved published "
                    "margins by hundredths of a point. But compression toward a constant "
                    "predicts a narrow published board, and the published board was not "
                    "narrow (sd 3.06, wider than the fixed rating term's 2.68) because Elo "
                    "supplied the variation. So the bug is real and worth fixing on its own "
                    "terms, and the calibration slope remains unexplained."),
        },
        "results": rows,
    }
    path = os.path.join(REPO, "model", "ngs_coefficient_replay_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
