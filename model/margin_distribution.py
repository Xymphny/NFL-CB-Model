"""
Two empirical replacements for the normal-CDF approximation, exported
as data/margin_dist.json for both backend and dashboard use:

1. ATS residual distribution: (actual margin - closing spread) over
   every nflverse game with a real closing line, 2010-2024. This is
   the real shape of scoring -- mass spiked on key margins -- and
   drives push probabilities and key-number math.

2. Calibrated edge -> cover probability: a logistic fit of ACTUAL
   cover outcomes against the model's edge vs the close, fit on the
   held-out full-ensemble backtest games (2022-2023). This is the
   honest answer to "when the model sees a 4-point edge, how often
   does that side really cover" -- and it is far flatter than the
   normal approximation, which is exactly why Kelly stakes must use
   it. Refit as real graded seasons accumulate.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DIST_SEASONS = range(2010, 2025)
BACKTEST_RESULTS = os.path.join(os.path.dirname(__file__), "ats_backtest_results.csv")


def build_residual_distribution(games=None):
    if games is None:
        games = pd.read_csv(GAMES_URL)
    g = games[games["season"].isin(DIST_SEASONS)].dropna(subset=["home_score", "away_score", "spread_line"])
    residual = (g["home_score"] - g["away_score"]) - g["spread_line"]

    lo, hi = -40, 40
    clipped = residual.clip(lo, hi)
    # Bin at 0.5 granularity (half-point lines make residuals half-integer).
    bins = np.round(clipped * 2) / 2
    pmf = bins.value_counts(normalize=True).sort_index()

    margins = (g["home_score"] - g["away_score"]).abs()
    margin_pmf = margins.value_counts(normalize=True).sort_index()

    return {
        "n_games": int(len(g)),
        "seasons": [int(DIST_SEASONS.start), int(DIST_SEASONS.stop - 1)],
        "residual_pmf": {str(k): round(float(v), 6) for k, v in pmf.items()},
        "residual_std": round(float(residual.std()), 3),
        "key_margin_mass": {str(int(m)): round(float(margin_pmf.get(m, 0)), 4) for m in [1, 2, 3, 4, 6, 7, 8, 10, 14]},
        # P(push | integer line): pushes can only occur on integer lines,
        # so divide the overall push rate by the share of integer lines.
        "push_prob_integer_line": round(float((residual == 0).mean() / max((g["spread_line"] % 1 == 0).mean(), 1e-9)), 6),
    }


def _fit_logistic(d):
    """1-parameter logistic through 0.5 at edge=0: p = 1/(1+exp(-b*edge)).
    Symmetry is forced deliberately -- with no edge the pick is a coin
    flip by construction, so the intercept is not a free parameter.
    Returns (b, standard_error, n)."""
    from scipy.optimize import minimize_scalar

    x = d["edge"].values
    y = d["covered"].values

    def nll(b):
        p = np.clip(1 / (1 + np.exp(-b * x)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    b = float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)
    p = 1 / (1 + np.exp(-b * x))
    info = float(np.sum(p * (1 - p) * x * x))      # observed information
    se = float(1 / np.sqrt(info)) if info > 0 else float("nan")
    return b, se, int(len(d))


def _log_loss(b, d):
    p = np.clip(1 / (1 + np.exp(-b * d["edge"].values)), 1e-9, 1 - 1e-9)
    y = d["covered"].values
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_edge_calibration():
    """Does the SIZE of the model's edge predict how often it covers?

    This is the layer that turns an opinion into money -- coverProb
    feeds sizeStake, so a curve fitted on noise sizes real bets. It was
    previously fitted AND reported on the same 2022-2023 games with no
    uncertainty attached, and the resulting coefficient shipped as
    though it were a calibrated instrument.

    Measured honestly on 2026-09-20 it does not survive:

      * b = +0.01302 on all 372 graded games, SE 0.03153 -- the 95%
        interval [-0.049, +0.075] contains zero several times over.
      * The sign FLIPS by season: +0.0346 fitting 2022, -0.0187 on
        2023. The relationship is not stable enough to have a sign.
      * Fit on 2022 and graded on 2023 it LOSES TO A COIN FLIP
        (log-loss 0.6959 vs 0.6931).
      * Realized cover is non-monotonic in edge: 52.5% / 47.6% /
        53.2% / 50.0% across the 0-2 / 2-4 / 4-6 / 6+ buckets.

    So the honest verdict is that the NFL model's edge magnitude does
    not yet measurably predict cover probability, and `supported` is
    False. Consumers must treat a False here the way the projection
    engine treats a withheld market: show no probability and size
    flat, rather than quote a number the evidence cannot carry. The
    machinery stays so the claim can be re-tested as graded seasons
    accumulate -- this is a verdict on the current sample, not a
    permanent one.
    """
    df = pd.read_csv(BACKTEST_RESULTS)
    edge = df["model_margin"] - df["spread_line"]
    pushes = df["actual_margin"] == df["spread_line"]
    home_covers = df["actual_margin"] > df["spread_line"]
    picked_home = edge > 0
    d = pd.DataFrame({
        "season": df["season"],
        "edge": edge.abs(),
        "covered": (picked_home == home_covers).astype(float),
    })[~pushes]

    b_all, se_all, n_all = _fit_logistic(d)

    # Honest split: fit on the earlier season, grade on the later one.
    seasons = sorted(d["season"].unique())
    oos = None
    if len(seasons) >= 2:
        tr = d[d["season"] == seasons[0]]
        te = d[d["season"] == seasons[-1]]
        b_tr, se_tr, n_tr = _fit_logistic(tr)
        b_te, se_te, n_te = _fit_logistic(te)
        coin = float(-np.log(0.5))
        oos = {
            "fit_season": int(seasons[0]), "fit_coef": round(b_tr, 5),
            "fit_se": round(se_tr, 5), "fit_n": n_tr,
            "check_season": int(seasons[-1]), "check_n": n_te,
            "check_season_own_coef": round(b_te, 5),
            "sign_flips_between_seasons": bool(b_tr * b_te < 0),
            "log_loss_on_check": round(_log_loss(b_tr, te), 5),
            "log_loss_coin_flip": round(coin, 5),
            "beats_coin_flip": bool(_log_loss(b_tr, te) < coin),
        }

    ci_lo, ci_hi = b_all - 1.96 * se_all, b_all + 1.96 * se_all
    supported = bool(ci_lo > 0 and (oos is None or oos["beats_coin_flip"]))

    # Realized rates by bucket: the primary evidence, reported whether
    # or not a curve is fitted to it.
    buckets = []
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 99)]:
        m = (d["edge"] >= lo) & (d["edge"] < hi)
        if m.sum() >= 15:
            buckets.append({
                "edge_range": [lo, hi if hi < 99 else None],
                "n": int(m.sum()),
                "actual_cover_pct": round(float(d.loc[m, "covered"].mean()), 4),
                "fitted_at_midpoint": round(float(1 / (1 + np.exp(-b_all * d.loc[m, "edge"].mean()))), 4),
            })
    realized = [bk["actual_cover_pct"] for bk in buckets]
    monotonic = all(x <= y for x, y in zip(realized, realized[1:]))

    return {
        "edge_coef": round(b_all, 5),
        "standard_error": round(se_all, 5),
        "ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        "n_games": n_all,
        "supported": supported,
        "in_sample": True,
        "realized_monotonic_in_edge": bool(monotonic),
        "out_of_sample": oos,
        "fit_check": buckets,
        "source": "2022-2023 full-ensemble backtest (coefficients fit 2016-2021)",
        "note": ("P(cover) = 1/(1+exp(-edge_coef*|edge|)). supported=false means the "
                 "relationship is NOT distinguishable from chance at this sample: do not "
                 "quote a cover probability or size stakes from it. Re-test as graded "
                 "seasons accumulate."),
    }


def main():
    dist = build_residual_distribution()
    calib = fit_edge_calibration()
    out = {"residual_distribution": dist, "edge_calibration": calib}
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "margin_dist.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    print(f"  residual dist over {dist['n_games']} games (2010-2024), std {dist['residual_std']}")
    print(f"  key margin mass: 3 -> {dist['key_margin_mass']['3']}, 7 -> {dist['key_margin_mass']['7']}")
    print(f"  edge calibration: coef {calib['edge_coef']} +/- {calib['standard_error']} "
          f"over {calib['n_games']} games -> SUPPORTED: {calib['supported']}")
    if not calib["supported"]:
        oos = calib.get("out_of_sample") or {}
        print(f"    95% CI {calib['ci95']} contains zero; realized monotonic in edge: "
              f"{calib['realized_monotonic_in_edge']}")
        if oos:
            print(f"    fit {oos['fit_season']} (b={oos['fit_coef']:+.5f}) graded on "
                  f"{oos['check_season']}: log-loss {oos['log_loss_on_check']} vs coin flip "
                  f"{oos['log_loss_coin_flip']} -> beats coin flip: {oos['beats_coin_flip']}")
        print("    => cover probability WITHHELD from the board; stakes stay flat.")
    for bkt in calib["fit_check"]:
        print(f"    edge {bkt['edge_range']}: actual {bkt['actual_cover_pct']*100:.1f}% vs fitted {bkt['fitted_at_midpoint']*100:.1f}% (n={bkt['n']})")
    big = [b for b in calib["fit_check"] if b["edge_range"][0] >= 4]
    realized_big = (sum(b["actual_cover_pct"] * b["n"] for b in big) / sum(b["n"] for b in big)) if big else float("nan")
    print(f"  A 4-pt edge: normal approximation says "
          f"{100*0.5*(1+__import__('math').erf((4/13.86)/np.sqrt(2))):.1f}%, the fitted curve says "
          f"{100/(1+np.exp(-calib['edge_coef']*4)):.1f}%, and the games with edge >= 4 actually "
          f"covered {100*realized_big:.1f}% (n={sum(b['n'] for b in big)}). Breakeven is 52.4%.")


if __name__ == "__main__":
    main()
