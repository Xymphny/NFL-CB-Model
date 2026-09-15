"""
The market test: does Layer 1's win probability beat the de-vigged
closing moneyline? Runs only when model/mlb_lines_cache.csv exists
(built by ingest/mlb_odds_archive.py on an open machine).

Grading: for every game with both closes, de-vig the two moneylines
to the market's implied home probability; the model 'flags' a side
when model_p - market_p exceeds an EV threshold; flags are graded at
the ACTUAL closing price (win: +100/|ml| or ml/100; loss: -1 unit).
Buckets by edge size, exactly the CFB evidence-table pattern -- the
output of this script IS the future board's threshold table.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mlb_model import run_walk_forward, fit_win_prob, win_prob


def american_to_prob(ml):
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def devig_home_prob(home_ml, away_ml):
    ph, pa = american_to_prob(home_ml), american_to_prob(away_ml)
    return ph / (ph + pa)


def payout(ml):
    ml = float(ml)
    return ml / 100 if ml > 0 else 100 / -ml


def join_lines(preds, lines):
    """Doubleheader-aware: retrosplits game_number is 0 for single
    games and 1/2 for twin bills; archive numbering is 0-based file
    order. Join on date+teams+ordinal."""
    p = preds.copy()
    p["ord"] = p.groupby(["date", "home_team", "away_team"]).cumcount()
    l = lines.copy()
    l = l.rename(columns={"game_number_in_day": "ord"})
    return p.merge(l[["date", "home_team", "away_team", "ord",
                      "home_ml_close", "away_ml_close", "home_ml_open", "away_ml_open"]],
                   on=["date", "home_team", "away_team", "ord"], how="inner")


def main(train_seasons, test_seasons, lines_path="model/mlb_lines_cache.csv"):
    if not os.path.exists(lines_path):
        print(f"[mlb_backtest] {lines_path} not present -- run ingest/mlb_odds_archive.py on an open machine first")
        return
    sched = pd.read_csv("model/mlb_schedule_cache.csv")
    pit = pd.read_csv("model/mlb_pitching_cache.csv")
    lines = pd.read_csv(lines_path)
    preds = run_walk_forward(sched, pit)
    a, b = fit_win_prob(preds[preds["season"].isin(train_seasons)])
    test = join_lines(preds[preds["season"].isin(test_seasons)], lines)
    test = test.dropna(subset=["home_ml_close", "away_ml_close"])
    test["model_p"] = win_prob(test, a, b)
    test["market_p"] = [devig_home_prob(h, aml) for h, aml in zip(test["home_ml_close"], test["away_ml_close"])]
    print(f"joined {len(test)} held-out games with closes "
          f"(join rate {len(test)/max(1,(preds['season'].isin(test_seasons)).sum())*100:.1f}%)")
    y = test["home_won"].values
    from model.mlb_model import brier, log_loss
    print(f"  model  Brier {brier(test['model_p'], y):.4f} | market Brier {brier(test['market_p'], y):.4f}")
    print("\nEV-threshold buckets (flag side where model_p - market_p > t):")
    for t in (0.02, 0.03, 0.04, 0.05, 0.07):
        units, n, wins = 0.0, 0, 0
        for _, r in test.iterrows():
            edge_home = r["model_p"] - r["market_p"]
            side = "home" if edge_home > t else ("away" if -edge_home > t else None)
            if side is None:
                continue
            n += 1
            won = (r["home_won"] == 1) if side == "home" else (r["home_won"] == 0)
            ml = r[f"{side}_ml_close"]
            units += payout(ml) if won else -1.0
            wins += won
        roi = units / n * 100 if n else 0.0
        print(f"  t={t:.2f}: {n:4d} flags | {wins}/{n} | {units:+7.2f}u | ROI {roi:+.2f}%")


if __name__ == "__main__":
    main([2015, 2016, 2017, 2018], [2019, 2020, 2021])
