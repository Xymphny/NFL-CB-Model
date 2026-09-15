"""
MLB Layer 1: stable team OFFENSE x conditional pitching stack
(today's starter -> bullpen) x park factor -> expected runs per side
-> win probability. The structure the scoping settled on: in baseball
the stable object is the lineup; run prevention is versioned by the
day's starter.

Everything is walk-forward: every rating uses only games strictly
before the game being predicted, with credibility regression toward
league mean (k in plate-appearance-ish units of innings). Park
factors are multi-year walk-forward ratios. Win probability comes
from the runs-differential via a logistic fit ON TRAIN ONLY.

Evaluation contract (same ship rule as football): train seasons fit
the two free parameters (runs->winprob scale, home advantage); the
held-out season grades Brier/log-loss against (a) coin flip, (b)
home-team-always, (c) a plain team Elo -- the model must beat all
three held out or it doesn't ship. Market comparison happens in
model/mlb_backtest.py once the lines cache exists.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEAGUE_RPG = 4.4        # rough center; actual league mean computed per walk
CRED_OUTS_SP = 380      # ~2 months of starts before a starter's own numbers dominate
CRED_OUTS_PEN = 900     # pens: bigger pools, slower trust
CRED_GAMES_OFF = 45     # lineup offense stabilizes fast-ish
PARK_CRED = 120         # games of park evidence vs neutral


class WalkForwardState:
    """All ratings updated game by game in date order."""

    def __init__(self):
        self.off_runs = defaultdict(lambda: [0.0, 0])      # team -> [runs, games] scored (park-adjusted)
        self.sp_outs = defaultdict(lambda: [0.0, 0.0])     # pitcher -> [runs allowed, outs]
        self.pen_outs = defaultdict(lambda: [0.0, 0.0])    # team -> [relief runs allowed, relief outs]
        self.park = defaultdict(lambda: [0.0, 0])          # park -> [total runs, games]
        self.league_runs = [0.0, 0]                        # [runs per team-game numerator, team-games]

    # ---- reads (before updating with the current game) ----
    def league_rpg(self):
        r, g = self.league_runs
        return (r / g) if g > 40 else LEAGUE_RPG

    def park_factor(self, park):
        r, g = self.park[park]
        lg = self.league_rpg() * 2
        if g == 0:
            return 1.0
        raw = (r / g) / lg
        return (raw * g + 1.0 * PARK_CRED) / (g + PARK_CRED)

    def offense(self, team):
        r, g = self.off_runs[team]
        lg = self.league_rpg()
        if g == 0:
            return lg
        return (r + lg * CRED_GAMES_OFF) / (g + CRED_GAMES_OFF)

    def sp_ra27(self, pitcher):
        lg = self.league_rpg()
        r, o = self.sp_outs[pitcher]
        return (r * 27 + lg * CRED_OUTS_SP) / (o + CRED_OUTS_SP) if (o or True) else lg

    def pen_ra27(self, team):
        lg = self.league_rpg()
        r, o = self.pen_outs[team]
        return (r * 27 + lg * CRED_OUTS_PEN) / (o + CRED_OUTS_PEN)

    def expected_runs(self, off_team, opp_sp, opp_team, park, sp_share=0.58):
        """Offense vs the opposing pitching stack in this park.
        sp_share: starter's expected share of outs (league ~58% now)."""
        lg = self.league_rpg()
        off = self.offense(off_team) / lg
        stack = (sp_share * self.sp_ra27(opp_sp) + (1 - sp_share) * self.pen_ra27(opp_team)) / lg
        return lg * off * stack * self.park_factor(park)

    # ---- update with a completed game ----
    def update(self, g, pitching_by_game):
        pf = self.park_factor(g["park"])
        for side, opp in (("home", "away"), ("away", "home")):
            score = g[f"{side}_score"]
            self.off_runs[g[f"{side}_team"]][0] += score / max(pf, 0.5)
            self.off_runs[g[f"{side}_team"]][1] += 1
            self.league_runs[0] += score
            self.league_runs[1] += 1
        self.park[g["park"]][0] += g["home_score"] + g["away_score"]
        self.park[g["park"]][1] += 1
        for row in pitching_by_game.get(g["game_key"], []):
            if row["started"]:
                self.sp_outs[row["pitcher"]][0] += row["runs"]
                self.sp_outs[row["pitcher"]][1] += row["outs"]
            else:
                self.pen_outs[row["team"]][0] += row["runs"]
                self.pen_outs[row["team"]][1] += row["outs"]


def run_walk_forward(schedule, pitching):
    """Predicted expected-runs pair for every game, strictly pregame."""
    pit_by_game = defaultdict(list)
    for r in pitching.to_dict("records"):
        pit_by_game[r["game_key"]].append(r)
    state = WalkForwardState()
    rows = []
    for g in schedule.sort_values(["date", "game_key"]).to_dict("records"):
        if pd.isna(g["home_sp"]) or pd.isna(g["away_sp"]):
            state.update(g, pit_by_game)
            continue
        eh = state.expected_runs(g["home_team"], g["away_sp"], g["away_team"], g["park"])
        ea = state.expected_runs(g["away_team"], g["home_sp"], g["home_team"], g["park"])
        rows.append({**{k: g[k] for k in ("season", "game_key", "date", "home_team", "away_team",
                                          "home_score", "away_score", "home_sp", "away_sp", "park")},
                     "exp_home": eh, "exp_away": ea,
                     "home_won": int(g["home_score"] > g["away_score"])})
        state.update(g, pit_by_game)
    return pd.DataFrame(rows)


def fit_win_prob(train):
    """Logistic: P(home) = sigmoid(a * log(exp_home/exp_away) + b).
    b absorbs home advantage; a converts run ratio to probability."""
    x = np.log(train["exp_home"] / train["exp_away"]).values
    y = train["home_won"].values
    a, b = 1.0, 0.0
    for _ in range(400):                      # simple Newton steps
        p = 1 / (1 + np.exp(-(a * x + b)))
        ga, gb = np.sum((y - p) * x), np.sum(y - p)
        haa, hab, hbb = -np.sum(p * (1 - p) * x * x), -np.sum(p * (1 - p) * x), -np.sum(p * (1 - p))
        det = haa * hbb - hab * hab
        a -= (hbb * ga - hab * gb) / det
        b -= (haa * gb - hab * ga) / det
    return a, b


def win_prob(df, a, b):
    return 1 / (1 + np.exp(-(a * np.log(df["exp_home"] / df["exp_away"]) + b)))


def elo_baseline(schedule, k=4, home_adv=24):
    """Plain team Elo, walk-forward -- the bar Layer 1 must clear."""
    ratings = defaultdict(lambda: 1500.0)
    probs, outcomes, seasons = [], [], []
    for g in schedule.sort_values(["date", "game_key"]).to_dict("records"):
        eh = 1 / (1 + 10 ** (-(ratings[g["home_team"]] + home_adv - ratings[g["away_team"]]) / 400))
        probs.append(eh)
        won = int(g["home_score"] > g["away_score"])
        outcomes.append(won)
        seasons.append(g["season"])
        ratings[g["home_team"]] += k * (won - eh)
        ratings[g["away_team"]] -= k * (won - eh)
    return pd.DataFrame({"season": seasons, "p": probs, "home_won": outcomes})


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main(train_seasons, test_season):
    sched = pd.read_csv("model/mlb_schedule_cache.csv")
    pit = pd.read_csv("model/mlb_pitching_cache.csv")
    preds = run_walk_forward(sched, pit)
    train = preds[preds["season"].isin(train_seasons)]
    test = preds[preds["season"] == test_season]
    a, b = fit_win_prob(train)
    p_test = win_prob(test, a, b)
    print(f"fit on {train_seasons}: a={a:.3f}, b={b:.3f} "
          f"(implied home edge at even runs: {1/(1+np.exp(-b))*100:.1f}%)")
    print(f"\nHELD OUT {test_season} ({len(test)} games):")
    y = test["home_won"].values
    print(f"  Layer 1        : Brier {brier(p_test, y):.4f} | log-loss {log_loss(p_test, y):.4f} "
          f"| home-pick acc {np.mean((p_test > .5) == y)*100:.1f}%")
    print(f"  coin flip      : Brier {brier(np.full(len(y), .5), y):.4f} | log-loss {log_loss(np.full(len(y), .5), y):.4f}")
    hb = train['home_won'].mean()
    print(f"  home-rate prior: Brier {brier(np.full(len(y), hb), y):.4f} | log-loss {log_loss(np.full(len(y), hb), y):.4f}")
    elo = elo_baseline(sched)
    elo_t = elo[elo["season"] == test_season]
    print(f"  plain Elo      : Brier {brier(elo_t['p'], elo_t['home_won']):.4f} | log-loss {log_loss(elo_t['p'], elo_t['home_won']):.4f}")
    # Calibration by decile
    dec = pd.cut(p_test, np.arange(0.3, 0.75, 0.05))
    cal = pd.DataFrame({"p": p_test, "y": y}).groupby(dec, observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    print("\ncalibration (held out):")
    print((cal * [1, 1, 1]).round(3).to_string())
    return preds, (a, b)


if __name__ == "__main__":
    main([2021, 2022, 2023, 2024], 2025)
