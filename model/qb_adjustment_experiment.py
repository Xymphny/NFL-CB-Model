"""
The QB question, tested properly: does a QUANTIFIED starter-quality
adjustment improve the model, where the earlier blunt test (just
excluding backup games) showed nothing?

QB value: EPA per dropback, walk-forward (current season through the
prior week, credibility-blended with the prior season, k=150
dropbacks). Game adjustment: (actual starter value - team's modal
starter value) for each side, difference taken home-minus-away --
zero in games where both usual starters play, so this feature ONLY
moves backup games, exactly where the model is known to be blind.

Fit: adjustment coefficient on 2016-2021 stacked with the ensemble.
Grade: held-out 2022-2023 MAE + ATS, overall and backup-games-only.
Ship rule, as always: only if held out says so.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from model.test_full_ensemble import build_combined_dataset
from model.prediction import predict_margin

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet"
SEASONS = list(range(2015, 2024))   # 2015 gives 2016 a prior season
TRAIN, TEST = list(range(2016, 2022)), [2022, 2023]
CRED_K = 150


def build_qb_values():
    """{(season, week, qb_name): value} -- walk-forward EPA/dropback,
    blended with prior season, centered on league average."""
    per = {}
    season_agg = {}
    for s in SEASONS:
        pbp = pd.read_parquet(PBP_URL.format(s=s), columns=["season","week","passer_player_name","qb_dropback","epa"])
        pbp = pbp[(pbp["qb_dropback"]==1) & pbp["passer_player_name"].notna() & pbp["epa"].notna()]
        league = pbp["epa"].mean()
        g = pbp.groupby(["passer_player_name","week"]).agg(n=("epa","size"), s=("epa","sum")).reset_index()
        season_agg[s] = pbp.groupby("passer_player_name").agg(n=("epa","size"), s=("epa","sum"))
        weeks = sorted(pbp["week"].unique())
        cum = {}
        for wk in weeks:
            wrows = g[g["week"]==wk]
            for qb in set(g["passer_player_name"]):
                cn, cs = cum.get(qb, (0.0, 0.0))
                pn, ps = (0.0, 0.0)
                if s-1 in season_agg and qb in season_agg[s-1].index:
                    r = season_agg[s-1].loc[qb]; pn, ps = r["n"]*0.6, r["s"]*0.6  # prior season downweighted
                n = cn + pn; ssum = cs + ps
                val = (ssum + CRED_K*league) / (n + CRED_K) - league
                per[(s, wk, qb)] = val
            for _, r in wrows.iterrows():
                cn, cs = cum.get(r["passer_player_name"], (0.0, 0.0))
                cum[r["passer_player_name"]] = (cn + r["n"], cs + r["s"])
    return per


def norm(name):
    if not isinstance(name, str): return ""
    parts = name.replace(".", "").split()
    while parts and parts[-1].lower() in ("jr","sr","ii","iii","iv","v"): parts.pop()
    return " ".join(parts).lower()


def main():
    print("Building QB values (9 seasons of pbp)...")
    qbv = build_qb_values()
    # Index by normalized name for schedule join ("P.Mahomes" in pbp vs "Patrick Mahomes" in schedule).
    # pbp passer names are like "P.Mahomes" -- match on last name + first initial.
    def keyname(n):
        n = norm(n)
        if not n: return ""
        bits = n.split()
        return (bits[0][0] + "." + bits[-1]) if len(bits) > 1 else n
    qbv_k = {}
    for (s, w, qb), v in qbv.items():
        qbv_k[(s, w, keyname(qb) if " " in qb else qb.lower())] = v

    games = pd.read_csv(os.environ.get("GAMES_CSV", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))
    games = games[(games["game_type"]=="REG") & games["season"].isin(TRAIN+TEST)]
    qb_sched = pd.concat([
        games.rename(columns={"home_team":"team","home_qb_name":"qb"})[["season","week","team","qb"]],
        games.rename(columns={"away_team":"team","away_qb_name":"qb"})[["season","week","team","qb"]]]).dropna()
    modal = qb_sched.groupby(["season","team"])["qb"].agg(lambda x: x.value_counts().index[0]).to_dict()

    def qval(season, week, name):
        return qbv_k.get((season, week, keyname(name)), None)

    print("Building combined ensemble dataset...")
    comb = build_combined_dataset()
    comb = comb.merge(games[["season","week","home_team","away_team","spread_line","home_qb_name","away_qb_name"]].dropna(subset=["spread_line"]),
                      on=["season","week","home_team","away_team"], how="inner")
    comb["ensemble"] = comb.apply(lambda r: predict_margin(r["rating_diff"], False, r["rest_diff"],
        cpoe_diff=r["cpoe_diff"], separation_diff=r["separation_diff"], yac_oe_diff=r["yac_oe_diff"],
        ryoe_diff=r["ryoe_diff"], elo_diff=r["elo_diff"]), axis=1)

    def qb_delta(r, side):
        team = r[f"{side}_team"]; starter = r[f"{side}_qb_name"]
        usual = modal.get((r["season"], team))
        if usual is None or norm(starter) == norm(usual): return 0.0
        sv, uv = qval(r["season"], r["week"], starter), qval(r["season"], r["week"], usual)
        if sv is None or uv is None: return 0.0
        return sv - uv   # EPA/dropback difference, negative when backup is worse

    comb["qb_adj"] = comb.apply(lambda r: qb_delta(r, "home") - qb_delta(r, "away"), axis=1)
    comb = comb.dropna(subset=["ensemble","actual_margin","spread_line"])
    nz = (comb["qb_adj"] != 0)
    print(f"{len(comb)} games; QB adjustment nonzero in {nz.sum()} ({nz.mean()*100:.0f}%) -- backup-start games with measurable QBs")

    train, test = comb[comb["season"].isin(TRAIN)], comb[comb["season"].isin(TEST)]
    def stack(features, d_tr, d_te):
        X = np.column_stack([d_tr[f] for f in features] + [np.ones(len(d_tr))])
        coef, *_ = np.linalg.lstsq(X, d_tr["actual_margin"].values, rcond=None)
        Xt = np.column_stack([d_te[f] for f in features] + [np.ones(len(d_te))])
        return coef, Xt @ coef

    def grade(pred, d, label):
        edge = pred - d["spread_line"].values
        pushes = d["actual_margin"].values == d["spread_line"].values
        win = ((edge>0) == (d["actual_margin"].values > d["spread_line"].values))[~pushes]
        mae = np.mean(np.abs(pred - d["actual_margin"].values))
        out = f"  {label}: MAE {mae:.3f} | ATS {win.mean()*100:.2f}% ({win.sum()}/{len(win)})"
        m = ~pushes & (np.abs(edge) >= 4)
        w = ((edge>0) == (d["actual_margin"].values > d["spread_line"].values))[m]
        if len(w) >= 15: out += f" | >=4: {w.mean()*100:.1f}% (n={len(w)})"
        print(out)

    print("\nHeld out 2022-2023, all games:")
    c0, p0 = stack(["ensemble"], train, test); grade(p0, test, "baseline        ")
    c1, p1 = stack(["ensemble","qb_adj"], train, test); grade(p1, test, "with QB adjust  ")
    print(f"  qb_adj coefficient: {c1[1]:+.1f} points of margin per EPA/dropback of starter downgrade")

    bt = test[test["qb_adj"] != 0]
    if len(bt) >= 30:
        print(f"\nHeld out, BACKUP-START games only ({len(bt)}):")
        grade(p0[(test["qb_adj"]!=0).values], bt, "baseline        ")
        grade(p1[(test["qb_adj"]!=0).values], bt, "with QB adjust  ")


if __name__ == "__main__":
    main()
