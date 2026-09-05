"""
CFB current-week predictions and market divergence -- the real,
in-season CFB equivalent of NFL's odds_watch_job.py, using the
validated DVOA + Elo ensemble (71.84% straight-up accuracy on the
full weekly backtest -- the legitimate, tested use case, unlike CFB
preseason predictions, which are documented as unreliable in
model/cfb_preseason_prior.py).

Cannot run against real, live 2026 CFB odds this session. Tested
against real 2023 mid-season ratings with a constructed but realistic
example line -- validates the REAL computation and JSON structure,
while being clear that real, live CFB odds gathering is a separate,
not-yet-done task.
"""

import sys
import glob
import os
import json
import gc
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingest.cfb_pbp import load_cfb_season
from model.ratings import (
    add_situation_buckets, score_all_plays, compute_baselines,
    compute_raw_voa, opponent_adjust, filter_garbage_time, team_ratings,
)
from model.elo_rating import compute_elo_walk_forward
from model.cfb_prediction import predict_margin
from model.prediction import margin_to_win_probability
from model.market_comparison import american_to_implied_prob, devig_two_way, flag_divergence

SCHEDULE_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "cfb_schedule_cache.csv")

# Real, validated CFB ensemble MAE from the full weekly backtest
# (model/cfb_full_walk_forward.py) -- NFL's margin_to_win_probability
# defaults to margin_std=13.0, calibrated for NFL's own, notably lower
# MAE (~10-11 points). Using that NFL-calibrated value for CFB without
# checking would have been a real, uncaught unit mismatch -- CFB's
# real residual spread is wider, reflecting the sport's much larger
# talent gaps between teams.
CFB_MARGIN_STD = 12.76


def build_cfb_week_predictions(season, current_week, home_teams_and_away):
    full_season_df, raw = load_cfb_season(season)
    del raw
    gc.collect()

    df = full_season_df[full_season_df["week"] < current_week].copy()
    df = add_situation_buckets(df)
    df = score_all_plays(df, use_turnover_luck_adjustment=True, league="CFB")
    df = filter_garbage_time(df)
    baselines = compute_baselines(df)
    df = compute_raw_voa(df, baselines)
    df = opponent_adjust(df, iterations=3, regression=0.5)
    ratings = team_ratings(df, use_recency_weights=False)

    schedule_cache = pd.read_csv(SCHEDULE_CACHE)
    _, elo_ratings = compute_elo_walk_forward(schedule_cache)

    predictions = {}
    for home, away in home_teams_and_away:
        if home not in ratings.index or away not in ratings.index:
            continue
        rating_diff = ratings.loc[home, "total_rating"] - ratings.loc[away, "total_rating"]
        elo_diff = elo_ratings.get(home, 1500) - elo_ratings.get(away, 1500)
        margin = predict_margin(rating_diff=rating_diff, elo_diff=elo_diff)
        predictions[home] = {"spread": margin, "away_team": away}

    return predictions


def compute_cfb_divergences(predictions, market_lines):
    """
    market_lines: list of (home_team, away_team, home_ml, away_ml, home_spread).

    HONEST LIMITATION: CFB has no total-points model yet (only margin/
    spread, unlike NFL's separate predict_total). Rather than fabricate
    a total prediction, this only computes and flags SPREAD and
    win-probability divergence -- total_gap/total_flagged are not
    included in the output at all, not silently set to a meaningless
    placeholder value.
    """
    divergences = []
    for home, away, home_ml, away_ml, home_spread in market_lines:
        if home not in predictions:
            continue
        pred = predictions[home]
        home_fair, away_fair = devig_two_way(
            american_to_implied_prob(home_ml), american_to_implied_prob(away_ml)
        )
        market_spread = -home_spread
        model_win_prob_home = margin_to_win_probability(pred["spread"], margin_std=CFB_MARGIN_STD)

        result = flag_divergence(
            model_spread=pred["spread"], model_total=0.0, model_win_prob_home=model_win_prob_home,
            market_spread=market_spread, market_total=0.0,
            market_odds_home=home_fair, market_odds_away=away_fair,
        )
        # Drop the total fields entirely -- they're meaningless
        # placeholders (0.0 vs 0.0) needed only to call the shared
        # flag_divergence function without crashing, not a real total
        # comparison. Never expose a fabricated "total_gap" downstream.
        result.pop("total_gap", None)
        result.pop("total_flagged", None)

        divergences.append({
            "home_team": home, "away_team": away,
            "model_spread": pred["spread"], "market_spread": market_spread,
            "model_win_prob_home": model_win_prob_home, "market_win_prob_home_fair": home_fair,
            **result,
        })
    return divergences


if __name__ == "__main__":
    print("Testing real CFB predictions against real 2023 mid-season ratings...\n")
    predictions = build_cfb_week_predictions(2023, 10, [("Michigan", "Purdue"), ("Georgia", "Missouri")])
    for home, pred in predictions.items():
        print(f"  {pred['away_team']} @ {home}: model spread {pred['spread']:+.1f}")

    print("\nTesting divergence computation with a constructed, realistic example line")
    print("(real, live CFB odds gathering is a separate, not-yet-done task)...\n")
    example_lines = [("Michigan", "Purdue", -1000, 650, -17.5)]
    divergences = compute_cfb_divergences(predictions, example_lines)
    for d in divergences:
        print(json.dumps(d, indent=2, default=str))


# ---------------------------------------------------------------------------
# Live CFB odds gathering -- the "separate, not-yet-done task" from this
# module's own docstring, now done. Uses the SAME Odds API key already
# in production for NFL (sport key americanfootball_ncaaf -- confirmed
# in The Odds API's public sport list), so no new credential is needed.
#
# Team-name mapping: The Odds API names CFB teams "School Mascot"
# ("Ohio State Buckeyes"); this repo's ratings use school-only
# ("Ohio State"). Matching is longest-prefix wins, which resolves the
# genuinely ambiguous cases correctly ("Miami (OH) RedHawks" matches
# "Miami (OH)" over "Miami"). Unmatched teams are REPORTED loudly, not
# silently dropped -- a low match rate means the mapping needs work,
# and pretending otherwise would quietly shrink the board.
#
# NOT run against the live API from the sandbox this was written in --
# verify on first real run: (1) match rate printed at the end should be
# well above 90% of games where both teams have ratings, (2) spot-check
# three spreads against a book for sign convention.
# ---------------------------------------------------------------------------

CFB_PLAY_NOTE = (
    "CFB board thresholds are backtest-derived (held-out 2023, 574 graded "
    "games, real CFBD closing lines): gaps of 5+ covered 54.6% and 7+ "
    "covered 55.2% vs the 52.4% breakeven, rising monotonically with gap "
    "size -- but ONLY from week 5 on. Weeks 1-4 graded BELOW breakeven "
    "(48.3% overall) because early-season ratings are data-starved while "
    "the market prices offseason information the model can't see; the "
    "board therefore caps early-season verdicts at Lean. One season of "
    "evidence -- promising, not proven. Bet flat and small."
)


def map_odds_names_to_ratings(odds_data, rating_teams):
    """Longest-prefix match from Odds API 'School Mascot' names to
    rating table school names. Returns (mapping, unmatched)."""
    sorted_teams = sorted(rating_teams, key=len, reverse=True)
    mapping, unmatched = {}, set()
    for game in odds_data:
        for name in (game.get("home_team"), game.get("away_team")):
            if not name or name in mapping:
                continue
            match = next((t for t in sorted_teams if name.startswith(t)), None)
            if match:
                mapping[name] = match
            else:
                unmatched.add(name)
    return mapping, unmatched


def parse_cfb_game_markets(game):
    """Multi-book consensus + best prices, same approach as the NFL
    odds watch. Returns None when no book has a full spread posted."""
    home_name, away_name = game.get("home_team"), game.get("away_team")
    h2h_prices, spread_rows = [], []
    for book in game.get("bookmakers", []):
        book_name = book.get("title") or book.get("key")
        markets = {m["key"]: m for m in book.get("markets", [])}
        h2h = markets.get("h2h")
        if h2h:
            outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
            if home_name in outcomes and away_name in outcomes:
                h2h_prices.append({"book": book_name, "home": outcomes[home_name], "away": outcomes[away_name]})
        spreads = markets.get("spreads")
        if spreads:
            for o in spreads["outcomes"]:
                if o.get("point") is None:
                    continue
                spread_rows.append({
                    "book": book_name,
                    "side": "home" if o["name"] == home_name else "away",
                    "point": -o["point"] if o["name"] == home_name else o["point"],
                    "price": o.get("price"),
                })
    if not h2h_prices or not spread_rows:
        return None

    fair_probs = []
    for hp in h2h_prices:
        hf, _ = devig_two_way(american_to_implied_prob(hp["home"]), american_to_implied_prob(hp["away"]))
        fair_probs.append(hf)
    home_fair = float(pd.Series(fair_probs).median())

    home_spreads = [r for r in spread_rows if r["side"] == "home"]
    if not home_spreads:
        return None
    market_spread = float(pd.Series([r["point"] for r in home_spreads]).median())

    def _best(rows, better):
        if not rows:
            return None
        best_point = better(r["point"] for r in rows)
        at_point = [r for r in rows if r["point"] == best_point and r["price"] is not None]
        if not at_point:
            return {"point": best_point, "price": None, "book": None}
        top = max(at_point, key=lambda r: r["price"])
        return {"point": top["point"], "price": top["price"], "book": top["book"]}

    return {
        "home_fair": home_fair,
        "market_spread": market_spread,
        "best_prices": {
            "home_spread": _best(home_spreads, min),
            "away_spread": _best([r for r in spread_rows if r["side"] == "away"], max),
            "home_ml": max(h2h_prices, key=lambda r: r["home"])["home"],
            "away_ml": max(h2h_prices, key=lambda r: r["away"])["away"],
            "n_books": len(game.get("bookmakers", [])),
        },
    }


def load_latest_cfb_ratings(data_dir):
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, "cfb_ratings", "*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def run_live_cfb_odds_watch(data_dir):
    from deploy.odds_watch_job import fetch_current_odds
    from deploy.git_utils import git_commit_and_push
    from model.market_comparison import flag_divergence as _flag

    snapshot = load_latest_cfb_ratings(data_dir)
    if snapshot is None:
        print("[cfb_odds_watch] no CFB ratings snapshot yet -- run cfb_weekly_job first")
        return None
    ratings = {t["team"]: t for t in snapshot["ratings"]}

    odds_data = fetch_current_odds(sport_key="americanfootball_ncaaf")

    # Same bug class the NFL watch hit live (135 games matched from a
    # multi-week feed): the API returns future weeks too. NCAAF has no
    # per-matchup prediction dict to pair-filter against, so filter by
    # kickoff time instead -- this week's slate is games starting
    # within the next 8 days.
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) + timedelta(days=8)).isoformat()
    before = len(odds_data)
    odds_data = [g for g in odds_data if (g.get("commence_time") or "9999") <= cutoff]
    print(f"[cfb_odds_watch] {len(odds_data)} of {before} games kick off within 8 days")

    mapping, unmatched = map_odds_names_to_ratings(odds_data, ratings.keys())

    # ---- Carryover scale alignment ------------------------------------
    # The preseason seed regresses ratings 50% toward zero, but the
    # margin equation was fit on full-scale in-season ratings -- so
    # seeded model spreads come out uniformly ~half the market's
    # magnitude, and EVERY flag lands on the underdog (caught live on
    # the first seeded board: 66 leans, all dogs). That directional
    # lean is scale mismatch, not signal. Fix: when the snapshot is
    # carryover, fit model spreads to the slate's market spreads
    # (slope + intercept) and use the aligned numbers, leaving only
    # cross-sectional disagreement -- the only information a regressed
    # prior legitimately carries. In-season snapshots are untouched:
    # the backtest validated those at native scale.
    is_carryover = "carryover" in (snapshot.get("source") or "")
    align = None
    if is_carryover:
        pairs = []
        for game in odds_data:
            h, a = mapping.get(game.get("home_team")), mapping.get(game.get("away_team"))
            if not h or not a:
                continue
            parsed = parse_cfb_game_markets(game)
            if parsed is None:
                continue
            rd = ratings[h]["total_rating"] - ratings[a]["total_rating"]
            eh, ea = ratings[h].get("elo_rating"), ratings[a].get("elo_rating")
            ed = (eh - ea) if eh is not None and ea is not None else None
            pairs.append((predict_margin(rd, elo_diff=ed), parsed["market_spread"]))
        if len(pairs) >= 8:
            xs = pd.Series([p[0] for p in pairs]); ys = pd.Series([p[1] for p in pairs])
            slope = ((xs - xs.mean()) * (ys - ys.mean())).sum() / max(((xs - xs.mean()) ** 2).sum(), 1e-9)
            intercept = ys.mean() - slope * xs.mean()
            if 0.1 < slope < 5:
                align = (slope, intercept)
                print(f"[cfb_odds_watch] carryover scale alignment: model' = {slope:.2f}*model + {intercept:+.2f} over {len(pairs)} games")
            else:
                print(f"[cfb_odds_watch] alignment skipped: degenerate slope {slope:.2f}")

    divergences, skipped_unrated = [], 0
    for game in odds_data:
        home = mapping.get(game.get("home_team"))
        away = mapping.get(game.get("away_team"))
        if not home or not away:
            skipped_unrated += 1
            continue
        parsed = parse_cfb_game_markets(game)
        if parsed is None:
            continue
        rating_diff = ratings[home]["total_rating"] - ratings[away]["total_rating"]
        elo_h, elo_a = ratings[home].get("elo_rating"), ratings[away].get("elo_rating")
        elo_diff = (elo_h - elo_a) if elo_h is not None and elo_a is not None else None
        model_spread = predict_margin(rating_diff, elo_diff=elo_diff)
        if align is not None:
            model_spread = align[0] * model_spread + align[1]
        model_wp = margin_to_win_probability(model_spread, margin_std=CFB_MARGIN_STD)

        result = _flag(
            model_spread=model_spread, model_total=0.0, model_win_prob_home=model_wp,
            market_spread=parsed["market_spread"], market_total=0.0,
            market_odds_home=parsed["home_fair"], market_odds_away=1 - parsed["home_fair"],
        )
        result.pop("total_gap", None)
        result.pop("total_flagged", None)
        divergences.append({
            "home_team": home, "away_team": away,
            "market_spread": parsed["market_spread"],
            "market_win_prob_home_fair": parsed["home_fair"],
            "best_prices": parsed["best_prices"],
            **result,
        })

    # CFB QB awareness comes ONLY from the manual override file for
    # now (no automated CFB starter source exists) -- entries there
    # light the same confidence pip the NFL cards use. The quantified
    # NFL QB-adjustment experiment argues against ever repricing for
    # this info; annotation is the job.
    try:
        from deploy.qb_status import load_overrides
        cfb_ov = load_overrides("cfb")
        for d in divergences:
            d["qb_alert"] = {
                "home": (cfb_ov.get(d["home_team"]) or {}).get("alert"),
                "away": (cfb_ov.get(d["away_team"]) or {}).get("alert"),
            }
    except Exception as ov_err:
        print(f"[cfb_odds_watch] qb overrides skipped: {ov_err}")

    # CFB injury context from ESPN -- the ONLY source for college
    # (nflverse is NFL-only). Sparse by nature: college reporting is
    # not mandated, so a team absent from the feed gets None ("no
    # report available"), never an empty "healthy" list.
    try:
        from deploy.espn_extras import fetch_cfb_injuries
        cfb_inj = fetch_cfb_injuries(list(ratings.keys()))
        for d in divergences:
            d["injuries"] = {
                "home": cfb_inj.get(d["home_team"]),
                "away": cfb_inj.get(d["away_team"]),
            }
            d["injury_source"] = "espn" if cfb_inj else None
        if cfb_inj:
            print(f"[cfb_odds_watch] ESPN injuries attached for {len(cfb_inj)} teams with disclosed reports")
    except Exception as inj_err:
        print(f"[cfb_odds_watch] injuries skipped: {inj_err}")

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "season": snapshot["season"], "week": snapshot.get("week"),
        "note": (CFB_PLAY_NOTE + " This board runs on scale-aligned preseason carryover ratings until in-season data publishes.") if is_carryover else CFB_PLAY_NOTE,
        "match_report": {
            "games_from_api": len(odds_data),
            "games_priced": len(divergences),
            "skipped_no_rating_match": skipped_unrated,
            "unmatched_names": sorted(unmatched)[:20],
        },
        "divergences": divergences,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(data_dir, "cfb_divergence")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{snapshot['season']}-week-{(snapshot.get('week') or 0):02d}-{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[cfb_odds_watch] wrote {out_path}: {len(divergences)} games priced, "
          f"{skipped_unrated} skipped (no rating match), {len(unmatched)} unmatched names")
    if os.environ.get("GIT_REPO_URL"):
        git_commit_and_push(out_path, commit_message=f"CFB odds snapshot week {snapshot.get('week')}")
    return out_path


def main():
    """Cron entrypoint for the live CFB odds watch (render.yaml).
    CFB plays Thursday-Saturday; the cron schedule handles day gating
    (unlike NFL, where game days needed schedule-aware detection)."""
    from deploy.notify import report_success, report_failure
    data_dir = os.environ.get("REPO_DATA_PATH", "./data")
    try:
        out = run_live_cfb_odds_watch(data_dir)
        report_success("cfb_odds_watch", summary=os.path.basename(out) if out else "skipped, no ratings yet")
    except Exception as e:
        report_failure("cfb_odds_watch", error=str(e))
        sys.exit(1)


if os.environ.get("RUN_LIVE_CFB_ODDS_WATCH", "").lower() == "true":
    main()
