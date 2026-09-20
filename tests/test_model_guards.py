"""Guard tests for the failure classes found in the 2026-09-20 audit.

The props gate had a test. The money layer did not, and not one of the
bugs that audit found would have been caught by the existing suite.
Each test here pins a specific failure that actually shipped, named in
its docstring, so a regression is a red test rather than two quiet
weeks of wrong numbers.

Network-free: everything here runs on constructed inputs.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ngs_absent_uses_rating_only_coefficients():
    """NGS missing must switch coefficient vectors, not zero the features.

    MARGIN_COEFFICIENTS carries rating_diff 0.1078 because the NGS
    block and Elo absorb that signal. Run it against zeroed NGS and the
    team rating is understated ~211x vs the rating-only fit's 22.7091 --
    the published margin collapses to rescaled Elo plus a constant.
    Every 2026 board did this while nflverse's per-season NGS files
    404'd, and week 1 of every season enters the path by construction.
    """
    from model.prediction import (predict_game, MARGIN_COEFFICIENTS,
                                  MARGIN_COEFFICIENTS_V1_RATING_ONLY)

    kw = dict(home_rating=0.12, away_rating=-0.12, home_offense=0.05,
              away_offense=0.01, is_neutral_site=False, rest_diff=0, elo_diff=40.0)
    with_ngs = predict_game(**kw, ngs_present=True)
    without = predict_game(**kw, ngs_present=False)

    assert with_ngs["coefficient_set"] == "full_ensemble"
    assert without["coefficient_set"] == "v1_rating_only_no_ngs"
    assert without["features"]["ngs"] is False and with_ngs["features"]["ngs"] is True

    # The rating must actually move the number when NGS is gone.
    rating_gap = 0.24
    effect_no_ngs = MARGIN_COEFFICIENTS_V1_RATING_ONLY["rating_diff"] * rating_gap
    effect_full = MARGIN_COEFFICIENTS["rating_diff"] * rating_gap
    assert effect_no_ngs > 20 * effect_full, "rating must not be near-zeroed without NGS"

    # A stronger home team must be favoured MORE than a weaker one on
    # the degraded path -- the property the collapse destroyed.
    strong = predict_game(**{**kw, "home_rating": 0.30}, ngs_present=False)
    assert strong["spread"] > without["spread"] + 1.0


def test_inseason_debias_survives_a_poisoned_quote():
    """One NaN residual must not erase the model's opinion on the slate.

    NaN is truthy, so an unguarded median propagated into every
    prediction and nulled 15 of 16 spread opinions on the live board
    (2026-09-20 13:38Z) while the job reported success.
    """
    import math
    from deploy.odds_watch_job import inseason_offsets

    probe = [{"spread_gap": i - 6.0, "total_gap": 1.0} for i in range(12)]
    clean = inseason_offsets(probe)
    poisoned = list(probe)
    poisoned[4] = {"spread_gap": float("nan"), "total_gap": float("nan")}
    got = inseason_offsets(poisoned)
    assert all(math.isfinite(v) for v in got), "offsets must be finite"
    assert abs(got[0] - clean[0]) < 2.0, "one bad quote must not swing the slate offset"

    thin = inseason_offsets([{"spread_gap": float("nan"), "total_gap": None}] * 12)
    assert thin == (0.0, 0.0)


def test_mlb_postponed_rows_never_enter_state():
    """Unplayed games must not reach the walk-forward OR the Elo baseline.

    A single April NaN-score row poisoned league_runs and 2,258 of
    2,351 predictions. elo_baseline separately scored NaN-score rows as
    phantom away wins, so the ship-gate bar was graded on a different
    row set than the model.
    """
    import numpy as np
    import pandas as pd
    from model.mlb_model import run_walk_forward, elo_baseline

    def g(key, date, hs, as_):
        return {"season": 2026, "game_key": key, "date": date, "home_team": "NYA",
                "away_team": "BOS", "home_score": hs, "away_score": as_,
                "home_sp": "SP A", "away_sp": "SP B", "park": "NYA"}

    games = [g(f"k{i}", f"2026-04-{i + 1:02d}", 5, 3) for i in range(12)]
    games.insert(5, g("postponed", "2026-04-03", np.nan, np.nan))
    sched = pd.DataFrame(games)
    pit = pd.DataFrame([{"game_key": "k0", "pitcher": "SP A", "team": "NYA",
                         "started": True, "runs": 2, "outs": 18}])

    preds = run_walk_forward(sched, pit)
    assert len(preds) and not preds["exp_home"].isna().any()
    assert "postponed" not in set(preds["game_key"])

    elo = elo_baseline(sched)
    assert len(elo) == len(sched) - 1, "postponed game must not be scored as a result"
    assert not elo["home_won"].isna().any()


def test_grader_thresholds_match_the_board():
    """The grader must book only what the board claimed.

    Totals were graded at PLAY_GAP/LEAN_GAP while the board required
    +1 on each, and both markets were graded where the board shows one
    verdict with spread priority -- so the record carried claims that
    were never made, including a fabricated week-1 ATL@PIT loss.
    """
    from deploy.generate_performance import PLAY_GAP, LEAN_GAP, grade_divergence

    results = {(1, "HOM", "AWY"): {"home_score": 20, "away_score": 17,
                                   "spread_line": -1.0, "total_line": 40.0}}
    base = {"home_team": "HOM", "away_team": "AWY",
            "market_spread": -1.0, "market_total": 40.0}

    # A total gap between LEAN_GAP and LEAN_GAP+1 is a PASS on the board.
    below = dict(base, spread_gap=0.0, total_gap=LEAN_GAP + 0.4)
    assert grade_divergence(below, 1, results) == []

    # Above LEAN_GAP+1 it is a lean.
    lean = dict(base, spread_gap=0.0, total_gap=LEAN_GAP + 1.2)
    got = grade_divergence(lean, 1, results)
    assert len(got) == 1 and got[0]["market"] == "total" and got[0]["tier"] == "lean"

    # Spread takes priority, and only ONE verdict is graded per game.
    both = dict(base, spread_gap=PLAY_GAP + 1.0, total_gap=PLAY_GAP + 2.0)
    got = grade_divergence(both, 1, results)
    assert len(got) == 1 and got[0]["market"] == "spread"

    # The regime cap the board displayed is the stake that gets graded.
    capped = dict(base, spread_gap=PLAY_GAP + 1.0, total_gap=0.0, tier_cap="lean")
    assert grade_divergence(capped, 1, results)[0]["tier"] == "lean"


def test_staking_refuses_to_guess_without_calibration():
    """A missing calibration file must not make the site more confident.

    coverProb used to fall back to a normal approximation: a 4-point
    edge read 51.3% -> 61.4% on a fetch failure, and because sizeStake
    consumes the same probability, full Kelly went ~0.014 -> ~0.19 of
    bankroll. Parsed from the source because there is no JS runtime in
    this suite.
    """
    src = open(os.path.join(REPO, "frontend", "src", "staking.js")).read()
    fn = src[src.index("export function coverProb"):]
    fn = fn[:fn.index("\n}")]
    assert "normCdf" not in fn, "coverProb must not fall back to the normal approximation"
    assert "return null" in fn

    size = src[src.index("export function sizeStake"):]
    size = size[:size.index("\n}\n")]
    assert "prob == null" in size and "uncalibrated" in size
    assert "capped" in size, "a regime-capped play must be capped in dollars too"


def test_grader_and_board_share_thresholds():
    """One set of numbers, two consumers -- they must agree."""
    from deploy.generate_performance import PLAY_GAP as PY_PLAY, LEAN_GAP as PY_LEAN

    src = open(os.path.join(REPO, "frontend", "src", "staking.js")).read()
    js = {}
    for name in ("PLAY_GAP", "LEAN_GAP"):
        marker = f"export const {name} = "
        val = src[src.index(marker) + len(marker):].split("\n")[0].strip().rstrip(";")
        js[name] = float(val)
    assert js["PLAY_GAP"] == PY_PLAY, f"{js['PLAY_GAP']} vs {PY_PLAY}"
    assert js["LEAN_GAP"] == PY_LEAN, f"{js['LEAN_GAP']} vs {PY_LEAN}"


def test_withheld_market_is_absent_from_the_artifact():
    """pass_yds must be unanswerable, not merely unanswered.

    The deployed npz used to ship the withheld market's shapes, so the
    withholding rested on a single set-membership test in live code.
    """
    import numpy as np
    from model.player_projection import CALIBRATED_MARKETS, MARKETS

    z = np.load(os.path.join(REPO, "model", "player_shape_ecdf.npz"), allow_pickle=False)
    for m in MARKETS:
        if m in CALIBRATED_MARKETS:
            assert m in z.files, f"{m} is calibrated but has no shapes"
        else:
            assert not any(k.startswith(m) for k in z.files), \
                f"{m} is withheld but its shapes ship in the artifact"


def test_engine_opinions_carry_a_version():
    """An ungraded claim with no version cannot be attributed later."""
    import pandas as pd
    from model.player_projection import LiveProjector, ENGINE_VERSION

    rows = []
    for wk in range(1, 7):
        rows.append({"player_id": "00-0001", "player_display_name": "Test Back",
                     "position": "RB", "team": "DET", "opponent_team": "CHI",
                     "season": 2025, "week": wk, "season_type": "REG",
                     "attempts": 0, "carries": 18, "targets": 3,
                     "passing_yards": 0, "rushing_yards": 85, "receiving_yards": 20,
                     "passing_tds": 0, "rushing_tds": 1, "receiving_tds": 0})
    lp = LiveProjector(2025, data=pd.DataFrame(rows))
    op = lp.prop_opinion("Test Back", "DET", "CHI", "player_anytime_td", None)
    assert op and op["engine"] == ENGINE_VERSION and "rz" in op


def test_credit_exhaustion_is_defended_not_just_logged():
    """Running out of Odds API credits must not silently delete claims.

    x-requests-remaining was read and only printed. With no odds, every
    game kicking off during the blackout has no pregame snapshot to
    freeze and leaves the record entirely -- so exhaustion erases
    claims rather than merely pausing the board. The optional spend
    (the per-game props bake) must yield to a reserve held for the
    core board, and the job must shout before it goes dark.
    """
    import deploy.odds_watch_job as ow

    assert ow.CREDIT_RESERVE > 0 and ow.CREDIT_ALERT > ow.CREDIT_RESERVE

    src = open(os.path.join(REPO, "deploy", "odds_watch_job.py")).read()
    assert "send_webhook_alert" in src.split("CREDIT_ALERT")[2], \
        "a low-credit balance must raise an alert, not just a log line"

    # With the budget below the reserve, the props bake declines to spend.
    calls = []
    orig_remaining = ow.CREDITS_REMAINING
    try:
        ow.CREDITS_REMAINING = ow.CREDIT_RESERVE - 1
        out = ow.fetch_week_props("KEY", 2026, 2, "/nonexistent-dir", [{"id": "x"}])
        assert out is None and not calls
    finally:
        ow.CREDITS_REMAINING = orig_remaining


def test_game_day_gate_precedes_any_spend():
    """No odds call on a day with no games -- the cheapest saving there is."""
    src = open(os.path.join(REPO, "deploy", "odds_watch_job.py")).read()
    assert "is_game_day" in src
    gate = src.index("if not force_run and not is_game_day(")
    fetch_def = src.index("def fetch_current_odds")
    assert gate > fetch_def, "the gate must be consulted in main(), not inside the fetch"
    # and it must come BEFORE the call that spends a credit
    spend = src.index("fetch_current_odds(", gate)
    assert spend > gate, "no odds call may precede the game-day gate"


def test_unsupported_edge_curve_is_withheld():
    """A cover curve that cannot beat a coin flip must not be quoted.

    coverProb feeds sizeStake, so a curve fitted on noise sizes real
    money. Measured 2026-09-20 the NFL curve has a 95% interval
    containing zero, flips sign between seasons, and loses to a coin
    flip when fitted on 2022 and graded on 2023. The artifact must say
    so, and the board must respect it.
    """
    import json
    cal = json.load(open(os.path.join(REPO, "data", "margin_dist.json")))["edge_calibration"]

    # The artifact carries its own uncertainty and verdict.
    for field in ("standard_error", "ci95", "supported", "out_of_sample", "n_games"):
        assert field in cal, f"edge_calibration is missing {field}"
    lo, hi = cal["ci95"]
    assert lo < cal["edge_coef"] < hi
    # supported must be derived, not asserted: false whenever the
    # interval spans zero or the out-of-sample check fails.
    oos = cal["out_of_sample"] or {}
    expect = lo > 0 and oos.get("beats_coin_flip", True)
    assert cal["supported"] == bool(expect)

    # The board must gate on `supported`, not merely on the field existing.
    src = open(os.path.join(REPO, "frontend", "src", "App.jsx")).read()
    assert "supported !== false" in src, "App.jsx must withhold an unsupported cover curve"
    # The historical value may appear in a comment documenting the fix;
    # what must not survive is it being USED as the coefficient.
    assert "edgeCoefOverride={0.01828}" not in src, \
        "CFB's cover coefficient must come from the artifact, not a JSX literal"
    assert "cfb_edge_calibration.json" in src, "the board must read the committed artifact"


def test_cfb_edge_calibration_artifact():
    """CFB's curve must be a committed, regenerable artifact."""
    import json
    path = os.path.join(REPO, "model", "cfb_edge_calibration.json")
    assert os.path.exists(path), "run model/cfb_edge_calibration.py"
    cal = json.load(open(path))
    for field in ("edge_coef", "standard_error", "ci95", "n_games",
                  "significant_at_95", "realized_monotonic_in_edge", "_provenance"):
        assert field in cal, f"missing {field}"
    lo, hi = cal["ci95"]
    assert cal["significant_at_95"] == bool(lo > 0)


def test_ngs_replay_withholds_degenerate_slopes():
    """A regression slope computed on predictions that barely vary is a
    ratio with a near-zero denominator, not an estimate. The replay
    must refuse to publish one -- the same withholding discipline the
    NFL cover curve and pass yards get."""
    import json
    path = os.path.join(REPO, "model", "ngs_coefficient_replay_results.json")
    assert os.path.exists(path), "run model/ngs_coefficient_replay.py"
    res = json.load(open(path))
    rows = res["results"]
    assert rows, "replay produced no configurations"
    for row in rows:
        if row["prediction_spread"] < 0.10:
            assert row["slope"] is None and row["slope_se"] is None, \
                f"{row['configuration']}: slope published on degenerate spread"
            assert row["slope_withheld_degenerate"] is True
        else:
            assert row["slope"] is not None


def test_ngs_replay_does_not_overclaim():
    """The replay confirmed the rating was switched off; it did NOT
    explain the 0.68 slope. The artifact must keep those separate, so
    a later reader cannot mistake the confirmed half for the whole."""
    import json
    res = json.load(open(os.path.join(REPO, "model", "ngs_coefficient_replay_results.json")))
    v = res["verdict"]
    assert v["compression_confirmed"] is True
    assert v["explains_the_068_slope"] is False, \
        "the slope was never explained -- do not let this flip without new evidence"
    contrib = res["rating_contribution_to_margin_spread"]
    # The whole point: the fix restores the rating by the coefficient ratio.
    assert contrib["ratio_fixed_over_shipped"] > 100, \
        "the fix must materially restore the rating's contribution"
    assert contrib["rating_share_of_published_variation"] < 0.05, \
        "under the shipped coefficients the rating was effectively absent"
    assert "elo_not_reconstructable" in res["_provenance"], \
        "the Elo limitation must stay disclosed"



if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: OK")
    print("all model guard tests passed")
