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



def _site_src():
    """Every source file of the site, concatenated. The 2026-09-23 revamp
    split the old App.jsx into views and removed staking.js; the guards
    below read the whole site so they cannot be dodged by moving code."""
    root = os.path.join(REPO, "frontend", "src")
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith((".js", ".jsx")):
                out.append(open(os.path.join(dp, f)).read())
    return "\n".join(out)

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


def test_the_site_computes_no_prices():
    """A missing calibration file must not make the site more confident --
    and since 2026-09-23 the site cannot be, because it prices nothing.

    coverProb used to fall back to a normal approximation: a 4-point edge
    read 51.3% -> 61.4% on a fetch failure, and sizeStake turned that into
    ~0.19 of bankroll at full Kelly. The dashboard revamp deleted the whole
    path (frontend/src/staking.js); every probability, tier and stake now
    arrives from the core in data/site/board_*.json. This guards that the
    site never grows a pricing function again.
    """
    assert not os.path.exists(os.path.join(REPO, "frontend", "src", "staking.js"))
    site = _site_src()
    for token in ("normCdf", "coverProb", "sizeStake", "fullKelly", "ATS_SIGMA",
                  "PLAY_GAP", "LEAN_GAP", "Math.exp(", "Math.erf", "edgeCoef"):
        assert token not in site, f"the site computes a price again: {token}"
    # The stake shown is the core's, and an unsized pick says so.
    assert "stake_fraction" in site and "Unsized" in site
def test_tier_thresholds_live_only_in_the_core():
    """One set of numbers, one owner. The board's Play/Lean cutoffs used to
    live in staking.js and in the legacy grader and had to be kept equal by
    this test. Tiers are now computed by core/tiers.py from
    data/tier_thresholds.json (ADR 0025) and shipped per game; the site
    reads `tier` and holds no cutoff of its own."""
    import json
    site = _site_src()
    assert "m.tier" in site or ".tier" in site
    art = json.load(open(os.path.join(REPO, "data", "tier_thresholds.json")))
    for league in ("nfl", "cfb", "mlb", "nhl", "nba"):
        b = art["leagues"][league]
        assert 0 < b["coin_flip"] < b["lean"] < b["play"] < 1
        # A cutoff typed into the site would show up as one of these values.
        for k in ("coin_flip", "lean", "play"):
            assert f"{b[k]}" not in site, f"{league} {k} cutoff is hard-coded in the site"
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

    # The board used to gate this curve with `supported !== false`. Since
    # the 2026-09-23 revamp the site quotes no cover curve at all: it reads
    # no curve file and no coefficient, so an unsupported curve cannot
    # reach a card however the artifact is set.
    site = _site_src()
    for f in ("margin_dist.json", "cfb_edge_calibration.json", "0.01828"):
        assert f not in site, f"the site reads a cover curve again: {f}"


def test_cfb_edge_calibration_artifact():
    """CFB's curve must be a committed, regenerable artifact."""
    import json
    # data/, not model/: the board fetches it, so it has to ship with
    # the site build like margin_dist.json does.
    path = os.path.join(REPO, "data", "cfb_edge_calibration.json")
    assert os.path.exists(path), "run model/cfb_edge_calibration.py"
    assert not os.path.exists(os.path.join(REPO, "model", "cfb_edge_calibration.json")), \
        "two copies will drift; data/ is the one the site reads"
    man = open(os.path.join(REPO, "deploy", "generate_manifest.py")).read()
    assert 'copy_single_file("cfb_edge_calibration.json")' in man, \
        "the artifact must be copied into the static build or the board 404s"
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



def test_holdout_vault_seals_until_selection():
    """The control that stops a test column from overriding a train
    argmin. Reading held-out metrics before select() must raise."""
    import sys
    sys.path.insert(0, REPO)
    from model.holdout_discipline import HoldoutLeak, HoldoutVault

    v = HoldoutVault(select_by="train_mae", minimize=True)
    v.record(2, train_mae=9.0, test_mae=10.5)
    v.record(100, train_mae=9.5, test_mae=10.1)
    try:
        v.held_out()
        raise AssertionError("held-out metrics readable before select()")
    except HoldoutLeak:
        pass
    # The training table is the only thing visible early.
    assert all("test_mae" not in m for m in v.training_table().values())
    assert v.select() == 2, "selection must follow the training argmin"
    assert v.held_out()["test_mae"] == 10.5
    # Selecting ON a held-out metric is refused outright.
    try:
        HoldoutVault(select_by="test_mae").record(1, test_mae=1.0)
        raise AssertionError("allowed selection on a held-out metric")
    except HoldoutLeak:
        pass


def test_calibrate_scripts_do_not_print_per_candidate_holdout():
    """The leak itself: a per-candidate test column printed beside the
    train argmin is how half_life=100 got chosen. It must not come back."""
    import glob
    import re
    offenders = []
    for path in glob.glob(os.path.join(REPO, "model", "calibrate_*.py")):
        for i, line in enumerate(open(path), 1):
            if not line.strip().startswith("print"):
                continue
            # A print carrying BOTH a train and a held-out metric is the
            # side-by-side comparison that does the damage.
            if re.search(r"train[_ ]", line) and re.search(r"test[_ ](acc|mae|MAE|brier|Brier)", line):
                offenders.append(f"{os.path.basename(path)}:{i}")
    assert not offenders, f"held-out column printed beside training error: {offenders}"


def test_half_life_claim_is_withdrawn_not_restated():
    """ratings.py claimed held-out validation for a value selected on the
    test set. The claim must stay withdrawn and the failing number must
    not be quoted as support."""
    src = open(os.path.join(REPO, "model", "ratings.py")).read()
    doc = src[src.index("def add_recency_weights"):]
    doc = doc[:doc.index('"""', doc.index('"""') + 3)]
    assert "WITHDRAWN" in doc.upper(), "the withdrawal must stay stated"
    assert "revalidate_half_life_2024_25" in doc, "must point at the grading run"
    # Narrating the history is fine and necessary. What must not happen is
    # the old figure appearing WITHOUT the reversal that retired it.
    if "59.13" in doc:
        assert "did not replicate" in doc.lower() or "reverse" in doc.lower(), \
            "the 2023 figure is quoted with no mention that it failed to replicate"
    assert "selection on the test set" in doc.lower() or "test set" in doc.lower(), \
        "the docstring must disclose HOW the value was chosen, not just that it lost"


def test_half_life_revalidation_recorded_the_failure():
    import json
    path = os.path.join(REPO, "model", "revalidate_half_life_2024_25_results.json")
    assert os.path.exists(path), "run model/revalidate_half_life_2024_25.py"
    r = json.load(open(path))
    q = r["questions"]
    # Pin the finding, so a future edit cannot quietly turn a failure into a pass.
    assert q["shipped_beats_prior_default"] is False
    assert q["monotone_in_half_life_accuracy"] is False
    assert r["_provenance"]["holdout_seasons"] == [2024, 2025]
    # And pin the refusal to re-tune on the holdout.
    assert "argmin_on_holdout_reported_not_adopted" in q
    import model.ratings as R
    import inspect
    shipped = inspect.signature(R.add_recency_weights).parameters["half_life_weeks"].default
    assert shipped == r["shipped_value"], \
        "the knob moved without the ledger moving with it"


def test_the_record_only_grades_rows_with_a_model_opinion():
    """MLB once rendered 36 'Moved away' CLV verdicts for rows that carried
    no model opinion at all. The record is now data/site/record.json, built
    by scripts/export_record.py from the paper ledger, and it counts only
    signals where the model preferred the side (p_model > p_market) -- a
    row with no opinion cannot be graded. The site renders that file and
    nothing else."""
    rec = open(os.path.join(REPO, "scripts", "export_record.py")).read()
    assert "s.p_model <= s.p_market" in rec, "the record must skip rows with no preferred side"
    site = _site_src()
    assert "record.json" in site
    for legacy in ("performance.json", "mlb_divergence", "cfb_divergence", "/data/divergence"):
        assert legacy not in site, f"the record reads a legacy source again: {legacy}"



def test_board_records_which_coefficients_made_it():
    """A published margin must carry the conditions that produced it.
    predict_game has returned coefficient_set since the NGS fix and its
    comment claimed it was 'published per row' -- the divergence writer
    dropped it, so no published board said whether it was pre- or
    post-fix. The week-6 slope checkpoint depends on telling them apart.
    """
    import sys
    sys.path.insert(0, REPO)
    import pandas as pd
    from model.prediction import build_week_predictions

    # Functional: the live condition all season has been NGS absent.
    ratings = pd.DataFrame(
        {"offense_voa": [0.1, -0.05], "defense_voa": [-0.02, 0.03],
         "total_rating": [0.12, -0.08]}, index=["KC", "DEN"])
    games = pd.DataFrame([{"home_team": "KC", "away_team": "DEN", "home_rest": 7,
                           "away_rest": 7, "wind": 0.0, "is_neutral_site": False}])
    pred = build_week_predictions(ratings, games, ngs_features=None, elo_ratings=None)["KC"]
    assert pred["coefficient_set"] == "v1_rating_only_no_ngs"
    assert pred["features"]["ngs"] is False
    # The rating must actually move the number under the fix; the bug
    # was that it moved it by hundredths of a point.
    assert abs(pred["spread"]) > 1.0, "rating is not reaching the published margin"

    # Structural: the writer must carry it through. The end-to-end write
    # needs live odds, so this pins the assembly rather than the file.
    src = open(os.path.join(REPO, "deploy", "odds_watch_job.py")).read()
    assert '"coefficient_set": pred.get("coefficient_set")' in src, \
        "divergence rows must stamp the coefficient vector"
    assert '"coefficient_sets": sorted(' in src, \
        "the snapshot needs a roll-up so a reader need not scan every row"


def test_prop_grading_failure_is_not_swallowed():
    """The first prop ledger is the largest claim surface in the system
    and the on-ramp rule is built on it. A grading failure used to print
    and let the job report success."""
    import sys
    sys.path.insert(0, REPO)
    from deploy.weekly_job import _published_prop_opinions

    src = open(os.path.join(REPO, "deploy", "weekly_job.py")).read()
    assert "prop grading skipped:" not in src, \
        "the swallow-and-continue path is back"
    assert "prop_failure" in src and "report_failure" in src, \
        "a prop grading failure must escalate, not print"
    # The success report must not fire over a known prop failure.
    i_fail = src.index("if prop_failure:\n            # The ratings pipeline")
    i_ok = src.index('report_success("weekly_ratings_job", summary=f"{season} week')
    assert i_fail < i_ok, "report_success must be gated behind the prop-failure check"

    # An unfinished week must NOT alarm: no props file means nothing owed.
    assert _published_prop_opinions(2026, 99) == 0
    # And a real published week must be counted, or the guard can never fire.
    assert _published_prop_opinions(2026, 2) > 0, \
        "week 2 published engine opinions; the alarm needs to see them"


def test_pass_yds_failure_has_a_named_mechanism():
    """pass_yds was withheld twice with no established cause. The
    diagnosis reproduces the engine's OWN published claim for all three
    markets from stratum composition alone -- that reconstruction is
    what makes the mechanism a finding rather than a story, so it is
    pinned."""
    import json
    path = os.path.join(REPO, "model", "pass_yds_stratum_drift_results.json")
    assert os.path.exists(path), "run model/pass_yds_stratum_drift.py"
    d = json.load(open(path))
    published = json.load(open(os.path.join(
        REPO, "model", "player_projection_results.json")))["held_out"]["yardage"]
    for mkt in ("pass_yds", "rush_yds", "rec_yds"):
        claimed = [l for l in published[mkt]["lines"] if l["mult"] == 1.0][0]["claimed"]
        rebuilt = d["markets"][mkt]["reconstructed_claim_at_1x"]
        assert abs(rebuilt - claimed) < 0.001, \
            f"{mkt}: composition no longer reproduces the published claim " \
            f"({rebuilt} vs {claimed}) -- the mechanism claim is stale"
    # The direction of the effect must match each market's gate status.
    assert d["markets"]["pass_yds"]["claim_inflation_vs_pooled"] > 0.03
    assert d["markets"]["rush_yds"]["claim_inflation_vs_pooled"] < 0
    assert abs(d["markets"]["rec_yds"]["claim_inflation_vs_pooled"]) < 0.01
    # And the rejected fix must stay rejected with its lookahead warning.
    fix = d["candidate_fix_rejected"]
    assert fix["pass_yds_mean_abs_err"]["causal_normalized"] > \
        fix["pass_yds_mean_abs_err"]["frozen"], "the rejected fix is recorded as an improvement"
    assert "lookahead" in json.dumps(fix).lower()


def test_pass_yds_stays_withheld():
    """The diagnosis explains the failure; it does not license shipping."""
    import json
    res = json.load(open(os.path.join(REPO, "model", "player_projection_results.json")))
    assert res["held_out"]["yardage"]["pass_yds"]["withheld"] is True
    assert "pass_yds" not in res["_provenance"]["calibrated_markets"]



def test_prop_ledger_reaches_the_site():
    """grade_props writes data/prop_grades/. The manifest published only
    'player_grades' -- a directory nothing has ever created -- so the
    ledger would have been graded, committed and never shown."""
    man = open(os.path.join(REPO, "deploy", "generate_manifest.py")).read()
    assert '"prop_grades": list_and_copy_snapshots("prop_grades")' in man, \
        "the manifest must publish the directory grade_props actually writes"
    app = _site_src()
    assert "/data/prop_grades/summary.json" in app, "the site must read the ledger"
    assert "EngineLedger" in app, "the ledger needs a surface, not just a file"


def test_empty_prop_report_is_not_written():
    """load_actuals returning rows is not the same as the week being
    over: mid-week it returns whichever games have finished. On
    2026-09-20 that was one final of sixteen and grading matched 0 of
    386 -- then wrote a file and returned its path, which every caller
    reads as success."""
    gp = open(os.path.join(REPO, "deploy", "grade_props.py")).read()
    assert "if not graded:" in gp and "return None" in gp, \
        "a zero-claim report must not be written or returned as a path"
    i_guard = gp.index("if not graded:")
    i_write = gp.index('out_path = os.path.join(out_dir, f"{season}-week-')
    assert i_guard < i_write, "the guard must precede the write"


def test_unfinished_week_does_not_false_alarm():
    """The prop alert must distinguish an unfinished week from a broken
    grader. 'Box scores exist' cannot be the test -- it is true every
    Sunday afternoon."""
    import sys
    sys.path.insert(0, REPO)
    from deploy.weekly_job import _published_prop_opinions

    wj = open(os.path.join(REPO, "deploy", "weekly_job.py")).read()
    assert "rate > 0.5" in wj, "the alert must gate on a match RATE"
    assert "with_matches=True" in wj
    n, matched = _published_prop_opinions(2026, 2, with_matches=True)
    assert n > 0, "week 2 published opinions; the check needs to see them"
    # Whatever the week's state, the helper must never raise and must
    # never report more matches than opinions.
    assert 0 <= matched <= n



def test_every_data_file_the_board_fetches_actually_ships():
    """The class of bug, not one instance of it.

    Three times a surface fetched a file nothing produced --
    mlb_performance.json, player_grades/, cfb_edge_calibration.json -- and
    each 404 was silent, because every hook catches and renders an empty
    state. Every /data/ path the site fetches must exist on disk and be
    copied by the build: top-level files by copy_single_file, and the
    site/ and prop_grades/ families by generate_manifest's family copies.
    A ${...} in a path is expanded over the five leagues.
    """
    import re
    app = _site_src()
    man = open(os.path.join(REPO, "deploy", "generate_manifest.py")).read()
    families = {"site": '"site"' in man and 'for fam in ("site"' in man,
                "prop_grades": '"prop_grades": list_and_copy_snapshots("prop_grades")' in man}
    missing = []
    paths = set(re.findall(r"['\"`]/data/([a-z0-9_/${}]+\.json)['\"`]", app))
    assert paths, "the site fetches nothing from /data -- the scan is broken"
    for path in sorted(paths):
        expanded = ([path.replace(m, lg) for lg in ("nfl", "cfb", "mlb", "nhl", "nba")
                     for m in re.findall(r"\$\{[^}]+\}", path)] if "${" in path else [path])
        for p_ in expanded:
            on_disk = os.path.exists(os.path.join(REPO, "data", p_))
            fam = p_.split("/")[0] if "/" in p_ else None
            copied = families.get(fam, False) if fam else f'copy_single_file("{p_}")' in man
            if not (on_disk and copied):
                missing.append(f"{p_} (on disk: {on_disk}, copied: {copied})")
    assert not missing, ("the site fetches these but nothing ships them, so they 404 "
                         "silently: " + "; ".join(missing))



def test_preseason_prior_is_regressed_not_raw():
    """The prior asserted that last season carries forward whole. It
    carries at 0.441, and below a year-over-year correlation of 0.5 a
    slope of 1.0 loses to a slope of zero -- the raw prior was worse
    than predicting the league mean."""
    import sys
    sys.path.insert(0, REPO)
    from model.preseason_prior import PRIOR_CARRYOVER, blend_rating, regress_prior

    for comp in ("offense_voa", "defense_voa", "total_rating"):
        slope = PRIOR_CARRYOVER[comp][0]
        assert 0.2 < slope < 0.7, f"{comp} carryover {slope} is implausible"
    # Defense carries over less than offense -- if that inverts, the fit
    # was rerun on something different and the comment no longer holds.
    assert PRIOR_CARRYOVER["defense_voa"][0] < PRIOR_CARRYOVER["offense_voa"][0]

    assert abs(regress_prior(0.10, "total_rating")) < 0.10, "prior must move toward the mean"
    assert regress_prior(0.10, "not_a_component") == 0.10, \
        "an unknown component must pass through, not be silently mis-regressed"

    # The live path regresses; the k-calibration path must not, because
    # k=2 was fitted against the raw prior.
    raw = blend_rating(0.10, 0.02, games_played=1)
    live = blend_rating(0.10, 0.02, games_played=1, carryover="offense_voa")
    assert live < raw, "the live blend must pull the prior toward the mean"
    assert abs(raw - (1 / 3 * 0.02 + 2 / 3 * 0.10)) < 1e-9, \
        "the default path changed; calibrate_credibility_k measures that path"

    cal = open(os.path.join(REPO, "model", "calibrate_credibility_k.py")).read()
    assert "carryover=" not in cal, \
        "the k calibration must keep measuring the raw prior it was fitted against"


def test_preseason_prior_regression_was_gated():
    """Fitted on 2017-2022, graded once on 2023-2025. Pin the result and
    H1a's rejection so neither drifts into folklore."""
    import json
    import sys
    sys.path.insert(0, REPO)
    from model.preseason_prior import PRIOR_CARRYOVER

    path = os.path.join(REPO, "model", "preseason_prior_regression_results.json")
    assert os.path.exists(path), "run model/preseason_prior_regression.py"
    d = json.load(open(path))

    assert d["_provenance"]["fit_seasons"] == [2017, 2022]
    assert d["_provenance"]["graded_seasons"] == [2023, 2025]
    g = d["holdout_grade"]
    assert g["regressed_per_component"]["mae"] < g["raw_prev_shipped"]["mae"]
    # The finding that makes this more than a tweak.
    assert g["raw_prev_shipped"]["mae"] > g["league_mean_no_prior"]["mae"], \
        "the raw prior being worse than no prior is the whole argument"
    assert d["paired_mae_gain"]["significant_at_95"] is True

    # H1a stays rejected: not significant, and negative on both targets.
    h = d["h1a_continuity_priors"]
    assert h["verdict"] == "REJECTED"
    for k in ("interaction_on_point_differential", "interaction_on_voa_rating"):
        assert abs(h[k]["t"]) < 2, f"{k} would now be significant -- re-open H1a deliberately"
        assert h[k]["estimate"] < 0, "the interaction was negative; the hypothesis wanted positive"

    # Shipped constants must match what was graded.
    for comp, rec in d["carryover"].items():
        assert abs(PRIOR_CARRYOVER[comp][0] - rec["slope"]) < 5e-4, \
            f"{comp}: shipped coefficient drifted from the graded artifact"



def test_totals_are_withheld_until_shown():
    """A market was on the board from the start with no accuracy
    measurement behind it. Measured, it loses to the market."""
    import json
    v = json.load(open(os.path.join(REPO, "data", "totals_validation.json")))
    assert v["supported"] is False
    assert v["accuracy"]["model"]["mae"] > v["accuracy"]["market"]["mae"]
    # The compression signature: a near-constant prediction cannot carry edge.
    assert v["accuracy"]["model"]["prediction_sd"] < v["accuracy"]["market"]["prediction_sd"]
    assert not any(b["clears_breakeven"] for b in v["by_threshold"])

    # Since 2026-09-23 the core withholds the market itself: NFL's model
    # does not offer totals, so the board export never prices one, and the
    # gates export states the verdict from this file. The site holds no
    # switch that could turn totals back on.
    from coverline.leagues.nfl.model import NFLModel
    assert "total" not in NFLModel.__dict__["primary_markets"].fget(NFLModel.__new__(NFLModel))
    rec = open(os.path.join(REPO, "scripts", "export_record.py")).read()
    assert '"data/totals_validation.json"' in rec, "the gates must state the totals verdict"
    assert "totalsSupported" not in _site_src(), "no site-side switch for totals"
    man = open(os.path.join(REPO, "deploy", "generate_manifest.py")).read()
    assert 'copy_single_file("totals_validation.json")' in man, \
        "an artifact the board fetches must ship, or it 404s silently"


def test_withholding_totals_does_not_rewrite_history():
    """generate_performance rebuilds the record from snapshots on every
    run, so a flag-only gate would DELETE the four totals already
    published and make the record look better than it was."""
    src = open(os.path.join(REPO, "deploy", "generate_performance.py")).read()
    assert "totals_withheld_from" in src, "the grader must gate totals"
    assert "computed_at" in src, "the gate must be by DATE, not by flag alone"
    assert "effective_from" in open(
        os.path.join(REPO, "data", "totals_validation.json")).read()
    # The already-graded totals must survive a regeneration.
    import json
    perf = json.load(open(os.path.join(REPO, "data", "performance.json")))
    totals = [p for p in perf.get("plays", []) if p.get("market") == "total"]
    assert totals, "the published totals record was erased; that is rewriting history"



def test_spread_thresholds_are_measured_and_disclosed():
    """PLAY_GAP and LEAN_GAP were bare constants with no backtest behind
    them. After withholding totals for being unmeasured, the spread side
    had to be measured too -- and the result is disclosed, not hidden."""
    import json
    v = json.load(open(os.path.join(REPO, "data", "spread_validation.json")))
    assert v["supported"] is False
    assert v["action"] == "DISCLOSED, NOT WITHHELD", \
        "changing this to a withhold is a product decision, not a test fix"
    play = [b for b in v["by_threshold"] if b["tier"] == "play"][0]
    assert play["min_abs_gap"] == 4.0, "the measured threshold must match the shipped one"
    assert not play["clears_breakeven"]
    # The interval containing breakeven is the reason this is a disclosure.
    assert play["ci95"][0] < 0.524 < play["ci95"][1]
    # The old Play threshold is gone from the site; its successor, the tier
    # bands, carries the same disclosure on every board (ADR 0025).
    assert "No band has been distinguishable from break-even" in _site_src(), \
        "the board must say tiers have not been shown to beat break-even"
    man = open(os.path.join(REPO, "deploy", "generate_manifest.py")).read()
    assert 'copy_single_file("spread_validation.json")' in man


def test_elo_gap_is_recorded():
    """The NGS fix restored the rating and dropped Elo. That is a real
    hole in the live model and must not be lost."""
    import json
    import sys
    sys.path.insert(0, REPO)
    from model.prediction import (MARGIN_COEFFICIENTS,
                                  MARGIN_COEFFICIENTS_V1_RATING_ONLY as V1)
    assert "elo_diff" in MARGIN_COEFFICIENTS
    assert "elo_diff" not in V1, \
        "V1 gained an elo term -- if that was deliberate, re-gate and update this"
    e = json.load(open(os.path.join(REPO, "data", "spread_validation.json")))["elo_gap"]
    assert e["elo_t_on_train"] > 5
    assert e["holdout_mae_with_elo"] < e["holdout_mae_shipped"]
    # And the dissociation, which is the part most likely to be forgotten.
    assert e["holdout_mae_market"] < e["holdout_mae_with_elo"], \
        "the market is still the better predictor; that is why edge does not follow"



def test_regime_cap_number_matches_its_artifact():
    """The board quoted 0/9 for backed-regime early flags -- on a gate
    card, in a live chip, in the odds job and in the README -- and the
    artifact says 0/6 at Lean, 0/3 at Play. Play is a SUBSET of Lean,
    so 6 + 3 = 9 double-counts three games. The number underwrites a
    staking rule, so it is pinned to the evidence that produces it."""
    import json
    import re
    path = os.path.join(REPO, "model", "coach_regime_results.json")
    assert os.path.exists(path), "run model/coach_regime_experiment.py"
    grades = json.load(open(path))["grades"]
    backed = {g["min_edge"]: (g["wins"], g["n_graded"]) for g in grades
              if g["label"] == "model BACKED the regime team"}
    wins, n = backed[2.5]
    quoted = f"{wins}/{n}"

    app = _site_src()
    job = open(os.path.join(REPO, "deploy", "odds_watch_job.py")).read()
    readme = open(os.path.join(REPO, "README.md")).read()

    # Since 2026-09-23 the cap is applied by the core's board export and its
    # text is GENERATED from this artifact, so the site cannot drift from it.
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import export_board, export_record
    assert export_board.regime_evidence() == quoted
    rules = export_record.gates()["rules"]
    assert any(f"{quoted} ATS" in r["text"] for r in rules), "the gate text must quote the artifact"
    import inspect
    assert "went {q} in backtests" in inspect.getsource(export_board.apply_regime_cap), \
        "the capped card's reason must quote the same number"
    assert f"graded {quoted} in 2016-2023" in job, "the odds job's chip must match"
    assert f"went {quoted} ATS" in readme, "the README must match"

    # And the double-count must not come back anywhere that faces a user.
    bad = wins + backed[4.0][0], n + backed[4.0][1]
    stale = f"{bad[0]}/{bad[1]}"
    for name, src in (("App.jsx", app), ("odds_watch_job.py", job)):
        hits = [ln for ln in src.splitlines()
                if stale in ln and "until 2026-09-20" not in ln]
        assert not hits, f"{name}: the double-counted {stale} is quoted again"



def test_spread_validation_grades_what_ships():
    """Twice this figure graded a simpler model than the board runs --
    wrong coefficients, then no de-bias. Pin it to the shipped
    configuration so a third version cannot drift back."""
    import json
    v = json.load(open(os.path.join(REPO, "data", "spread_validation.json")))
    prov = v["_provenance"]
    assert "full_ensemble" in prov["coefficients"], "must grade the shipped coefficient path"
    assert "applied" in prov["debias"], "the board de-biases; the grade must too"
    assert prov["coefficient_split"], "the path split must be recorded"
    play = [b for b in v["shipped"]["by_threshold"] if b["tier"] == "play"][0]
    assert not play["clears_breakeven"] and play["ci95"][0] < 0.524 < play["ci95"][1]
    # The compression argument was withdrawn -- the shipped model is not near-constant.
    assert v["shipped"]["prediction_sd"] > 5.0, \
        "the shipped model's spread collapsed; the withdrawn compression claim may be back"
    assert "compression_claim_withdrawn" in v
    # The de-bias guard is inert, not a frozen-threshold bug.
    assert v["debias_sweep"]["verdict"] == "inert"
    # The legacy Play threshold no longer ships on the site (ADR 0025); what
    # must not happen is its figure being quoted as if it still did.
    assert f'Play tier {play["ats"]*100:.1f}%' not in _site_src(), \
        "the site quotes the retired legacy Play-tier figure"


def test_cfb_returning_production_finding_holds():
    """CFB's discount was carried as untested, then as untestable
    (CFBD unreachable), then tested via cfbfastR play-by-play -- and it
    came back the OPPOSITE of the NFL result. That reversal is the part
    most likely to be flattened into 'returning production is noise',
    so it is pinned."""
    import json
    d = json.load(open(os.path.join(REPO, "model", "cfb_carryover_check_results.json")))
    assert 0.3 < d["cfb_carryover"] < 0.6
    assert abs(d["cfb_carryover"] - d["nfl_carryover"]) < 0.05, \
        "the two leagues' carryover converged; if that changed, the write-up is stale"

    t = d["returning_production_test"]
    assert t["interaction"]["t"] > 2 and t["monotonic"], \
        "the CFB interaction was significant and monotonic; a change here reopens the question"
    r = t["held_out_rank_correlation"]
    assert r["returning_times_prev"] > r["flat_shrink"], \
        "in CFB the discount IMPROVES ordering -- the opposite of NFL, and the whole point"

    # Significant on the pooled fit, still not shipped: the gate is what decides.
    assert t["gate"]["t"] < 1.96 and "not shipped" in t["gate"]["verdict"]
    src = open(os.path.join(REPO, "model", "cfb_preseason_prior.py")).read()
    assert "returning_production_pct" in src, "the CFB prior was changed on a t=1.93 result"

    # The usage aggregates must ship, or the test cannot be re-run.
    import glob
    assert len(glob.glob(os.path.join(REPO, "model", "cfb_usage", "*.parquet"))) >= 3

# RESTORED 2026-09-20. Both of these were written, committed, and then
# silently dropped by a cherry-pick that resolved test_model_guards.py
# with -X theirs, taking a later whole-file version built on a base
# that lacked them. They pin REJECTIONS -- the frozen-threshold
# verdicts and H1b's failed gate -- which is exactly the kind of
# result that quietly becomes folklore once its guard disappears.

def test_frozen_threshold_sweep_is_closed():
    """A finished audit should not be re-run from scratch. Pin the
    verdicts and the rule they produced, so the next person inherits
    the conclusion rather than the search."""
    import json
    path = os.path.join(REPO, "model", "frozen_threshold_sweep_results.json")
    assert os.path.exists(path), "run model/frozen_threshold_sweep.py"
    d = json.load(open(path))
    v = d["verdicts"]
    assert v["yardage_stratum_cuts"].startswith("HIT")
    for clean in ("TIER_CUTS", "CONV_DEFAULT", "CRED_OPP"):
        assert v[clean] == "clean", f"{clean} changed verdict without a new entry"
    # The distinction that makes the rule useful.
    assert "PARTITION" in d["rule_of_thumb"].upper()
    # The sweep is closed: nothing may return to "unswept" without a
    # new entry saying why, and what WAS swept must stay recorded.
    assert d["still_unswept"] == [], "a constant went back to unswept without explanation"
    assert set(d["swept_since"]) == {"PLAY_GAP/LEAN_GAP", "in-season de-bias offsets"}, \
        "the last two sweeps must stay recorded with their findings"
    # And the one real hit must still be reproducible from its own artifact.
    assert os.path.exists(os.path.join(
        REPO, "model", "pass_yds_stratum_drift_results.json"))


def test_qb_continuity_stayed_out():
    """H1b passed on train with a clean placebo and still failed its
    gate. Pin the rejection: a hypothesis this well-supported on train
    is exactly the one that gets quietly promoted later."""
    import json
    import sys
    sys.path.insert(0, REPO)
    from model.preseason_prior import PRIOR_CARRYOVER

    path = os.path.join(REPO, "model", "qb_continuity_prior_results.json")
    assert os.path.exists(path), "run model/qb_continuity_prior.py"
    d = json.load(open(path))
    assert d["verdict"].startswith("REJECTED")

    # The train evidence was real -- that is why the record matters.
    assert d["train_interactions"]["offense_voa"]["t"] > 2
    # And the placebo held: offense moves, defense does not.
    assert abs(d["train_interactions"]["defense_voa"]["t"]) < 1
    # The gate is what kept it out.
    assert abs(d["paired_gain"]["t"]) < 2, \
        "the gate now passes -- re-open H1b deliberately, do not let this drift"
    assert (d["holdout_grade"]["regressed_qb_conditional"]["mae"]
            >= d["holdout_grade"]["regressed_unconditional_shipped"]["mae"])

    # And it must not have leaked into the shipped prior, which stays
    # unconditional: one offense coefficient, not one per QB state.
    assert set(PRIOR_CARRYOVER) == {"offense_voa", "defense_voa", "total_rating"}, \
        "the shipped prior gained a QB-conditional branch that never passed its gate"



if __name__ == "__main__":
    # RUN EVERYTHING, THEN REPORT (2026-09-20). This loop used to let
    # the first failure abort the process. On 2026-09-20 one stale file
    # -- a delete that did not travel with a file-by-file delivery --
    # failed the second test alphabetically and left 25 of 27 guards
    # silently unrun, including every one protecting the Tuesday prop
    # ledger. A suite that stops at the first failure hides exactly the
    # failures it exists to catch.
    import traceback

    passed, failed = [], []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            passed.append(name)
            print(f"  {name}: OK")
        except Exception as err:                          # noqa: BLE001
            failed.append((name, err))
            print(f"  {name}: FAIL -- {err}")
            if os.environ.get("GUARD_TRACEBACKS"):
                traceback.print_exc()

    print(f"\n{len(passed)} passed, {len(failed)} failed, "
          f"{len(passed) + len(failed)} total")
    if failed:
        print("\nfailures:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)
    print("all model guard tests passed")
