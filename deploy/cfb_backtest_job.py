"""
One-off Render job: the entire CFB historical-lines workflow in one
triggered run, results readable straight from the job log.

Sequence:
  1. ingest/cfb_lines.py       -> model/cfb_lines_cache.csv (CFBD, 2021-2023)
  2. automatic sign sanity print: the 10 largest home spreads by our
     positive-=-home-favored convention, so a human can eyeball that
     ranked-team-hosts-cupcake games show LARGE POSITIVE numbers.
     (This repo has shipped a flipped-sign bug before; never again
     without it being loud.)
  3. model/cfb_ats_backtest.py -> join rate + held-out ATS by edge bucket
  4. if RUN_ROSTER_PRIORS=true: ingest/cfb_roster_priors.py, then the
     backtest again with roster features graded.
  5. commits both caches to the repo (a cron's filesystem is wiped on
     exit -- without the push, the CFBD work would vanish and cost
     credits again next time).

Trigger manually from the Render dashboard (the cron schedule is set
to Jan 1 so it effectively never fires on its own). Requires
CFBD_API_KEY in this job's environment -- and only this job's; no
other service needs it.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def run_step(cmd, label):
    print(f"\n===== {label} =====", flush=True)
    result = subprocess.run([sys.executable] + cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[cfb_backtest_job] {label} FAILED (exit {result.returncode}) -- stopping here")
        sys.exit(result.returncode)


def sign_sanity_print():
    path = "model/cfb_lines_cache.csv"
    df = pd.read_csv(path)
    print("\n===== SIGN SANITY CHECK (eyeball this before believing anything below) =====")
    print("Convention: POSITIVE spread_line = HOME team favored.")
    print("The 10 biggest home favorites in the cache -- these should all be")
    print("power programs hosting overmatched opponents. If the big names are")
    print("on the AWAY side of these rows, the sign is flipped: STOP.")
    top = df.nlargest(10, "spread_line")[["season", "week", "home_team", "away_team", "spread_line"]]
    print(top.to_string(index=False))
    n_pos = (df["spread_line"] > 0).mean()
    print(f"\n{n_pos*100:.0f}% of games have a home favorite -- sane range is roughly 55-65%.")
    if not 0.45 <= n_pos <= 0.75:
        print("WARNING: home-favorite rate outside the sane band -- treat results below with suspicion.")


def commit_caches():
    from deploy.git_utils import git_commit_and_push
    if not os.environ.get("GIT_REPO_URL"):
        print("[cfb_backtest_job] GIT_REPO_URL not set; caches NOT persisted (local run?)")
        return
    for path, msg in [
        ("model/cfb_lines_cache.csv", "Add CFBD closing lines cache (2021-2023)"),
        ("model/cfb_roster_priors.csv", "Add CFBD roster priors cache"),
    ]:
        if os.path.exists(path):
            try:
                git_commit_and_push(path, commit_message=msg)
            except Exception as e:
                print(f"[cfb_backtest_job] could not persist {path}: {e}")


def main():
    if not os.environ.get("CFBD_API_KEY"):
        print("[cfb_backtest_job] CFBD_API_KEY not set on this job's environment -- add it and re-trigger")
        sys.exit(1)

    run_step(["ingest/cfb_lines.py"], "STEP 1: fetch CFBD closing lines 2021-2023")
    sign_sanity_print()
    run_step(["model/cfb_ats_backtest.py"], "STEP 2: CFB ATS backtest (rating-only baseline)")

    if os.environ.get("RUN_ROSTER_PRIORS", "").lower() == "true":
        run_step(["ingest/cfb_roster_priors.py"], "STEP 3: fetch roster priors")
        run_step(["model/cfb_ats_backtest.py"], "STEP 4: backtest with roster features")
    else:
        print("\n(RUN_ROSTER_PRIORS not set -- skipping roster-feature pass; set it to true and re-trigger to test)")

    commit_caches()
    print("\n[cfb_backtest_job] done -- everything above this line is the deliverable; paste it back for interpretation")


if __name__ == "__main__":
    main()
