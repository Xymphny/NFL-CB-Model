"""
Cron wrapper for cfb_weekly_job: computes the current CFB week from
the calendar, runs the real ratings job, and pushes the snapshot.

Week math: CFB weeks advance on Sundays; CFB_WEEK1_SATURDAY (env,
default 2026-08-29) anchors week 1. Simple, documented, and easily
corrected by env var if the anchor is off by a week.

Expected behavior while 2026 play-by-play remains unpublished
(cfbfastR's release assets stop at 2025 as of 2026-09-04): the job
prints the situation and exits 0 -- a missing upstream file is a
known waiting state, not a failure worth paging over. The moment the
asset appears, the same schedule starts producing real in-season
ratings that supersede the preseason seed, and the board's week-5+
Play tier turns on by itself when the calendar gets there.
"""

import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def current_cfb_week(anchor_str):
    anchor = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    days = (date.today() - anchor).days
    return max(1, days // 7 + 1)


def main():
    season = int(os.environ.get("SEASON", 2026))
    week = current_cfb_week(os.environ.get("CFB_WEEK1_SATURDAY", "2026-08-29"))
    print(f"[cfb_weekly_runner] season {season}, computed current week {week}")

    # Grade completed CFB plays FIRST -- grading needs only snapshots
    # and finals, so it runs even while 2026 pbp remains unpublished
    # and the ratings step below is in its waiting state.
    try:
        from deploy.generate_cfb_performance import generate as generate_cfb_perf
        perf_path = generate_cfb_perf(os.environ.get("REPO_DATA_PATH", "./data"), season)
        if perf_path and os.environ.get("GIT_REPO_URL"):
            from deploy.git_utils import git_commit_and_push
            git_commit_and_push(perf_path, commit_message=f"CFB performance through week {week - 1}")
    except Exception as perf_err:
        print(f"[cfb_weekly_runner] CFB grading soft-fail: {perf_err}")

    try:
        from deploy.cfb_weekly_job import run_cfb_weekly_job
        run_cfb_weekly_job(season, week, os.environ.get("REPO_DATA_PATH", "./data"))
    except Exception as e:
        msg = str(e)
        if "404" in msg or "Not Found" in msg:
            print(f"[cfb_weekly_runner] {season} play-by-play not published upstream yet ({msg[:100]}) "
                  f"-- waiting state, not a failure; the preseason seed keeps the board running")
            sys.exit(0)
        raise
    out = os.path.join(os.environ.get("REPO_DATA_PATH", "./data"), "cfb_ratings", f"{season}-week-{week:02d}.json")
    if os.environ.get("GIT_REPO_URL") and os.path.exists(out):
        from deploy.git_utils import git_commit_and_push
        git_commit_and_push(out, commit_message=f"CFB ratings: {season} week {week}")


if __name__ == "__main__":
    main()
