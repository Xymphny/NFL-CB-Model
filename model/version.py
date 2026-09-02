"""
Methodology version tracking. The formula has changed substantially
across this project's development (preseason prior, disruption
weighting, QB persistence, special teams all added at different
points) -- without a version tag on the output data itself, comparing
Week 3's rating to Week 10's later would have no way to tell whether a
change reflects real team performance or a formula change in between.

Bump METHODOLOGY_VERSION whenever a change would affect what a rating
means, not for pure bug fixes that don't change the methodology itself
(e.g. the chunked-CSV-reading memory fix doesn't get a version bump;
adding special teams to the output does).
"""

METHODOLOGY_VERSION = "1.4.0"

CHANGELOG = {
    "1.0.0": "Initial Layer 1: DVOA-style opponent-adjusted rating, success thresholds, turnover-luck adjustment, recency weighting",
    "1.1.0": "Added points-prediction layer (spread/moneyline/totals), garbage-time filtering, home-field/rest",
    "1.2.0": "Added preseason prior (last-season rating, k=2 credibility weighting, backtested)",
    "1.3.0": "Added real 2026 preseason performance signal, Vegas win totals, coaching/QB-change and season-ending-injury disruption weighting",
    "1.3.1": "Added QB persistence (real per-play VAR) and bootstrap uncertainty to weekly output",
    "1.4.0": "Added special teams sub-model (field goals/punts/kickoffs) as a separate rating component; extended VAR computation to receivers (validated) -- explicitly NOT extended to pass rushers (tested, found unreliable -- see model/injuries_and_var.py)",
}
