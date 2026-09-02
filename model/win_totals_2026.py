"""
Real 2026 NFL season win totals — gathered from Squawka's win-totals
page, sourced to BetMGM, re-checked by them August 28, 2026.

This is exactly the market signal Section 11.1's design called for but
was never populated: the market's own aggregated view of team
strength, synthesizing beat-reporter access, scouting, and roster
analysis that this model's box-score/point-differential approach can't
replicate on its own.

Lines move before the season starts (9/9) — treat as a snapshot, not
a permanent record. Sanity check: the 32 lines sum to 273, one win of
"optimism" above the 272 actual regular-season games — consistent with
a normally-functioning, honestly-priced market per the source's own
methodology note.
"""

GATHERED_AT = "2026-08-28"  # per source's own re-check date
SOURCE = "Squawka (lines from BetMGM)"

WIN_TOTALS_2026 = {
    "BUF": 10.5, "NE": 9.5, "NYJ": 5.5, "MIA": 3.5,
    "BAL": 11.5, "CIN": 10.5, "PIT": 8.5, "CLE": 5.5,
    "HOU": 9.5, "JAX": 8.5, "IND": 7.5, "TEN": 6.5,
    "KC": 10.5, "DEN": 9.5, "LAC": 9.5, "LV": 5.5,
    "PHI": 10.5, "DAL": 9.5, "WAS": 7.5, "NYG": 7.5,
    "DET": 10.5, "CHI": 9.5, "GB": 9.5, "MIN": 8.5,
    "TB": 8.5, "ATL": 7.5, "CAR": 7.5, "NO": 7.5,
    "LA": 11.5, "SF": 10.5, "SEA": 10.5, "ARI": 3.5,
}
