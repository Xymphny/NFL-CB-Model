"""
Real, confirmed season-ending injuries from 2026 training camp/preseason
— gathered from CBS Sports and Sharp Football's injury trackers.

Deliberately limited to CONFIRMED season-ending cases (torn ACL, IR
placements, "out for season" designations), not the much larger set of
"questionable, could go either way before kickoff" injuries that
remain genuinely uncertain this far out — those aren't actionable with
real confidence.

Note on why this exists at all, given official Week 1 injury reports
don't exist yet (confirmed directly: NFL.com's Week 1 page shows "No
Injuries Reported" for every game — teams aren't required to file
until Week 1 itself): these season-ending injuries are already public
knowledge from camp, and by the time Vegas set the win totals gathered
in model/win_totals_2026.py (Aug 28), the market had almost certainly
already priced these in — while this model's own last-season-based
component obviously can't see them. Same design logic as the
coaching/QB change disruption weighting.
"""

SEASON_ENDING_INJURIES_2026 = {
    "PIT": {"player": "Calvin Austin III", "position": "WR", "injury": "torn ACL"},
    "CLE": {"player": "Alex Wright", "position": "EDGE", "injury": "torn Achilles"},
    "LAC": {"player": "Tyler Biadasz", "position": "OL (starting interior)", "injury": "ACL/knee damage"},
    "SEA": {"player": "Bud Clark", "position": "S (2nd-round rookie)", "injury": "broken ankle"},
    "SF": {"player": "Ricky Pearsall", "position": "WR", "injury": "PCL surgery"},
    "HOU": {"player": "Jayden Higgins", "position": "WR", "injury": "torn ACL"},
    "BAL": {"player": "Danny Pinter", "position": "C (starter)", "injury": "torn patellar tendon"},
    "ARI": {"player": "Trey Benson", "position": "RB", "injury": "knee, out for season"},
}
