"""
Real PRE WK3 2026 box-score team stats — gathered directly from ESPN's
box score pages for all 16 PRE WK3 games (the week closest to final
roster cuts, per the scoping decision to limit box-score gathering to
this week only).

Each entry: (away_team, home_team, away_yards, home_yards,
             away_giveaways, home_giveaways)

yards = passing yards + rushing yards (team totals from the box score)
giveaways = interceptions thrown + fumbles lost

One known data-quality note: the DET/IND game's home/away was
corrected here after the box score explicitly showed "Detroit Lions @
Indianapolis Colts" (DET away), contradicting the original schedule-
based preseason_2026_results.py entry (which has DET listed as home).
That earlier error doesn't affect the point-differential signal there
(symmetric either way), but is corrected here since the enrichment
below needs the right team attributed to the right stat line.
"""

PRESEASON_WK3_BOXSCORES = [
    # (away, home, away_yards, home_yards, away_giveaways, home_giveaways)
    ("PIT", "BUF", 232, 266, 1, 1),
    ("NE", "CLE", 276, 379, 2, 0),
    ("SF", "LV", 281, 251, 0, 1),
    ("LAC", "LA", 298, 394, 0, 2),
    ("WAS", "BAL", 177, 456, 1, 1),
    ("ATL", "MIA", 279, 236, 3, 1),
    ("HOU", "CAR", 173, 288, 4, 1),
    ("NYJ", "NYG", 401, 139, 2, 1),
    ("TB", "JAX", 87, 249, 1, 0),
    ("NO", "DAL", 326, 328, 0, 0),
    ("ARI", "GB", 492, 282, 2, 1),
    ("SEA", "KC", 266, 353, 1, 1),
    ("PHI", "CIN", 420, 210, 2, 1),
    ("MIN", "DEN", 269, 412, 1, 1),
    ("DET", "IND", 356, 229, 4, 1),  # corrected home/away, see docstring
    ("TEN", "CHI", 398, 259, 1, 0),
]
