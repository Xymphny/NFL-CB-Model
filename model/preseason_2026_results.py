"""
Real 2026 NFL preseason results — gathered directly from ESPN's
schedule pages (espn.com/nfl/schedule, seasontype=1, weeks 1-4
covering the Hall of Fame Game and all three preseason weeks).

Every game here is real and actually happened (Aug 6 - Aug 29, 2026).
This is NOT synthetic or projected data.

Format: (home_team, away_team, home_score, away_score, preseason_week)
preseason_week: 0 = Hall of Fame Game, 1-3 = PRE WK 1-3
"""

PRESEASON_2026_RESULTS = [
    # Hall of Fame Game (neutral site, Canton OH) — week 0
    ("CAR", "ARI", 33, 30, 0),

    # PRE WK 1 (Aug 13-15)
    ("CIN", "DET", 16, 14, 1),
    ("PIT", "GB", 28, 9, 1),
    ("NE", "IND", 13, 13, 1),
    ("LV", "ARI", 14, 27, 1),
    ("HOU", "LAC", 7, 27, 1),
    ("SF", "TEN", 13, 19, 1),
    ("ATL", "DEN", 7, 27, 1),
    ("NYJ", "TB", 16, 24, 1),
    ("WAS", "MIA", 20, 7, 1),
    ("BUF", "CAR", 29, 14, 1),
    ("CHI", "CLE", 34, 10, 1),
    ("NYG", "MIN", 10, 13, 1),
    ("KC", "LA", 12, 20, 1),
    ("NO", "JAX", 20, 24, 1),
    ("BAL", "PHI", 24, 7, 1),
    ("SEA", "DAL", 7, 17, 1),

    # PRE WK 2 (Aug 20-23)
    ("HOU", "LV", 20, 22, 2),
    ("LAC", "SF", 17, 41, 2),
    ("PIT", "NYJ", 0, 17, 2),
    ("JAX", "CAR", 17, 34, 2),
    ("DEN", "GB", 13, 33, 2),
    ("DET", "WAS", 17, 13, 2),
    ("CLE", "BUF", 7, 31, 2),
    ("IND", "ATL", 6, 34, 2),
    ("MIN", "BAL", 3, 13, 2),
    ("LA", "NO", 34, 0, 2),
    ("MIA", "NYG", 3, 26, 2),
    ("NE", "PHI", 24, 21, 2),
    ("TB", "KC", 16, 15, 2),
    ("ARI", "DAL", 13, 34, 2),
    ("TEN", "SEA", 19, 16, 2),

    # PRE WK 3 (Aug 27-29)
    ("BUF", "PIT", 28, 27, 3),
    ("CLE", "NE", 37, 13, 3),
    ("LV", "SF", 12, 18, 3),
    ("LA", "LAC", 20, 18, 3),
    ("BAL", "WAS", 41, 3, 3),
    ("MIA", "ATL", 12, 17, 3),
    ("CAR", "HOU", 16, 13, 3),
    ("NYG", "NYJ", 23, 6, 3),
    ("JAX", "TB", 19, 0, 3),
    ("DAL", "NO", 24, 27, 3),
    ("GB", "ARI", 42, 38, 3),
    ("KC", "SEA", 9, 9, 3),
    ("CIN", "PHI", 30, 13, 3),
    ("DEN", "MIN", 34, 6, 3),
    ("DET", "IND", 25, 16, 3),
    ("CHI", "TEN", 24, 15, 3),
]
