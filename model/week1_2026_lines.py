"""
Real, current Week 1 2026 book lines — gathered directly from ESPN's
odds page (espn.com/nfl/odds) on the date noted below. These are live
market lines, not historical/closing lines — they will move before
kickoff, so treat this as a snapshot, not a permanent record.

Format: (away_team, home_team, away_moneyline, home_moneyline,
         home_spread, total)
home_spread: negative = home favored (matches this project's
"positive = home favored" convention when negated for comparison)
"""

GATHERED_AT = "2026-09-01"  # date these lines were pulled

WEEK1_2026_CURRENT_LINES = [
    ("NE", "SEA", 150, -180, -3.5, 44.5),
    ("SF", "LA", 164, -198, -3.5, 48.5),
    ("ATL", "PIT", 145, -175, -3.0, 41.5),
    ("BAL", "IND", -175, 145, 3.5, 48.5),   # fixed: IND (home) is the underdog here, +3.5 not -3.5
    ("BUF", "HOU", -118, -102, 1.5, 44.5),  # fixed: HOU (home) is the underdog here, +1.5 not -1.5
    ("CHI", "CAR", -162, 136, 2.5, 47.5),   # fixed: CAR (home) is the underdog here, +2.5 not -2.5
    ("CLE", "JAX", 310, -395, -7.5, 40.5),
    ("NO", "DET", 240, -298, -7.0, 49.5),
    ("NYJ", "TEN", 110, -130, -1.5, 38.5),
    ("TB", "CIN", 164, -198, -3.5, 50.5),
    ("ARI", "LAC", 410, -550, -10.5, 46.5),
    ("GB", "MIN", -102, -118, -1.5, 45.5),
    ("MIA", "LV", 164, -198, -3.5, 40.5),
    ("WAS", "PHI", 170, -205, -4.5, 45.5),
    ("DAL", "NYG", -148, 124, 2.5, 48.5),   # fixed: NYG (home) is the underdog here, +2.5 not -2.5
    ("DEN", "KC", 124, -148, -3.0, 42.5),
]
