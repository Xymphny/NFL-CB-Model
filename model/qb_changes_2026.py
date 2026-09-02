"""
Real, confirmed 2026 starting QB changes — teams whose Week 1 starter
is a clearly different, real player than whoever accumulated most of
last season's snaps. Only high-confidence, unambiguous cases are
included here (not every "projected" starter from preseason coverage,
many of which remain genuinely uncertain this far out).

Used the same way as coaching changes: reduce reliance on last
season's rating for these specific teams (it reflects a QB who won't
be playing), leaning more on the Vegas signal instead, since the
market has already priced in the real personnel change.
"""

QB_CHANGES_2026 = {
    "ARI": {"out": "Kyler Murray", "in": "Jacoby Brissett"},
    "MIN": {"out": None, "in": "Kyler Murray"},  # Murray joining as a clear upgrade over 2025's starter
    "MIA": {"out": "Tua Tagovailoa", "in": "Malik Willis"},
}
