"""
Real, confirmed 2026 non-QB personnel changes significant enough to
mean last season's rating doesn't reflect the current roster — same
confidence bar as model/qb_changes_2026.py (clear, sourced,
unambiguous moves, not every roster tweak or depth-chart battle).

Gathered via web search of real 2026 offseason trade/free-agency
coverage (ESPN, NFL.com), cross-referenced for consistency.
"""

PERSONNEL_CHANGES_2026 = {
    "KC": {"change": "Traded CB Trent McDuffie to LA Rams", "type": "significant_loss"},
    "LA": {"change": "Acquired CB Trent McDuffie from KC; DT Aaron Donald reportedly returning from retirement", "type": "significant_gain"},
    "MIA": {"change": "Traded S Minkah Fitzpatrick to NY Jets", "type": "significant_loss"},
    "NYJ": {"change": "Acquired S Minkah Fitzpatrick from MIA (3-year, $40M extension)", "type": "significant_gain"},
    "CHI": {"change": "Traded WR DJ Moore to Buffalo (for a 2026 2nd-round pick)", "type": "significant_loss"},
    "BUF": {"change": "Acquired WR DJ Moore from Chicago", "type": "significant_gain"},
    "SF": {"change": "Signed WR Mike Evans in free agency (12-year Buccaneer, 866 career receptions)", "type": "significant_gain"},
}
