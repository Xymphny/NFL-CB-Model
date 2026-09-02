"""
Real 2026 NFL head coaching changes — all 10 confirmed across multiple
sources (ESPN, Yahoo Sports, FOX Sports, NFL.com), cross-referenced
for consistency.

Used as a "how much should we trust last season's rating" signal — a
new head coach means real scheme/roster uncertainty last season's
rating can't see, so teams here should lean more heavily on the
Vegas/preseason signals (which already price in the coaching change)
and less on the last-season component.
"""

COACHING_CHANGES_2026 = {
    "ARI": {"new_coach": "Mike LaFleur", "former_coach": "Jonathan Gannon"},
    "ATL": {"new_coach": "Kevin Stefanski", "former_coach": "Raheem Morris"},
    "BAL": {"new_coach": "Jesse Minter", "former_coach": "John Harbaugh"},
    "BUF": {"new_coach": "Joe Brady", "former_coach": "Sean McDermott"},
    "CLE": {"new_coach": "Todd Monken", "former_coach": "Kevin Stefanski"},
    "LV": {"new_coach": "Klint Kubiak", "former_coach": "Pete Carroll"},
    "MIA": {"new_coach": "Jeff Hafley", "former_coach": "Mike McDaniel"},
    "NYG": {"new_coach": "John Harbaugh", "former_coach": "Brian Daboll"},
    "PIT": {"new_coach": "Mike McCarthy", "former_coach": "Mike Tomlin"},
    "TEN": {"new_coach": "Robert Saleh", "former_coach": "Brian Callahan"},
}
