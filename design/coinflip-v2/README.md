# Coinflip dashboard v2 — design reference

These are the mockup artboards for the dashboard v2 build. The build brief is
`docs/briefs/2026-09-26-dashboard-v2.md`; it says what to build and where each number comes from.
These files show what it should look like.

- Live canvas (Pedro's, private): https://claude.ai/artifact/QrqJFqKwe1fq5zqwkj3N96
- Each `*.dc.html` is one artboard from that canvas, and `canvas.json` is the layout. The files are Design
  Component markup: HTML with `{{holes}}` and a small `renderVals()` class. They are **reference, not
  source**. Don't import them into the frontend. Rebuild each view in `frontend/src/views/` against
  `data/site/*.json`.
- **Data in the mockups is mixed.**
  - **Real:** CFB board (week 4 ratings plus Thursday's lines); NFL Teams (week-2 ratings, teams.json);
    NFL playoff odds (season simulator run); MLB Pitchers & parks (walk-forward replayed to Sep 23);
    NHL Attack & defence (2025-26); NBA Teams (2025-26 season file); every NFL injury list.
  - **Sample:** the NFL, MLB, NBA and NHL board cards; the Record page; the matchup page's prices.
    Sample numbers are placeholders for layout only. The site must render whatever the export says.

| Artboard | View it maps to |
|---|---|
| Main.dc.html | NFL board |
| CFB.dc.html | CFB board |
| MLB.dc.html | MLB board |
| NBA.dc.html | NBA board |
| NHL.dc.html | NHL board |
| Matchup.dc.html | Game detail (any league; the NFL example) |
| NFLTeams.dc.html, NBATeams.dc.html | Teams tab |
| MLBTeams.dc.html | MLB "Pitchers & parks" tab |
| NHLTeams.dc.html | NHL "Attack & defence" tab |
| NBAPlayers.dc.html | NBA Players tab (already built) |
| Record.dc.html | Record |
| Header.dc.html | Shared header and evidence strip |
