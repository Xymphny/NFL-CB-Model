---
name: CoinFlip
description: A network operations wall for betting markets; every number on the screen was earned by evidence.
colors:
  ground: "#0b0f14"
  panel: "#0f151c"
  panel-2: "#141c26"
  panel-3: "#19232f"
  rule: "#223040"
  rule-soft: "#18222d"
  ink: "#e6e2d9"
  ink-2: "#bcb8ae"
  ink-3: "#908c84"
  ghost: "#5a5e63"
  ghost-text: "#7f7c75"
  accent: "#8fb3de"
  accent-soft: "rgba(143, 179, 222, 0.14)"
  up: "#58b88c"
  degraded: "#d9a646"
  refused: "#e0716a"
  idle: "#8894a2"
  sport-nfl: "#7b93c6"
  sport-cfb: "#c08a5b"
  sport-mlb: "#5fae98"
  sport-nhl: "#9c8fd6"
  sport-nba: "#d58a4a"
typography:
  display:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.02em"
    fontFeature: "'tnum' 1, 'lnum' 1"
  wordmark:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  title:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 650
    lineHeight: 1.3
    letterSpacing: "-0.01em"
    fontFeature: "'tnum' 1, 'lnum' 1"
  body:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "'tnum' 1, 'lnum' 1"
  body-sm:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.45
  caption:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.35
  label:
    fontFamily: "'Public Sans Variable', 'Public Sans', system-ui, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 500
    lineHeight: 1.35
  mono:
    fontFamily: "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"
    fontSize: "0.85em"
rounded:
  hair: "2px"
  cell: "3px"
  tile: "4px"
  dot: "50%"
spacing:
  s1: "4px"
  s2: "8px"
  s3: "12px"
  s4: "16px"
  s5: "24px"
  s6: "32px"
  s7: "48px"
components:
  circuit-tile:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.tile}"
    padding: "8px 12px"
  circuit-tile-hover:
    backgroundColor: "{colors.panel-2}"
  circuit-tile-active:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.ink}"
  evidence-cell:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    typography: "{typography.headline}"
    padding: "8px 12px"
  view-tab:
    textColor: "{colors.ink-3}"
    typography: "{typography.body-sm}"
    padding: "8px 12px"
  view-tab-active:
    textColor: "{colors.ink}"
  link-row:
    textColor: "{colors.ink}"
    padding: "12px 8px"
  link-row-hover:
    backgroundColor: "{colors.panel}"
  link-row-open:
    backgroundColor: "{colors.panel-2}"
  annunciator-cell:
    textColor: "{colors.ghost-text}"
    rounded: "{rounded.cell}"
    height: "20px"
  annunciator-cell-lit:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    rounded: "{rounded.cell}"
  absence-cell:
    textColor: "{colors.ink-3}"
    typography: "{typography.caption}"
    rounded: "{rounded.cell}"
    padding: "1px 8px"
  stamp:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink-2}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.tile}"
    padding: "8px 12px"
  button:
    backgroundColor: "{colors.panel-3}"
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.tile}"
    height: "32px"
    padding: "0 12px"
  button-ghost:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.tile}"
    height: "32px"
    padding: "0 12px"
  button-ghost-hover:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.ink}"
  segment:
    textColor: "{colors.ink-2}"
    typography: "{typography.caption}"
    rounded: "{rounded.cell}"
    height: "28px"
    padding: "0 10px"
  segment-on:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  input-number:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.tile}"
    height: "36px"
    padding: "0 12px"
  rail:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.tile}"
    padding: "16px"
---

# Design System: CoinFlip

## Overview

**Creative North Star: "The Network Operations Wall"**

CoinFlip is set up like a network operations wall. Each league is a monitored circuit, and you should be able to read its state (up, degraded, refused, idle) and its freshness at a glance. Each game is a link row on a dense, ruled table. The ground is blue-black. On top of it sit a few slightly cooler panel steps, divided by hairline rules. Figures are bone-grey tabular numerals that stay put when the data changes. The mood is an instrument panel that has been on all night: calm, dense, and honest about what it doesn't know.

Nothing on the wall shouts. The board has no hero, no gradient cards and no coloured confidence badges. Every row prints all four tiers as an annunciator, with only the game's own tier lit. Missing data is drawn rather than left out. "Unsized" and "No price" have their own cells, a refusal is a stamped line, and a league with no source still gets a tile. Motion shows up only when state changes.

The layout is built for laptop use at desktop density, with the phone as a fallback, and it expects a keyboard: every tile and tab carries its key.

**Key Characteristics:**
- Blue-black ground with three panel steps; depth comes from tone and 1px rules, not shadows.
- One typeface, Public Sans Variable, with tabular lining numerals everywhere.
- State always comes as a drawn 12px glyph plus a word. Colour only reinforces it.
- One cool blue accent for focus, selection, the active item and "changed since last visit".
- Sport colours appear only as an 8px swatch on the circuit tiles.
- Absence and refusal are drawn: dashed cells, stamped lines, ghosted unlit tiers.

## Colors

The palette is a near-black blue ground with warm bone ink, one cool blue signal and four state colours, all held at low chroma so no single hue takes over the wall.

### Primary
- **Signal Blue** (accent): the only interactive and attention colour. Used for the focus outline, the active circuit tile border, the active view-tab underline, links, the change tick and change-log glyphs, the "here" row in tables and the input focus border. Its soft wash (accent-soft) appears only as the input focus halo.

### Secondary (state)
- **Link Green** (up): the lit-link glyph on a circuit that has live prices.
- **Amber Degraded** (degraded): the split-dot glyph, inline warnings in row context (early season, back-to-back, scratches), the cap stamp border tint and the fixture banner ground.
- **Refusal Red** (refused): the barred-square glyph, refusal log lines and the refusal stamp border tint.
- **Slate Idle** (idle): the dash glyph for an off-season or unknown circuit.

### Tertiary (league identification)
- **NFL Steel, CFB Leather, MLB Turf, NHL Ice Violet, NBA Hardwood** (sport-nfl … sport-nba): an 8px, 2px-radius swatch before the league code on the circuit tiles. Nowhere else.

### Neutral
- **Operations Ground** (ground): the page, the top band and the inside of hollow markers.
- **Panel** (panel): circuit tiles, the evidence strip, the rail, stamps, inputs and row hover.
- **Panel Raised** (panel-2): the active or hovered tile, the open row and its detail, key help.
- **Panel Top** (panel-3): button fill, the avatar and the skip link.
- **Rule** (rule): structural hairlines (the band bottom, the tab bar, group heads, table heads) and dashed absence borders.
- **Soft Rule** (rule-soft): row dividers, cell dividers inside the evidence strip, resting tile borders.
- **Bone Ink** (ink): primary figures and text, and the lit annunciator cell fill.
- **Bone Ink 2** (ink-2): secondary figures, the market marker ring and positive bars.
- **Bone Ink 3** (ink-3): labels, notes, captions and column heads.
- **Ghost Text** (ghost-text): unlit annunciator labels; the dimmest text allowed, at 4.6:1 on the ground.
- **Ghost** (ghost): unlit marks, including the half-probability tick, key-number ticks, negative bars, the "no stake" dash and scrollbar hover.

### Named Rules
**The Glyph-Plus-Word Rule.** State is never colour alone. Each state has its own shape (a filled dot for up, a split dot for degraded, a barred hollow square for refused, a dash for idle) and a word next to it. Colour tints the shape and nothing more.

**The Swatch-Only Rule.** Sport colours identify a league on its circuit tile and do nothing else. No sport-tinted rows, headings, charts or backgrounds.

**The One Signal Rule.** Signal Blue marks where you are and what changed. It never decorates, and it never stands for "good".

## Typography

**Display Font:** Public Sans Variable, self-hosted (with system-ui, -apple-system, Segoe UI fallback)
**Body Font:** Public Sans Variable (same)
**Label/Mono Font:** the system monospace stack, used only for inline code such as file paths and ADR references

**Character:** This is one neutral civic grotesque doing all the work. Hierarchy comes from size steps of 11 to 24px and weights of 400 to 700, not from a second face. The body sets `tnum` and `lnum` globally, so every figure lines up in its column.

### Hierarchy
- **Display** (600, 24px, -0.02em): the slate title ("2026 · Week 3") and view titles. One per view.
- **Wordmark** (700, 20px, -0.02em, line-height 1): "CoinFlip" in the top band only.
- **Headline** (600, 16px, -0.01em): tier group heads, panel sub-heads, block heads and the big evidence figures.
- **Title** (650, 16px): the pick itself in a link row. It is the heaviest text in the row.
- **Body** (400, 14px, 1.45): team pairs, table cells and base text.
- **Body small** (400 to 600, 13px): the working size for notes, tabs, circuit state, buttons, detail tables and the rail. Prose measure is capped at 80 to 90ch.
- **Caption** (400, 12px): row context, scale figures, sources and the footer.
- **Label** (500, 11px, Bone Ink 3, sentence case): evidence-strip terms, column heads, table heads and capture ages.

### Named Rules
**The Tabular Rule.** Every figure uses tabular lining numerals, and numeric columns align right. No proportional digits anywhere on the wall.

**The One Face Rule.** Public Sans carries display, body and labels. Monospace is for literal code only, never for figures.

## Layout

The top band is a three-column grid: the brand on the left, five equal circuit tiles across the middle, and keys and account on the right. Below it sit the evidence strip (six equal cells in one bordered row), then the view tabs, then the body. The board body has two columns: the link table (flexible) and a sticky change-log rail (300px, or 260px at 1320px and below). Link rows share a fixed nine-column grid (tick, start, game, pick, annunciator 248px, scale, edge, stake, chevron), so the figures line up down the whole slate. Page gutters are 24px (s5). Group spacing is 24px, and rows are padded 12px vertically (6px in compact no-edge rows).

Spacing follows a 4px base: 4, 8, 12, 16, 24, 32, 48.

The layout responds at three points:
- **1100px:** the circuits wrap to their own row, the rail moves above the table and stops being sticky, and the evidence strip becomes three columns.
- **760px:** gutters drop to 16px, the tiles shed their figures and ages, key hints are hidden, the evidence strip becomes two columns, link rows restack into named areas (start | game/pick | stake/edge, then tier and scale full width), and the detail market tables turn into labelled card rows.
- **Reduced motion:** all animation and transitions are turned off.

## Elevation & Depth

The wall is flat. Depth comes from tone: ground, then panel, then panel-2, then panel-3, with 1px rules at two strengths. Borders separate things and fills show state. There is exactly one lifted element: the active circuit tile, which carries a soft ambient shadow along with its accent border. The view-tab underline is an inset 2px shadow used as a rule, not as elevation.

### Shadow Vocabulary
- **Active circuit lift** (`box-shadow: 0 1px 8px rgba(0, 0, 0, 0.35)`): only on the selected league tile.
- **Tab underline** (`box-shadow: inset 0 -2px 0 var(--accent)`): the active view tab. Transparent at rest.
- **Focus halo** (`box-shadow: 0 0 0 3px var(--accent-soft)`): number inputs on focus.

### Named Rules
**The Hairline Rule.** Separate things with a 1px rule or a panel step, never with a shadow. If a new surface seems to need elevation, give it the next panel tone.

## Shapes

Corners are small and square-shouldered. Tiles, panels, buttons, inputs, stamps and the evidence strip use 4px. Annunciator cells, segments, absence cells and kbd hints use 3px. Meters and the focus outline use 2px. Only markers are round: status dots, the market and model position markers, the avatar. Borders are 1px throughout. **Dashed borders mean absence or a stamp**: "No price", "Unsized", the empty state, and the dashed stamp line around refusals and caps. Solid borders mean structure.

Glyphs are drawn on one 12px grid with one 1.5px stroke and square geometry. They're inline SVG using currentColor, never an icon font.

## Components

### Circuit Tiles (signature)
These are the five league tiles across the top band, and they're the first thing read on the wall.
- **Structure:** the league code with its sport swatch and key hint, then the state glyph and word (13px/600), then priced/total and play count (12px, Bone Ink 2), then capture age (11px, Bone Ink 3).
- **Rest:** panel fill, soft-rule border, 4px radius.
- **Hover:** panel-2 fill and a rule-strength border (160ms, ease-out-expo `cubic-bezier(0.16, 1, 0.3, 1)`).
- **Active:** panel-2, a Signal Blue border and the one ambient lift.

### Evidence Strip
- One bordered, 4px-radius row of six `dt`/`dd` cells divided by soft rules. Each cell has an 11px label, a 16px/600 figure (with a smaller denominator such as "0/150"), an optional 4px meter, and an 11px source line. The figure always carries where it came from.

### Link Rows and the Annunciator (signature)
- **Row:** a full-width button on the shared grid, divided by soft rules. Hover goes to panel, open goes to panel-2, and the detail slides in at 180ms (opacity 0.4 to 1, 3px rise). A Signal Blue tick marks a row that changed since your last visit.
- **Annunciator:** four equal cells for Play / Lean / Coin flip / No edge, always all printed. Unlit cells have a soft-rule border and Ghost Text (4.6:1 on the ground: dimmed, still legible). The lit cell is a Bone Ink fill with Ground text at 700 weight.
- **Scale:** a 2px rule track with a half-probability tick, a hollow market marker (1.5px Bone Ink 2 ring), a solid Bone Ink model marker and a pale bar across the gap between them.
- **Absence:** "No price" and "Unsized" are dashed 3px cells in caption type. A "no stake" is a ghost dash, never a blank.

### Stamps
- **Style:** a dashed rule border, 4px radius, panel fill, 13px Bone Ink 2 text, led by a state glyph.
- **Variants:** the refusal stamp has a red-tinted border, the cap stamp an amber-tinted border, and the idle stamp a solid soft rule.

### Buttons
- **Shape:** gently squared (4px), 32px high, 12px side padding, 13px/600.
- **Primary:** panel-3 fill, rule border, Bone Ink text. On hover the border turns Signal Blue. When disabled the button goes transparent with Bone Ink 3 text.
- **Ghost (Keys, account):** transparent with a rule border and Bone Ink 2 text. On hover it fills with panel-2 and the text goes to Bone Ink.
- **Link button:** Signal Blue text with no chrome, underlined on hover.

### Segments (chips)
- A 28px, 3px-radius outlined toggle in 12px text. Hover fills with panel-2. When on, it takes a Bone Ink fill with Ground text, the same "lit" language as the annunciator.

### Inputs / Fields
- **Style:** 36px high, panel fill, rule border, 4px radius, 14px text. The label sits above in 13px Bone Ink 2.
- **Focus:** a Signal Blue border plus a 3px accent-soft halo. Range inputs use accent-color Signal Blue.

### Navigation
- **View tabs:** text tabs in 13px/500 Bone Ink 3, each with its kbd hint, on a tab bar with a rule-strength bottom border. Hover goes to Bone Ink. When active, the tab is Bone Ink with a 2px Signal Blue inset underline. On phones the bar scrolls horizontally and the hints are hidden.
- **Keyboard hints:** 11px/600 kbd chips with a 1px rule border and 3px radius.

### Change-Log Rail
- A sticky panel (4px radius, soft-rule border, 16px padding). Log lines are a 14px glyph column plus text, divided by soft rules. Change glyphs are drawn by kind (line or price move, tier step, side switch, refusal) in Signal Blue, and refusals in Refusal Red. A facts list at the bottom sits under a rule.

## Do's and Don'ts

### Do:
- **Do** pair every state with its drawn 12px glyph and a word; colour only tints the glyph.
- **Do** print all four tiers on every priced row and light only the game's own.
- **Do** draw absence: dashed cells for "No price" and "Unsized", a ghost dash for no stake, a stamped line for a refusal. Never drop a row or leave a blank.
- **Do** carry the source under a figure (an 11px line in Bone Ink 3) wherever the figure is a claim.
- **Do** set every figure in tabular lining numerals and right-align numeric columns.
- **Do** separate with 1px rules and panel steps. The only lift is on the active circuit tile.
- **Do** limit motion to state changes at 160 to 180ms on `cubic-bezier(0.16, 1, 0.3, 1)`, and turn it off under reduced motion.
- **Do** keep labels and headings in sentence case at the 11 to 24px steps.

### Don't:
- **Don't** use sport colours anywhere except the 8px swatch on a circuit tile.
- **Don't** signal state, confidence or tier with colour alone, and don't give tiers their own colours. The lit annunciator cell is Bone Ink, not green.
- **Don't** use Signal Blue for decoration or to mean "good". It marks focus, selection, the active item and change only.
- **Don't** build pick cards with gradients, coloured confidence badges or a hero. The board is a ruled table of link rows.
- **Don't** add a second typeface, or monospace figures. Public Sans carries everything, and mono is for literal code.
- **Don't** use icon fonts or glyph characters for state. Draw inline SVG on the 12px, 1.5px-stroke grid.
- **Don't** add load choreography. Nothing animates in on arrival except the opened row's detail.
