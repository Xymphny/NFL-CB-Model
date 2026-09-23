---
version: 1
slug: "frontend-src-app-jsx"
primary_target: "frontend/src/App.jsx"
related_targets: []
---

# Surface: CoinFlip dashboard (frontend/src/App.jsx)

Mode: Operate. The owner keeps it open all day on a laptop and reloads as news
lands; friends open it for the picks. Same site for both, picks-first.

Task: see, per league, what the model thinks against the market for every game,
how strongly (tier), whether it is sized (never, until graded), what moved since
the last look, and whether each number is backed. Constraints: renders
data/site/*.json and computes nothing; status never by colour alone; tabular
numerals; dark; desktop-dense, phone as fallback; keyboard-first.

## Direction contract

THESIS: A network operations wall for betting markets. Each league is a monitored
circuit whose state (up, degraded, refused) and freshness read at a glance; each
game is a link row. It refuses the category default: a sportsbook grid of
gradient pick cards with coloured badges shouting confidence.

OWN-WORLD: Blue-black operations ground, one cooler panel layer, hairline rules.
Bone-grey figures in tabular numerals. State is a drawn glyph plus a word: filled
dot up, split dot degraded, hollow square refused, dash unknown; colour only
reinforces. Sport accents are small identification swatches on the circuit
tiles, nowhere else. All four tiers are printed on every row as a four-cell
annunciator; the game's tier is lit and weighted, the rest ghosted. Absence is
drawn: "Unsized" and "No price" have designed cells. Nothing disappears: a
refusal is a stamped row, never a missing one.

STORY: The owner sees five circuits and knows which leagues have live prices;
picks one; reads the evidence strip (grade, paper trades toward 150, CLV,
freshness, refusals); scans link rows grouped Play, Lean, Coin flip, then the
compact No-edge list; opens a row for the reasoning; checks the change log for
what moved since the last visit.

FIRST VIEWPORT: Top band: CoinFlip wordmark left, five circuit tiles across (code,
state glyph and word, priced/total games, Play count, capture age), account chip
right. Under it the evidence strip, one ruled row. Then a two-column body: the
link table (about 70%) with the annunciator column and a model-vs-market bar per
row; the change-log rail (about 30%) on the right. View tabs (Board, Teams,
Players where a player model exists, Record, Gates, My book) sit between the
strip and the body. No hero, no kicker.

FORM: Network Operations Wall, IMPECCABLE'S PICK from the grounded list
(position 1 of 7), chosen over the assigned Past Performances; seed ed246bed.
Kept disciplines: every tier printed with the active one struck (cathode);
nothing disappears, it cancels (ticket wallet); absence drawn (seven-segment);
figures carry their source (claim and proof).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

Signature interaction: the change log. On load the site compares the board with
the owner's last view (per browser) and lists line moves, tier changes and new
refusals as log lines; the matching rows carry a change tick until seen.
Motion: state changes only, 150-200ms, no load choreography.
