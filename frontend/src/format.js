/* Formatting only. Nothing here derives a number the core did not publish:
 * these turn published values into text. */

export const LEAGUE_NAME = { nfl: 'NFL', cfb: 'College football', mlb: 'MLB', nhl: 'NHL', nba: 'NBA' }
export const LEAGUE_CODE = { nfl: 'NFL', cfb: 'CFB', mlb: 'MLB', nhl: 'NHL', nba: 'NBA' }

export const TIERS = ['play', 'lean', 'coin_flip', 'no_edge']
export const TIER_LABEL = { play: 'Play', lean: 'Lean', coin_flip: 'Coin flip', no_edge: 'No edge' }

export const STATE_LABEL = { up: 'Priced', degraded: 'Partial', refused: 'Refused', idle: 'Idle' }

export const MARKET_LABEL = {
  spread: 'Spread', moneyline: 'Moneyline', total: 'Total', runline: 'Run line',
  puck_line: 'Puck line',
}

export const pct = (p, digits = 1) => (p == null ? '—' : `${(p * 100).toFixed(digits)}%`)

/* An edge is published in probability points (0-1); shown as points. */
export const pts = (e, digits = 1) =>
  e == null ? '—' : `${e > 0 ? '+' : e < 0 ? '−' : ''}${Math.abs(e * 100).toFixed(digits)}`

export const signed = (v, digits = 1) =>
  v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`

export const line = (v) => {
  if (v == null) return 'ML'
  if (v === 0) return 'PK'
  return `${v > 0 ? '+' : '−'}${Math.abs(v) % 1 === 0 ? Math.abs(v) : Math.abs(v).toFixed(1)}`
}

export const american = (a) => (a == null ? '—' : a > 0 ? `+${a}` : `−${Math.abs(a)}`)

export const BOOK = {
  pinnacle: 'Pinnacle', fanduel: 'FanDuel', draftkings: 'DraftKings', betmgm: 'BetMGM',
  caesars: 'Caesars', betrivers: 'BetRivers', bovada: 'Bovada', mybookieag: 'MyBookie',
  lowvig: 'LowVig', betonlineag: 'BetOnline', williamhill_us: 'Caesars', espnbet: 'ESPN Bet',
}
export const book = (b) => (b ? BOOK[b] || b : '—')

const ET = 'America/New_York'

export function startTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: ET })
}

export function startDay(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: ET })
}

export function dateLabel(isoDate) {
  if (!isoDate) return ''
  const [y, m, d] = isoDate.split('-').map(Number)
  return new Date(Date.UTC(y, m - 1, d, 12)).toLocaleDateString('en-US', {
    weekday: 'long', month: 'long', day: 'numeric', timeZone: 'UTC',
  })
}

export function age(iso, now = Date.now()) {
  if (!iso) return null
  const t = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T')).getTime()
  if (Number.isNaN(t)) return null
  const m = Math.round((now - t) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 48) return `${h} h ago`
  return `${Math.round(h / 24)} d ago`
}

export const money = (v) =>
  v == null ? '—' : v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })

export const sideTeam = (g, side) =>
  side === 'home' ? g.home : side === 'away' ? g.away : side === 'over' ? 'Over' : side === 'under' ? 'Under' : '—'
