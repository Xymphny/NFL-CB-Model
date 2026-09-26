/* Data access. The site renders files the core writes and computes nothing
 * of its own: every probability, edge, tier, stake and status arrives in
 * data/site/*.json (scripts/export_board.py, scripts/export_record.py).
 *
 * The one thing done here is DIFFING: comparing this board with the one the
 * owner last looked at, to list what moved. That compares two published
 * numbers; it never produces a new one.
 */
import { useEffect, useState } from 'react'

export const LEAGUES = ['nfl', 'cfb', 'mlb', 'nhl', 'nba']

/* ?fixture=1 loads SYNTHETIC prices from /dev-fixtures (scripts/dev/
 * board_fixture.py) for design work. Never the default. */
const FIXTURE = typeof window !== 'undefined'
  && new URLSearchParams(window.location.search).get('fixture') === '1'

const paths = FIXTURE
  ? {
      board: (l) => `/dev-fixtures/board_${l}.json`,
      record: '/dev-fixtures/record.json',
      gates: '/dev-fixtures/gates.json',
      props: '/dev-fixtures/prop_grades_summary.json',
      nbaPlayers: '/dev-fixtures/players_nba.json',
      page: (f) => `/dev-fixtures/${f}`,
    }
  : {
      board: (l) => `/data/site/board_${l}.json`,
      record: '/data/site/record.json',
      gates: '/data/site/gates.json',
      props: '/data/prop_grades/summary.json',
      nbaPlayers: '/data/site/players_nba.json',
      page: (f) => `/data/site/${f}`,
    }

export const isFixture = FIXTURE

function useJson(url) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let live = true
    setState((s) => ({ ...s, loading: true }))
    fetch(url, { cache: 'no-store' })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${url}`)
        return r.json()
      })
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error) => live && setState({ data: null, error, loading: false }))
    return () => { live = false }
  }, [url])
  return state
}

export function useBoards() {
  const [boards, setBoards] = useState({})
  const [errors, setErrors] = useState({})
  useEffect(() => {
    let live = true
    LEAGUES.forEach((l) => {
      fetch(paths.board(l), { cache: 'no-store' })
        .then((r) => {
          if (!r.ok) throw new Error(`${r.status}`)
          return r.json()
        })
        .then((b) => live && setBoards((prev) => ({ ...prev, [l]: b })))
        .catch((e) => live && setErrors((prev) => ({ ...prev, [l]: String(e.message || e) })))
    })
    return () => { live = false }
  }, [])
  return { boards, errors }
}

export const useRecord = () => useJson(paths.record)
export const useGates = () => useJson(paths.gates)
export const usePropLedger = () => useJson(paths.props)
export const useNbaPlayers = () => useJson(paths.nbaPlayers)

/* Team and league pages (scripts/export_teams.py, scripts/export_outlook.py).
 * Every rank arrives computed; the views lay them out. */
export const TEAM_PAGES = {
  nfl: 'teams_nfl.json', nba: 'teams_nba.json',
  mlb: 'pitchers_parks_mlb.json', nhl: 'attack_defence_nhl.json',
}
export const useTeamPage = (league) => useJson(paths.page(TEAM_PAGES[league] || 'none.json'))
export const useOutlook = () => useJson(paths.page('outlook_nfl.json'))

/* ------------------------------------------------ since your last look ---- */

const SEEN_KEY = 'coinflip_seen_v1'

function readSeen() {
  try {
    return JSON.parse(localStorage.getItem(SEEN_KEY)) || {}
  } catch {
    return {}
  }
}

function snapshotOf(board) {
  const games = {}
  ;(board.games || []).forEach((g) => {
    const m = g.markets?.[g.headline]
    games[g.game_id] = {
      line: m?.line ?? null,
      side: m?.side ?? null,
      p_market: m?.status === 'priced' ? m.p_market : null,
      tier: m?.status === 'priced' ? m.tier : null,
      refusal: g.refusal || (m && m.status === 'refused' ? m.refusal : null),
    }
  })
  const leagueRefusals = (board.refusals || [])
    .filter((r) => r.scope === 'league').map((r) => r.reason)
  return { generated_at: board.generated_at, games, leagueRefusals }
}

/* Events between the owner's last view of a league's board and this one.
 * Returns [] on a first visit: nothing is "new" without a baseline. */
export function changesSince(prev, board) {
  if (!prev || !board) return []
  const now = snapshotOf(board)
  const out = []
  const label = Object.fromEntries((board.games || []).map((g) => [g.game_id, `${g.away} @ ${g.home}`]))
  Object.entries(now.games).forEach(([id, cur]) => {
    const was = prev.games?.[id]
    if (!was) {
      out.push({ id, kind: 'new', text: `${label[id]} added to the slate` })
      return
    }
    if (was.line != null && cur.line != null && was.side === cur.side && was.line !== cur.line) {
      out.push({ id, kind: 'line', from: was.line, to: cur.line, text: `${label[id]} line` })
    } else if (was.line == null && cur.line == null && was.side === cur.side
               && was.p_market != null && cur.p_market != null
               && Math.abs(cur.p_market - was.p_market) >= 0.005) {
      // A moneyline has no line to move; its price does. Both numbers are
      // the core's published market probabilities -- this only compares them.
      out.push({ id, kind: 'price', from: was.p_market, to: cur.p_market, text: `${label[id]} market` })
    } else if (was.side && cur.side && was.side !== cur.side) {
      out.push({ id, kind: 'side', text: `${label[id]}: the model now prefers the ${cur.side} side` })
    }
    if (was.tier !== cur.tier && cur.tier) {
      out.push({ id, kind: 'tier', from: was.tier, to: cur.tier, text: `${label[id]} tier` })
    }
    if (cur.refusal && cur.refusal !== was.refusal) {
      out.push({ id, kind: 'refusal', text: `${label[id]}: ${cur.refusal}` })
    }
  })
  now.leagueRefusals.forEach((r) => {
    if (!(prev.leagueRefusals || []).includes(r)) out.push({ id: null, kind: 'refusal', text: r })
  })
  return out
}

/* The baseline is the board the owner saw on the previous visit. It is read
 * once per league per page load -- so changes stay listed while this visit
 * lasts -- and the current board is stored as the next visit's baseline. */
const baselines = {}

export function useSinceLastVisit(league, board) {
  const [, bump] = useState(0)
  useEffect(() => {
    if (!board) return
    if (!(league in baselines)) {
      const all = readSeen()
      baselines[league] = all[league] || null
      try {
        all[league] = { ...snapshotOf(board), seen_at: new Date().toISOString() }
        localStorage.setItem(SEEN_KEY, JSON.stringify(all))
      } catch { /* private mode: no baseline next time */ }
      bump((n) => n + 1)
    }
  }, [league, board])
  const baseline = baselines[league] || null
  return { baseline, changes: changesSince(baseline, board) }
}
