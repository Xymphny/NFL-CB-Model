import { Fragment, useCallback, useEffect, useState } from 'react'
import { useAccount } from './account'
import { useBook } from './store'
import { LEAGUES, isFixture, useBoards, useGates, useRecord, useSinceLastVisit } from './data'
import { LEAGUE_CODE, LEAGUE_NAME, STATE_LABEL, age, dateLabel, pts, signed } from './format'
import { StateGlyph } from './Glyph'
import Board from './views/Board'
import Teams from './views/Teams'
import Players from './views/Players'
import Record from './views/Record'
import Gates from './views/Gates'
import Book, { AccountChip } from './views/Book'

/* Views a league actually has. Players only where a player model exists
 * (NFL's props engine); Teams only where the board says what the model rates.
 * Tabs that do not apply are hidden, never rendered empty. */
function viewsFor(league, board) {
  const v = [{ id: 'board', label: 'Board', key: 'b' }]
  if (board?.teams?.length) v.push({ id: 'teams', label: 'Teams', key: 't' })
  if (league === 'nfl') v.push({ id: 'players', label: 'Players', key: 'p' })
  v.push({ id: 'record', label: 'Record', key: 'r' }, { id: 'gates', label: 'Gates', key: 'g' },
         { id: 'book', label: 'My book', key: 'm' })
  return v
}

function readPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback } catch { return fallback }
}
function writePref(key, value) {
  try { localStorage.setItem(key, value) } catch { /* private mode */ }
}

export default function App() {
  const account = useAccount()
  const book = useBook(account)
  const { boards, errors } = useBoards()
  const record = useRecord()
  const gates = useGates()

  /* ?league=nhl&view=record opens a specific board: a link a friend can
   * be sent. Without it, the last league viewed on this browser. */
  const params = new URLSearchParams(window.location.search)
  const [league, setLeagueState] = useState(() => {
    const l = params.get('league') || readPref('coinflip_league', 'nfl')
    return LEAGUES.includes(l) ? l : 'nfl'
  })
  const [view, setView] = useState(() => params.get('view') || 'board')
  const [showKeys, setShowKeys] = useState(false)
  const board = boards[league]
  const views = viewsFor(league, board)
  const since = useSinceLastVisit(league, board)

  const setLeague = useCallback((l) => {
    setLeagueState(l)
    writePref('coinflip_league', l)
  }, [])

  /* A view the league does not have falls back to the board -- but only once
   * the board has loaded. Before that, viewsFor() cannot know whether Teams
   * exists, and resetting then broke every ?view=teams deep link. */
  useEffect(() => {
    if (board && !views.some((v) => v.id === view)) setView('board')
  }, [league, board])

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const tag = (e.target.tagName || '').toLowerCase()
      if (['input', 'select', 'textarea'].includes(tag)) return
      const n = Number(e.key)
      if (n >= 1 && n <= LEAGUES.length) { setLeague(LEAGUES[n - 1]); return }
      if (e.key === '?') { setShowKeys((s) => !s); return }
      if (e.key === 'Escape') { setShowKeys(false); return }
      const v = views.find((x) => x.key === e.key)
      if (v) setView(v.id)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [views, setLeague])

  return (
    <div className="shell">
      <a className="skip" href="#main">Skip to the board</a>
      <header className="top">
        <div className="brand">
          <span className="wordmark">Coinflip</span>
          <span className="motto">Every number on this screen was earned by evidence.</span>
        </div>
        <nav className="circuits" aria-label="Leagues">
          {LEAGUES.map((l, i) => (
            <Circuit key={l} league={l} index={i + 1} board={boards[l]} error={errors[l]}
                     active={l === league} onSelect={() => setLeague(l)} />
          ))}
        </nav>
        <div className="top-end">
          <AccountChip account={account} />
          <button className="keys-btn" onClick={() => setShowKeys((s) => !s)} aria-expanded={showKeys}>
            Keys <kbd>?</kbd>
          </button>
        </div>
      </header>

      {isFixture && (
        <div className="fixture-banner" role="status">
          Synthetic prices loaded for design work (<code>?fixture=1</code>). These are not real odds.
        </div>
      )}
      {showKeys && <KeyHelp views={views} onClose={() => setShowKeys(false)} />}

      <EvidenceStrip league={league} board={board} record={record.data} />

      <nav className="views" aria-label={`${LEAGUE_NAME[league]} views`}>
        {views.map((v) => (
          <button key={v.id} className={`view-tab${view === v.id ? ' on' : ''}`}
                  aria-current={view === v.id ? 'page' : undefined} onClick={() => setView(v.id)}>
            {v.label}<kbd>{v.key}</kbd>
          </button>
        ))}
      </nav>

      <main id="main" className="main">
        {!board && !errors[league] && <LoadingBoard />}
        {errors[league] && (
          <div className="stamp stamp-refused">
            <StateGlyph state="refused" />
            <p>The {LEAGUE_NAME[league]} board could not be loaded ({errors[league]}). Nothing
              below is current; reload when the site has redeployed.</p>
          </div>
        )}
        {board && view === 'board' && <Board league={league} board={board} since={since} book={book} />}
        {board && view === 'teams' && <Teams league={league} board={board} />}
        {view === 'players' && league === 'nfl' && <Players />}
        {view === 'record' && <Record league={league} record={record} />}
        {view === 'gates' && <Gates league={league} gates={gates} />}
        {view === 'book' && <Book book={book} account={account} />}
      </main>

      <footer className="foot">
        <p>Coinflip suggests; it never instructs. It shows what the model thinks against what the
          market thinks. Nothing here is sized until a league is graded against closing prices.
          Bet only what you can afford to lose. If gambling is causing harm, call 1-800-GAMBLER.</p>
      </footer>
    </div>
  )
}

function Circuit({ league, index, board, error, active, onSelect }) {
  const st = board?.status
  const state = error ? 'refused' : st?.state || 'idle'
  const captured = board?.freshness?.odds_captured_at
  return (
    <button className={`circuit c-${league}${active ? ' on' : ''} s-${state}`} onClick={onSelect}
            aria-pressed={active} aria-label={`${LEAGUE_NAME[league]}: ${STATE_LABEL[state]}`}>
      <span className="c-head">
        <span className="c-code">{LEAGUE_CODE[league]}</span>
        <kbd>{index}</kbd>
      </span>
      <span className="c-state">
        <StateGlyph state={state} />
        {board ? STATE_LABEL[state] : error ? 'Unavailable' : 'Loading'}
      </span>
      <span className="c-figs">
        {!board ? ' ' : st?.games
          ? <>{st.priced}/{st.games} priced{st.tiers?.play ? ` · ${st.tiers.play} Play` : ''}</>
          : board.next_slate ? <>Next {board.next_slate.date.slice(5).replace('-', '/')}</>
          : state === 'refused' ? 'See refusal' : 'No games'}
      </span>
      <span className="c-age">{captured ? `odds ${age(captured)}` : 'no odds yet'}</span>
    </button>
  )
}

function EvidenceStrip({ league, board, record }) {
  const g = board?.grade
  const r = record?.leagues?.[league]
  const floor = record?.floor || 150
  const settled = r?.settled_games ?? 0
  const tiers = board?.tiers
  const leagueRefusals = (board?.refusals || []).filter((x) => x.scope === 'league')
  const fresh = board?.freshness || {}
  return (
    <section className="evidence" aria-label={`${LEAGUE_NAME[league]} evidence`}>
      <dl className="ev-grid">
        <div className="ev">
          <dt>Market grade</dt>
          <dd>
            <span className="ev-big">{g ? g.weight.toFixed(2) : '—'}</span>
            <span className="ev-sub" title={g?.source ? `measured weight ${g.w_hat} ± ${g.se} over ${g.n} games${g.contamination ? `; ${g.contamination}` : ''}` : undefined}>
              {g?.source
                ? <>w {signed(g.w_hat, 3)} ± {g.se?.toFixed(3)} · {g.n?.toLocaleString()} games{g.contamination?.startsWith('UPPER') ? ' · upper bound' : ''}</>
                : 'not yet graded against closing prices'}
            </span>
          </dd>
        </div>
        <div className="ev">
          <dt>Paper trades</dt>
          <dd>
            <span className="ev-big">{settled}<small>/{floor}</small></span>
            <span className="meter" role="meter" aria-valuemin={0} aria-valuemax={floor} aria-valuenow={settled}
                  aria-label={`${settled} of ${floor} settled`}>
              <span style={{ width: `${Math.min(100, (r?.progress_to_floor || 0) * 100)}%` }} />
            </span>
          </dd>
        </div>
        <div className="ev">
          <dt>Mean CLV</dt>
          <dd>
            <span className="ev-big">{r?.mean_clv_prob_points != null ? pts(r.mean_clv_prob_points, 2) : '—'}</span>
            <span className="ev-sub">{r?.clv_graded ? `${r.clv_graded} closes · points of probability` : 'no closes graded yet'}</span>
          </dd>
        </div>
        <div className="ev">
          <dt>Odds</dt>
          <dd>
            <span className="ev-big">{fresh.odds_captured_at ? age(fresh.odds_captured_at) : 'none'}</span>
            <span className="ev-sub">{fresh.odds_captured_at ? 'last capture' : 'no capture yet'}</span>
          </dd>
        </div>
        <div className="ev">
          <dt>Inputs</dt>
          <dd>
            <span className="ev-big">{board?.generated_at ? age(board.generated_at) : '—'}</span>
            <span className="ev-sub">
              {fresh.ratings_version ? `ratings through week ${fresh.ratings_through_week}`
                : fresh.latest_final ? `results to ${String(fresh.latest_final).slice(0, 10)}`
                : 'board exported'}
            </span>
          </dd>
        </div>
        <div className="ev">
          <dt>Tiers</dt>
          <dd>
            <span className="ev-big ev-word">{tiers ? (tiers.provisional ? 'Provisional' : 'Calibrated') : '—'}</span>
            <span className="ev-sub">
              {tiers ? (tiers.source?.startsWith('borrowed') ? `borrowed from ${tiers.source.split(':')[1].toUpperCase()}` : `from the ${tiers.source}`) : ''}
              {tiers ? ` · Play ≥ ${pts(tiers.play)} pts` : ''}
            </span>
          </dd>
        </div>
      </dl>
      {leagueRefusals.map((x, i) => (
        <p className="stamp stamp-inline" key={i}>
          <StateGlyph state={board?.status?.state === 'refused' ? 'refused' : 'degraded'} />
          <span>{x.reason}</span>
        </p>
      ))}
      {(board?.status?.game_refusals || []).map((x) => (
        <p className="stamp stamp-inline" key={x.reason}>
          <StateGlyph state="refused" />
          <span>{x.games} game{x.games === 1 ? '' : 's'} refused: {x.reason}.</span>
        </p>
      ))}
      {board?.status?.state === 'idle' && (
        <p className="stamp stamp-inline stamp-idle">
          <StateGlyph state="idle" />
          <span>No {LEAGUE_NAME[league]} games today.
            {board.next_slate ? ` Next slate ${dateLabel(board.next_slate.date)}, ${board.next_slate.games} game${board.next_slate.games === 1 ? '' : 's'}.` : ''}</span>
        </p>
      )}
    </section>
  )
}

function LoadingBoard() {
  return (
    <div className="skeleton" aria-busy="true" aria-label="Loading the board">
      {Array.from({ length: 6 }).map((_, i) => <div key={i} className="sk-row" />)}
    </div>
  )
}

function KeyHelp({ views, onClose }) {
  return (
    <aside className="keyhelp" aria-label="Keyboard shortcuts">
      <div className="kh-grid">
        <span><kbd>1</kbd>–<kbd>5</kbd></span><span>Switch league</span>
        {views.map((v) => (
          <Fragment key={v.id}><span><kbd>{v.key}</kbd></span><span>{v.label}</span></Fragment>
        ))}
        <span><kbd>j</kbd> <kbd>k</kbd></span><span>Move between games</span>
        <span><kbd>Enter</kbd></span><span>Open or close the reasoning</span>
        <span><kbd>?</kbd></span><span>Show or hide this</span>
      </div>
      <button className="link-btn" onClick={onClose}>Close</button>
    </aside>
  )
}

