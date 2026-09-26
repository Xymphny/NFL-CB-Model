import { useEffect, useMemo, useRef, useState } from 'react'
import { ChangeGlyph, Chevron, StateGlyph } from '../Glyph'
import { ContextLines, InjuryReport, KeyPlayerStrip, LineSpark, OpenMatchup } from './GameContext'
import {
  MARKET_LABEL, TIERS, TIER_LABEL, american, book as bookName, line as fmtLine, money, pct, pts,
  signed, startDay, startTime,
} from '../format'

/* The board renders data/site/board_{league}.json. It computes no
 * probability, edge, tier or stake: every one arrives from the core
 * (scripts/export_board.py). */

const GROUPS = [
  { tier: 'play', label: 'Play', note: "the model's strongest disagreements with the market" },
  { tier: 'lean', label: 'Lean', note: 'a real but smaller disagreement' },
  { tier: 'coin_flip', label: 'Coin flip', note: 'near agreement; it names the side it slightly prefers' },
]

export default function Board({ league, board, since, book, onOpenMatchup, onPlayers }) {
  const [open, setOpen] = useState(() => new Set())
  const rows = useRef([])

  const changed = useMemo(() => {
    const m = new Map()
    since.changes.forEach((c) => { if (c.id) m.set(c.id, [...(m.get(c.id) || []), c]) })
    return m
  }, [since.changes])

  const head = (g) => g.markets?.[g.headline]
  const grouped = useMemo(() => {
    const by = { play: [], lean: [], coin_flip: [], no_edge: [], unpriced: [], started: [] }
    ;(board.games || []).forEach((g) => {
      const m = head(g)
      if (g.started) by.started.push(g)
      else if (m?.status === 'priced' && m.tier) by[m.tier].push(g)
      else by.unpriced.push(g)
    })
    return by
  }, [board])

  const order = [...grouped.play, ...grouped.lean, ...grouped.coin_flip, ...grouped.no_edge,
    ...grouped.unpriced, ...grouped.started]
  const toggle = (id) => setOpen((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n })

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (!['j', 'k'].includes(e.key)) return
      const tag = (e.target.tagName || '').toLowerCase()
      if (['input', 'select', 'textarea'].includes(tag)) return
      const els = rows.current.filter(Boolean)
      const i = els.indexOf(document.activeElement)
      const next = e.key === 'j' ? Math.min(els.length - 1, i + 1) : Math.max(0, i - 1)
      els[i === -1 ? 0 : next]?.focus()
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [board])

  const games = board.games || []
  const idx = (g) => order.indexOf(g)
  const rowProps = (g) => ({
    g, league, board, book, onOpenMatchup, onPlayers,
    isOpen: open.has(g.game_id), onToggle: () => toggle(g.game_id),
    changes: changed.get(g.game_id) || [],
    rowRef: (el) => { rows.current[idx(g)] = el },
  })

  return (
    <div className={`board-grid${board.slate?.kind === 'week' ? ' weekly' : ''}`}>
      <div className="board-main">
        <SlateLine league={league} board={board} />
        {board.synthetic && <p className="stamp stamp-inline"><StateGlyph state="degraded" />{board.synthetic}</p>}

        {games.length === 0 && <EmptySlate board={board} />}

        {GROUPS.map(({ tier, label, note }) => grouped[tier].length > 0 && (
          <section className="group" key={tier} aria-label={`${label} tier`}>
            <header className="group-head">
              <h2>{label}</h2>
              <span className="group-count">{grouped[tier].length}</span>
              <span className="group-note">{note}</span>
            </header>
            <ColumnHeads />
            {grouped[tier].map((g) => <GameRow key={g.game_id} {...rowProps(g)} />)}
          </section>
        ))}

        {grouped.no_edge.length > 0 && (
          <section className="group group-quiet" aria-label="No edge">
            <header className="group-head">
              <h2>No edge</h2>
              <span className="group-count">{grouped.no_edge.length}</span>
              <span className="group-note">the model agrees with the market; shown so you can see it</span>
            </header>
            {grouped.no_edge.map((g) => <GameRow key={g.game_id} {...rowProps(g)} compact />)}
          </section>
        )}

        {grouped.unpriced.length > 0 && (
          <section className="group group-quiet" aria-label="Not priced against the market">
            <header className="group-head">
              <h2>Not priced</h2>
              <span className="group-count">{grouped.unpriced.length}</span>
              <span className="group-note">no market captured yet, or the core refused the line; the model's own view is kept</span>
            </header>
            {grouped.unpriced.map((g) => <GameRow key={g.game_id} {...rowProps(g)} compact />)}
          </section>
        )}

        {grouped.started.length > 0 && (
          <section className="group group-quiet" aria-label="Started">
            <header className="group-head">
              <h2>Started</h2>
              <span className="group-count">{grouped.started.length}</span>
              <span className="group-note">kicked off; nothing is priced after the start, and each settles against its close and final score</span>
            </header>
            {grouped.started.map((g) => <GameRow key={g.game_id} {...rowProps(g)} compact />)}
          </section>
        )}

        <p className="board-note">
          A tier is conviction, not edge: how far the model sits from the market on this game,
          against the league's own history. No band has been distinguishable from break-even
          (ADR 0025). Stakes come only from the market grade, which is {board.grade?.weight > 0
            ? `${board.grade.weight.toFixed(2)} for this league.` : 'zero for this league, so every pick is unsized.'}
        </p>
      </div>
      <ChangeLog since={since} board={board} />
    </div>
  )
}

function SlateLine({ league, board }) {
  const s = board.slate
  const st = board.status || {}
  return (
    <div className="slate">
      <h1>
        {s?.kind === 'week' ? `${s.season} · Week ${s.week}` : s?.date ? startDay(`${s.date}T16:00:00Z`) : 'No slate'}
      </h1>
      <span className="slate-figs">
        {st.games ? `${st.games} games · ${st.priced} priced against the market` : ''}
      </span>
      <p className="slate-note">{board.note}</p>
    </div>
  )
}

function EmptySlate({ board }) {
  const st = board.status || {}
  if (st.state === 'refused') {
    return (
      <div className="empty">
        <h2>Nothing is priced</h2>
        <p>The core refused this league, and the reason is stamped above. The board stays
          empty rather than show a stale slate as today's.</p>
      </div>
    )
  }
  return (
    <div className="empty">
      <h2>No games on today's slate</h2>
      <p>{board.next_slate
        ? `The next slate has ${board.next_slate.games} game${board.next_slate.games === 1 ? '' : 's'}. Prices appear here after the odds capture before them.`
        : 'Prices appear here once games are scheduled and the odds capture has run.'}</p>
    </div>
  )
}

function ColumnHeads() {
  return (
    <div className="cols" aria-hidden="true">
      <span />
      <span>Start</span>
      <span>Game</span>
      <span>Pick</span>
      <span>Tier</span>
      <span>Model vs market</span>
      <span className="num" title="Model probability minus the market's, in points. Not edge: the stake comes from the league's grade against the close.">Gap</span>
      <span className="num">Stake</span>
      <span />
    </div>
  )
}

function Annunciator({ tier, capped }) {
  return (
    <span className="ann" role="img" aria-label={`Tier: ${TIER_LABEL[tier]}${capped ? ', capped' : ''}`}>
      {TIERS.map((t) => (
        <span key={t} className={`ann-cell${t === tier ? ' lit' : ''}`}>{TIER_LABEL[t]}</span>
      ))}
    </span>
  )
}

/* Model and market on one 0-100% scale for the side the model prefers.
 * Positions only; the gap between the marks is the published edge. */
function Scale({ model, market }) {
  if (model == null || market == null) return <span className="scale scale-none">—</span>
  const lo = Math.min(model, market), hi = Math.max(model, market)
  return (
    <span className="scale" role="img" aria-label={`Model ${pct(model)}, market ${pct(market)}`}>
      <span className="scale-track">
        <span className="scale-half" />
        <span className="scale-gap" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
        <span className="scale-mkt" style={{ left: `${market * 100}%` }} />
        <span className="scale-mdl" style={{ left: `${model * 100}%` }} />
      </span>
      <span className="scale-figs"><b>{pct(model)}</b> <span>vs</span> {pct(market)}</span>
    </span>
  )
}

function StakeCell({ m, bankroll }) {
  if (!m || m.status !== 'priced') return <span className="stake-dash">—</span>
  if (m.unsized_reason || !m.stake_fraction) {
    return <span className="stake stake-none" title={m.unsized_reason || 'No stake at this price'}>Unsized</span>
  }
  return <span className="stake">{money(m.stake_fraction * bankroll)}</span>
}

/* A refusal's reason, cut to its first clause for the row; the full text
 * is in the title and in the opened row. */
function shortReason(r) {
  if (!r) return ''
  if (/integer line/.test(r)) return 'whole-number line (ADR 0022)'
  if (/no book quotes/.test(r)) return 'no complete market'
  return r.split(/[.;:(]/)[0].slice(0, 48)
}

function pickText(g, m) {
  if (!m || m.status !== 'priced') return null
  const who = m.side === 'home' ? g.home : m.side === 'away' ? g.away : m.side === 'over' ? 'Over' : 'Under'
  const ln = m.market === 'moneyline' ? 'ML' : m.market === 'total' ? `${m.line}` : fmtLine(m.line)
  return { who, ln }
}

function Context({ league, g, full }) {
  const c = g.context || {}
  const bits = []
  if (league === 'mlb') {
    if (c.away_probable || c.home_probable) {
      bits.push({ k: 'sp', t: `${c.away_probable || 'TBD'} vs ${c.home_probable || 'TBD'}`, lead: true })
    }
    ;['away', 'home'].forEach((s) => {
      if (c[`${s}_probable_changed_from`]) bits.push({ k: `scr${s}`, t: `Scratch: ${s === 'home' ? g.home : g.away} was ${c[`${s}_probable_changed_from`]}`, warn: true })
    })
  }
  if (league === 'nba') {
    ;['away', 'home'].forEach((s) => {
      const team = s === 'home' ? g.home : g.away
      if (c[`${s}_back_to_back`]) bits.push({ k: `b2b${s}`, t: `${team} back-to-back`, warn: true })
      else if (full && c[`${s}_rest_days`] != null) bits.push({ k: `r${s}`, t: `${team} ${c[`${s}_rest_days`]} days' rest` })
    })
  }
  if (league === 'nhl' && c.low_information) {
    bits.push({ k: 'low', t: full ? `Early season: ${g.away} has played ${c.away_games_played}, ${g.home} ${c.home_games_played}; ratings still near league average` : `Early season · ${c.away_games_played}/${c.home_games_played} GP`, warn: true })
  }
  if (league === 'nfl') {
    const last = (n) => (n ? n.split(' ').filter((w) => !/^(Jr\.?|Sr\.?|II|III|IV)$/.test(w)).slice(-1)[0] : '?')
    if (c.away_qb_name || c.home_qb_name) {
      bits.push({ k: 'qb', t: full ? `QB ${c.away_qb_name || '?'} / ${c.home_qb_name || '?'}` : `QB ${last(c.away_qb_name)} / ${last(c.home_qb_name)}` })
    }
    if (full && c.roof) {
      const wx = [c.roof, c.temp != null ? `${c.temp}°F` : null, c.wind != null ? `wind ${c.wind} mph` : null].filter(Boolean)
      bits.push({ k: 'roof', t: wx.join(', ') })
    }
    if (c.regime) {
      const who = [c.regime.away && `${g.away} (${c.regime.away.coach})`, c.regime.home && `${g.home} (${c.regime.home.coach})`].filter(Boolean).join(', ')
      bits.push({ k: 'reg', t: `First-year staff: ${who}` })
    }
  }
  if (!bits.length) return null
  return (
    <span className="ctx">
      {bits.map((b) => (
        <span key={b.k} className={`ctx-bit${b.warn ? ' warn' : ''}${b.lead ? ' lead' : ''}`}>
          {b.warn && <StateGlyph state="degraded" size={10} />}{b.t}
        </span>
      ))}
    </span>
  )
}

function GameRow({ g, league, board, book, isOpen, onToggle, changes, rowRef, compact, onOpenMatchup, onPlayers }) {
  const m = g.markets?.[g.headline]
  const pick = pickText(g, m)
  const bankroll = book.settings.bankroll
  const refused = g.refusal || (m && m.status === 'refused')
  return (
    <div className={`row${isOpen ? ' open' : ''}${compact ? ' compact' : ''}${changes.length ? ' changed' : ''}`}>
      <button className="row-line" onClick={onToggle} aria-expanded={isOpen} ref={rowRef}
              aria-label={`${g.away} at ${g.home}${pick ? `, pick ${pick.who} ${pick.ln}` : ''}`}>
        <span className="tick" aria-hidden="true">{changes.length ? <ChangeGlyph kind={changes[0].kind} /> : null}</span>
        <span className="start"><span>{startTime(g.start)}</span><small>{startDay(g.start)}</small></span>
        <span className="game">
          <span className="teams"><span>{g.away}</span><i>@</i><span>{g.home}</span></span>
          <Context league={league} g={g} />
        </span>
        <span className="pick">
          {pick ? (
            <>
              <b>{pick.who} {pick.ln}</b>
              <small title={`best price, at ${bookName(m.best_book)}`}>{american(m.best_price_american)}{compact ? '' : ` at ${bookName(m.best_book)}`}</small>
            </>
          ) : refused ? (
            <span className="refused-word">
              <b><StateGlyph state="refused" /> Refused</b>
              <small title={g.refusal || m?.refusal}>{shortReason(g.refusal || m?.refusal)}</small>
            </span>
          ) : (
            <span className="model-only">
              <b>{g.model?.p_home_win != null ? `${g.home} ${pct(g.model.p_home_win, 0)}` : '—'}</b>
              <small>to win · model</small>
            </span>
          )}
        </span>
        <span className="tier">
          {m?.status !== 'priced' ? <span className="ann-none">No price</span>
            : compact ? <span className="tier-word">{TIER_LABEL[m.tier]}</span>
            : <Annunciator tier={m.tier} capped={!!m.cap} />}
        </span>
        <span className="vs">
          {compact
            ? (m?.status === 'priced' ? <span className="scale-figs"><b>{pct(m.p_model)}</b> <span>vs</span> {pct(m.p_market)}</span> : null)
            : <Scale model={m?.p_model} market={m?.p_market} />}
        </span>
        <span className="edge num">{m?.status === 'priced' ? pts(m.edge) : null}</span>
        <span className="num"><StakeCell m={m} bankroll={bankroll} /></span>
        <Chevron open={isOpen} />
      </button>
      <KeyPlayerStrip items={g.context?.key_player} onPlayers={league === 'nba' ? onPlayers : undefined} />
      {isOpen && <Detail g={g} league={league} board={board} book={book} changes={changes} onOpenMatchup={onOpenMatchup} />}
    </div>
  )
}

function Detail({ g, league, board, book, changes, onOpenMatchup }) {
  const markets = Object.values(g.markets || {})
  const m = g.markets?.[g.headline]
  return (
    <div className="detail">
      {changes.length > 0 && (
        <ul className="d-changes" aria-label="Changed since your last visit">
          {changes.map((c, i) => <li key={i}><ChangeGlyph kind={c.kind} /><ChangeText c={c} /></li>)}
        </ul>
      )}
      {g.refusal && <p className="stamp stamp-inline"><StateGlyph state="refused" /><span>{g.refusal}</span></p>}
      {m?.cap && <p className="stamp stamp-inline stamp-cap"><StateGlyph state="degraded" /><span>{m.cap.reason}</span></p>}

      <div className="d-cols">
        <div className="d-model">
          <h3>The model's view</h3>
          <dl>
            <dt>{g.home} to win</dt><dd>{pct(g.model?.p_home_win)}</dd>
            <dt>Expected margin</dt><dd>{g.model?.margin_mean == null ? '—' : Math.abs(g.model.margin_mean) < 0.05 ? 'Even' : `${g.home} ${signed(g.model.margin_mean)}`}</dd>
            {g.model?.total_mean != null && <><dt>Expected total</dt><dd>{g.model.total_mean.toFixed(1)}</dd></>}
          </dl>
          <Context league={league} g={g} full />
          {(league === 'nfl' || league === 'cfb') && m?.status === 'priced' && m.market === 'spread' && (
            <KeyNumbers home={g.home} line={m.side === 'home' ? m.line : -m.line} margin={g.model?.margin_mean} />
          )}
        </div>

        {markets.length > 0 && (
          <table className="d-markets">
            <caption>Every market the feed quotes, priced by the core</caption>
            <thead>
              <tr><th>Market</th><th>Side</th><th className="num" title="Line at the first capture; market probability for a moneyline">Open</th><th className="num">Now</th>
                <th>Best</th><th className="num">Model</th><th className="num">Market</th><th className="num" title="Model probability minus the market's, in points. Not edge.">Gap</th><th>Tier</th></tr>
            </thead>
            <tbody>
              {markets.map((v) => {
                if (v.status !== 'priced') {
                  return (
                    <tr key={v.market} className="refused">
                      <th scope="row" data-label="Market">{MARKET_LABEL[v.market] || v.market}</th>
                      <td colSpan={8} data-label="Refused"><StateGlyph state="refused" /> {v.refusal}</td>
                    </tr>
                  )
                }
                const who = v.side === 'home' ? g.home : v.side === 'away' ? g.away : v.side === 'over' ? 'Over' : 'Under'
                /* A moneyline has no line; its movement is the market's price,
                 * so Open and Now show the market probability at open and now. */
                const ml = v.market === 'moneyline'
                const show = (x) => (v.market === 'total' ? x : fmtLine(x))
                const open = ml ? pct(v.open_p_market) : v.open_line != null ? show(v.open_line) : '—'
                const now = ml ? pct(v.p_market) : show(v.line)
                return (
                  <tr key={v.market}>
                    <th scope="row" data-label="Market">{MARKET_LABEL[v.market] || v.market}</th>
                    <td data-label="Side">{who}{ml ? ' ML' : ''}</td>
                    <td className="num" data-label={ml ? 'Open (market %)' : 'Open'}>{open}</td>
                    <td className="num" data-label={ml ? 'Now (market %)' : 'Now'}>{now}</td>
                    <td data-label="Best" title={`best of ${v.books} books quoting this line`}>{american(v.best_price_american)} <small>{bookName(v.best_book)}</small></td>
                    <td className="num" data-label="Model">{pct(v.p_model)}</td>
                    <td className="num" data-label="Market">{pct(v.p_market)}</td>
                    <td className="num" data-label="Gap">{pts(v.edge)}</td>
                    <td data-label="Tier">{TIER_LABEL[v.tier]}{v.cap ? ' (capped)' : ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="d-v2">
        <div className="d-v2-col">
          <h3>Line since open</h3>
          <LineSpark g={g} market={m} />
          <ContextLines g={g} />
        </div>
        <div className="d-v2-col">
          <InjuryReport g={g} />
        </div>
      </div>
      <p className="d-open"><OpenMatchup league={league} g={g} onOpen={onOpenMatchup} /></p>

      {m?.status === 'priced' ? (
        <div className="d-foot">
          <p className="d-source">
            Model probability at the consensus line; market is the median across books after
            removing the margin ({board.tiers?.provisional ? 'tiers provisional' : 'tiers calibrated'}).
            {m.unsized_reason ? ` ${m.unsized_reason}.` : ''}
          </p>
          <LogButton g={g} league={league} m={m} book={book} />
        </div>
      ) : (
        <p className="d-source d-foot">No market price has been captured for this game, so there is
          nothing to compare the model with and nothing to size.</p>
      )}
    </div>
  )
}

function ChangeText({ c }) {
  if (c.kind === 'line') return <span>{c.text} moved {fmtLine(c.from)} → {fmtLine(c.to)}</span>
  if (c.kind === 'price') return <span>{c.text} moved {pct(c.from)} → {pct(c.to)}</span>
  if (c.kind === 'tier') return <span>{c.text}: {c.from ? TIER_LABEL[c.from] : 'no price'} → {TIER_LABEL[c.to]}</span>
  return <span>{c.text}</span>
}

/* NFL spreads land on 3 and 7 far more than on their neighbours. Shows where
 * the market's number and the model's margin sit against them -- positions of
 * two published values, nothing derived. */
function KeyNumbers({ home, line, margin }) {
  if (line == null || margin == null) return null
  const span = 14
  const pos = (v) => `${Math.max(0, Math.min(100, ((v + span) / (2 * span)) * 100))}%`
  const marketMargin = -line
  return (
    <figure className="keys" aria-label={`Key numbers: market needs ${home} by ${signed(marketMargin)}, model expects ${signed(margin)}`}>
      <div className="keys-track">
        {[-7, -3, 3, 7].map((k) => <span key={k} className="keys-key" style={{ left: pos(k) }}><i>{Math.abs(k)}</i></span>)}
        <span className="keys-zero" style={{ left: pos(0) }} />
        <span className="keys-mkt" style={{ left: pos(marketMargin) }} title="market" />
        <span className="keys-mdl" style={{ left: pos(margin) }} title="model" />
      </div>
      <figcaption><span className="lg-mkt" /> market {signed(marketMargin)} <span className="lg-mdl" /> model {signed(margin)} · {home} margin</figcaption>
    </figure>
  )
}

function weekStart(ts) {
  const d = new Date(ts); const day = (d.getDay() + 6) % 7
  d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - day); return d.getTime()
}

function LogButton({ g, league, m, book }) {
  const { settings, betLog, logBet } = book
  const unit = (settings.bankroll * settings.unitPct) / 100 || 1
  const units = m.stake_fraction ? Math.round((m.stake_fraction * settings.bankroll / unit) * 100) / 100 : 0
  const wk = weekStart(Date.now())
  const exposed = betLog.filter((b) => b.ts >= wk).reduce((s, b) => s + (b.stakeUnits || 0), 0)
  const blocked = exposed + units > settings.weeklyCapUnits
  const [done, setDone] = useState(false)
  const who = m.side === 'home' ? g.home : m.side === 'away' ? g.away : m.side
  return (
    <span className="log">
      <button className="btn" disabled={blocked || done}
              onClick={() => {
                logBet({ label: `${league.toUpperCase()} ${who} ${m.market === 'moneyline' ? 'ML' : fmtLine(m.line)} (${g.away} @ ${g.home})`,
                         league, gameId: g.game_id, market: m.market, line: m.line, price: m.best_price_american,
                         book: m.best_book, stakeUnits: units, tier: m.tier, unsized: !m.stake_fraction })
                setDone(true)
              }}>
        {done ? 'Logged' : 'Log this pick'}
      </button>
      <small>{blocked ? `Weekly cap reached (${exposed}/${settings.weeklyCapUnits} units)` : units ? `${units} units from the core's stake` : 'Logged unsized: the core sized nothing'}</small>
    </span>
  )
}

function ChangeLog({ since, board }) {
  const { baseline, changes } = since
  return (
    <aside className="rail" aria-label="Since your last visit">
      <h2>Since your last visit</h2>
      {!baseline && <p className="rail-note">First visit on this browser: nothing to compare yet. Changes from here on are listed on your next visit.</p>}
      {baseline && changes.length === 0 && (
        <p className="rail-note">Nothing moved since {baseline.seen_at ? new Date(baseline.seen_at).toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' }) : 'your last visit'}.</p>
      )}
      {changes.length > 0 && (
        <ol className="log-lines">
          {changes.map((c, i) => (
            <li key={i} className={`ll ll-${c.kind}`}><ChangeGlyph kind={c.kind} /><ChangeText c={c} /></li>
          ))}
        </ol>
      )}
      <dl className="rail-facts">
        <dt>Board exported</dt><dd>{board.generated_at ? new Date(board.generated_at).toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' }) : '—'}</dd>
        <dt>Last odds capture</dt><dd>{board.freshness?.odds_captured_at ? new Date(board.freshness.odds_captured_at).toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' }) : 'none yet'}</dd>
      </dl>
    </aside>
  )
}
