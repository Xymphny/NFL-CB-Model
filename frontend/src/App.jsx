import { useEffect, useState, useMemo } from 'react'
import { coverProb, sizeStake, confidenceDrivers, confidenceScore, altLineFairPrices, PLAY_GAP, LEAN_GAP, DEFAULT_PRICE } from './staking'
import { useAccount } from './account'
import { useBook } from './store'
import MyBook, { AccountChip } from './MyBook'

/* ---------------- Data hooks (unchanged snapshot architecture) ---------------- */

function useLatestSnapshot(kind) {
  const [state, setState] = useState({ data: null, loading: true, error: null })

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => { if (!res.ok) throw new Error('no manifest'); return res.json() })
      .then((manifest) => {
        const files = manifest[kind] || []
        if (files.length === 0) {
          setState({ data: null, loading: false, error: new Error('no snapshots yet') })
          return
        }
        const latest = files[files.length - 1]
        return fetch(`/data/${kind}/${latest}`)
          .then((res) => res.json())
          .then((data) => setState({ data, loading: false, error: null }))
      })
      .catch((error) => setState({ data: null, loading: false, error }))
  }, [kind])

  return state
}

function useRatingsHistory(kind = 'ratings') {
  const [state, setState] = useState({ history: null, loading: true, error: null })

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => { if (!res.ok) throw new Error('no manifest'); return res.json() })
      .then((manifest) => {
        const files = manifest[kind] || []
        if (files.length === 0) {
          setState({ history: null, loading: false, error: new Error('no snapshots yet') })
          return
        }
        return Promise.all(files.map((f) => fetch(`/data/${kind}/${f}`).then((r) => r.json()))).then(
          (snapshots) => {
            const latestSeason = Math.max(...snapshots.map((s) => s.season))
            const history = {}
            snapshots
              .filter((snap) => snap.season === latestSeason)
              .sort((a, b) => a.week - b.week)
              .forEach((snap) => {
                snap.ratings.forEach((team) => {
                  if (!history[team.team]) history[team.team] = []
                  history[team.team].push({ week: snap.week, total_rating: team.total_rating })
                })
              })
            setState({ history, loading: false, error: null })
          }
        )
      })
      .catch((error) => setState({ history: null, loading: false, error }))
  }, [kind])

  return state
}

/* ---------------- Formatting ---------------- */

function formatSigned(value, digits = 1) {
  if (value === null || value === undefined) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined) return '—'
  return value.toFixed(digits)
}

/* ---------------- Edge grading ----------------
 * Thresholds and probability math live in staking.js. Verdicts stay
 * deliberately conservative: sub-4-point gaps did not clear the 52.4%
 * ATS breakeven in the walk-forward backtest.
 */

function gradeGame(d, playGap = PLAY_GAP, leanGap = LEAN_GAP) {
  const g = gradeGameInner(d, playGap, leanGap)
  if (g.verdict === 'play' && g.market === 'spread' && d.tier_cap === 'lean') {
    return { verdict: 'lean', market: 'spread', stake: '0.5u', capped: 'regime' }
  }
  return g
}

function gradeGameInner(d, playGap = PLAY_GAP, leanGap = LEAN_GAP) {
  const spreadEdge = Math.abs(d.spread_gap)
  const totalEdge = d.total_gap != null ? Math.abs(d.total_gap) : 0

  if (spreadEdge >= playGap || totalEdge >= playGap + 1) {
    const isSpread = spreadEdge >= playGap
    return { verdict: 'play', market: isSpread ? 'spread' : 'total', stake: '1u' }
  }
  if (spreadEdge >= leanGap || totalEdge >= leanGap + 1) {
    const isSpread = spreadEdge >= leanGap
    return { verdict: 'lean', market: isSpread ? 'spread' : 'total', stake: '0.5u' }
  }
  return { verdict: 'pass', market: null, stake: null }
}

function describePick(d, market) {
  if (market === 'spread') {
    const modelLikesHome = d.spread_gap > 0
    const side = modelLikesHome ? d.home_team : d.away_team
    const line = modelLikesHome ? -d.market_spread : d.market_spread
    return `${side} ${formatSigned(line, 1)}`
  }
  const over = d.total_gap > 0
  return `${d.away_team}/${d.home_team} ${over ? 'over' : 'under'} ${d.market_total.toFixed(1)}`
}

function describeReason(d, market) {
  const parts = []
  if (market === 'spread') {
    const modelSpread = d.market_spread + d.spread_gap
    parts.push(`Model makes it ${formatSigned(modelSpread, 1)} vs the market's ${formatSigned(d.market_spread, 1)}`)
  } else {
    const modelTotal = d.market_total + d.total_gap
    parts.push(`Model projects ${modelTotal.toFixed(1)} vs the market's ${d.market_total.toFixed(1)}`)
  }
  if (d.moved_toward_model === true) parts.push('line has moved toward the model since open')
  if (d.moved_toward_model === false) parts.push('line has moved away from the model since open')
  return parts.join(' — ')
}

function describePass(d) {
  const spreadEdge = Math.abs(d.spread_gap)
  const totalEdge = Math.abs(d.total_gap)
  const best = Math.max(spreadEdge, totalEdge)
  if (best < LEAN_GAP) return 'Model and market agree'
  return 'Edge below play threshold'
}

/* ---------------- This week: edge board ---------------- */

function ConfidenceMeter({ drivers }) {
  const score = confidenceScore(drivers)
  return (
    <div className="conf">
      <div className="conf-head">
        <span className="bet-stat-label">Confidence</span>
        <span className="conf-label">{score.label} — {score.filled} of {score.total}</span>
      </div>
      <div className="conf-bar">
        {drivers.map((dr, i) => (
          <div key={i} className={`conf-seg ${dr.ok === true ? 'on' : dr.ok === false ? 'bad' : ''}`} />
        ))}
      </div>
      <div className="conf-chips">
        {drivers.map((dr) => (
          <span key={dr.key} className={`conf-chip ${dr.ok === true ? 'on' : dr.ok === false ? 'bad' : ''}`}>
            {dr.label}
          </span>
        ))}
      </div>
    </div>
  )
}

function formatKickoff(iso) {
  if (!iso) return null
  const dt = new Date(iso)
  if (isNaN(dt)) return null
  return dt.toLocaleString('en-US', {
    weekday: 'short', month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit', timeZone: 'America/New_York',
  }) + ' ET'
}

function weatherBrief(wx) {
  if (!wx) return null
  if (wx.roof === 'dome' || wx.roof === 'closed') return 'Indoors'
  if (wx.temp_f != null) return Math.round(wx.temp_f) + '\u00B0F \u00B7 ' + Math.round(wx.wind_mph) + ' mph wind'
  return null
}

function GameContext({ d, qb1Map }) {
  const [open, setOpen] = useState(false)
  const inj = d.injuries
  const wx = d.weather
  const nInj = inj ? (inj.home?.length || 0) + (inj.away?.length || 0) : 0
  const qbHome = qb1Map ? qb1Map[d.home_team] : null
  const qbAway = qb1Map ? qb1Map[d.away_team] : null
  if (!inj && !wx && !qbHome && !qbAway) return null

  const wxLine = wx
    ? wx.roof === 'dome' || wx.roof === 'closed'
      ? 'Indoors'
      : wx.temp_f != null
      ? `${Math.round(wx.temp_f)}°F · wind ${Math.round(wx.wind_mph)} mph${wx.precip_prob != null ? ` · ${wx.precip_prob}% precip` : ''}${wx.wind_mph >= 15 ? ' — wind worth watching on the total' : ''}`
      : 'Outdoors — forecast pending'
    : null

  return (
    <div className="game-context">
      <button className="alt-lines-toggle" onClick={() => setOpen(!open)}>
        {open ? 'Hide game context' : `Game context${nInj ? ` — ${nInj} on injury report` : ''}${wxLine && !open ? ` · ${wxLine.split(' — ')[0]}` : ''}`}
      </button>
      {open && (
        <div className="context-body">
          {(qbHome || qbAway) && (
            <p className="context-qbs">
              QBs: {d.away_team} {qbAway || '—'} · {d.home_team} {qbHome || '—'}
            </p>
          )}
          {wxLine && <p className="context-weather">{wxLine}</p>}
          {inj && (
            <div className="context-injuries">
              {[['home', d.home_team], ['away', d.away_team]].map(([side, team]) => (
                <div key={side}>
                  <p className="context-team">{team}</p>
                  {inj[side] == null && <p className="context-none">No report available</p>}
                  {Array.isArray(inj[side]) && inj[side].length === 0 && <p className="context-none">No one listed</p>}
                  {(inj[side] || []).map((p) => (
                    <p className="context-inj-row" key={p.player}>
                      <span className={`inj-status ${p.status.toLowerCase()}`}>{p.status === 'Questionable' ? 'Q' : p.status === 'Doubtful' ? 'D' : 'OUT'}</span>
                      {p.player} <span className="context-pos">{p.position}</span>
                    </p>
                  ))}
                </div>
              ))}
            </div>
          )}
          <p className="alt-lines-note">
            Reports and forecasts are context the market already prices — use them to understand the
            number, not as extra edge on top of it.
          </p>
        </div>
      )}
    </div>
  )
}

function AltLines({ d, marginDist }) {
  const [open, setOpen] = useState(false)
  const pmf = marginDist.residual_distribution?.residual_pmf
  if (!pmf) return null
  const modelMargin = d.market_spread + d.spread_gap
  const rows = open ? altLineFairPrices(modelMargin, d.market_spread, pmf, 1.5) : []
  return (
    <div className="alt-lines">
      <button className="alt-lines-toggle" onClick={() => setOpen(!open)}>
        {open ? 'Hide alt lines' : 'Alt lines — our fair prices'}
      </button>
      {open && (
        <div className="alt-lines-grid">
          {rows.map((r) => (
            <div className="alt-line-cell" key={r.line}>
              <span className="alt-line-num">{formatSigned(-r.line, 1)}</span>
              <span className="alt-line-fair">{r.fair > 0 ? `+${r.fair}` : r.fair}</span>
            </div>
          ))}
        </div>
      )}
      {open && (
        <p className="alt-lines-note">
          Home side at each half point, our fair (no-vig) price from 4,078 games of real margin
          distribution. Beat these numbers at a book and the rung is +EV by our count.
        </p>
      )}
    </div>
  )
}

function MlbBoard({ snap }) {
  if (!snap) return <p className="section-sub">The MLB board populates once the first odds snapshot lands (push + sync + first cron run).</p>
  const fmtMl = (x) => (x == null ? '—' : x > 0 ? `+${x}` : `${x}`)
  return (
    <section>
      <p className="preseason-note">{snap.note}</p>
      {snap.divergences.map((d) => (
        <div key={`${d.away_team}@${d.home_team}${d.kickoff || ''}`} className="bet-card">
          <div className="bet-card-top">
            <div className="matchup-block">
              <p className="matchup-line">{d.away_name || d.away_team} <span className="matchup-at">@</span> {d.home_name || d.home_team}</p>
              <p className="kickoff-line">
                {d.line_status === 'closed' && <span className="closed-chip">Closed</span>}
                {[formatKickoff(d.kickoff),
                  (d.away_probable || d.home_probable) ? `${d.away_probable || 'TBD'} vs ${d.home_probable || 'TBD'}` : null,
                  d.market_total != null ? `O/U ${d.market_total}` : null].filter(Boolean).join(' · ')}
              </p>
            </div>
            <span className="verdict lean">Watch</span>
          </div>
          <div className="card-chips">
            <span className="card-chip">Best: {d.away_team} {fmtMl(d.away_ml)} ({d.away_ml_book}) · {d.home_team} {fmtMl(d.home_ml)} ({d.home_ml_book})</span>
            <span className="card-chip">Market {Math.round(d.market_home_prob * 100)}% home{d.model_home_prob != null ? ` · Model ${Math.round(d.model_home_prob * 100)}%` : ''}{d.ev_gap != null ? ` · EV gap ${(d.ev_gap * 100).toFixed(1)}%` : ''}</span>
          </div>
        </div>
      ))}
    </section>
  )
}

function EdgeBoard({ divergences, note, season, week, book, ratingsByTeam, perf, marginDist, playGap = PLAY_GAP, leanGap = LEAN_GAP, edgeCoefOverride = null, qb1Map = null, boardLeague = 'NFL', boardCfbLogos = null, boardScores = null, boardOpenLines = null, boardSiteTeams = null }) {
  const [showPassed, setShowPassed] = useState(false)
  const { settings, logBet, betLog } = book

  const graded = useMemo(
    () => divergences.map((d) => ({ ...d, grade: gradeGame(d, playGap, leanGap) })),
    [divergences, playGap, leanGap]
  )
  const plays = graded.filter((g) => g.grade.verdict === 'play')
  const leans = graded.filter((g) => g.grade.verdict === 'lean')
  const passed = graded.filter((g) => g.grade.verdict === 'pass')
  const actionable = [...plays, ...leans]

  const unitDollars = (settings.bankroll * settings.unitPct) / 100 || 1
  const weekAgo = Date.now() - 7 * 24 * 3600 * 1000
  const exposedUnits = betLog
    .filter((b) => b.ts > weekAgo && b.result == null)
    .reduce((s, b) => s + (b.stakeUnits || 0), 0)

  return (
    <div>
      <div className="board-summary">
        <span className="board-count">{actionable.length}</span>
        <span className="board-count-label">
          {actionable.length === 1 ? 'edge' : 'edges'} this week · {passed.length} games passed
        </span>
      </div>

      {note && <p className="section-sub">{note}</p>}

      {exposedUnits >= settings.weeklyCapUnits && (
        <div className="empty-state cap-warning">
          <strong>Weekly cap reached</strong>
          You have {exposedUnits.toFixed(1)} units open against a {settings.weeklyCapUnits}-unit cap.
          The best bet available is not betting past your limits.
        </div>
      )}

      {actionable.length === 0 && (
        <div className="empty-state">
          <strong>No plays this week</strong>
          The model and the market are in agreement across the board. Passing is a position — forcing
          bets without an edge is how bankrolls die.
        </div>
      )}

      {actionable.map((d) => {
        const { verdict, market } = d.grade
        const gap = market === 'spread' ? d.spread_gap : d.total_gap
        const edgeCoef = edgeCoefOverride ?? (marginDist ? marginDist.edge_calibration?.edge_coef : null)
        const prob = coverProb(gap, edgeCoef)
        const stake = sizeStake({ prob, price: DEFAULT_PRICE, settings })
        const drivers = confidenceDrivers(d, market, ratingsByTeam, perf ? perf.tier_stats : null)
        const pick = describePick(d, market)
        const capBlocked = exposedUnits + stake.units > settings.weeklyCapUnits
        const pickLine = market === 'spread'
          ? (d.spread_gap > 0 ? -d.market_spread : d.market_spread)
          : d.market_total
        return (
          <div className={`bet-card ${verdict}`} key={`${d.away_team}-${d.home_team}`}>
            <div className="bet-card-top">
              <div className="matchup-block">
                <p className="matchup-line">
                  <TeamMark league={boardLeague} team={d.away_team} cfbLogos={boardCfbLogos} />
                  {d.away_name || d.away_team} <span className="matchup-at">@</span>{' '}
                  <TeamMark league={boardLeague} team={d.home_team} cfbLogos={boardCfbLogos} />
                  {d.home_name || d.home_team}
                </p>
                {(() => {
                  const ls = boardScores && boardScores[`${d.away_team}@${d.home_team}`]
                  if (!ls || ls.state === 'pre') return null
                  return (
                    <p className={`live-score ${ls.state === 'in' ? 'live' : ''}`}>
                      {ls.state === 'in' ? 'LIVE' : 'FINAL'} — {d.away_team} {ls.as}, {d.home_team} {ls.hs}
                      {ls.state === 'in' && ls.detail ? ` · ${ls.detail}` : ''}
                    </p>
                  )
                })()}
                {(formatKickoff(d.kickoff) || weatherBrief(d.weather) || d.line_status === 'closed') && (
                  <p className="kickoff-line">
                    {d.line_status === 'closed' && <span className="closed-chip">Closed</span>}
                    {[formatKickoff(d.kickoff), weatherBrief(d.weather)].filter(Boolean).join(' · ')}
                    {d.line_status === 'closed' && ' — game started, line frozen at close'}
                  </p>
                )}
              </div>
              <span className={`verdict ${verdict}`}>{verdict === 'play' ? 'Play' : 'Lean'}</span>
            </div>
            <p className="bet-pick">{pick}</p>
            {(() => {
              const chips = []
              // Rest/bye chips (NFL): from the team schedules already on the site.
              if (boardSiteTeams && boardSiteTeams.teams && week != null) {
                for (const side of ['away_team', 'home_team']) {
                  const t = boardSiteTeams.teams[d[side]]
                  if (!t) continue
                  const prev = t.schedule.find((g2) => g2.week === week - 1)
                  if (week > 1 && !prev) chips.push(`${d[side]} off bye`)
                  else if (d.kickoff && new Date(d.kickoff).getUTCDay() === 5 && prev) chips.push(`${d[side]} short week`)
                }
              }
              // Wind flag on total picks: annotation, never adjustment.
              const isTotalPick = /over|under/i.test(String(pick))
              const wx = d.weather
              if (isTotalPick && wx && wx.roof !== 'dome' && wx.roof !== 'closed' && wx.wind_mph >= 15) {
                chips.push(`${Math.round(wx.wind_mph)} mph wind — not modeled; unders historically aided`)
              }
              // Regime-change context (advisory; cap evidence in tier_cap_reason).
              if (d.regime) {
                const pickedHome = d.spread_gap > 0
                for (const side of ['away', 'home']) {
                  const r = d.regime[side]
                  if (!r) continue
                  const team = side === 'home' ? d.home_team : d.away_team
                  const backed = (side === 'home') === pickedHome
                  chips.push(`${team} new regime (${r.coach}${r.tier >= 2 ? ' — HC + staff' : ''})${backed && d.tier_cap ? ': capped at Lean — backed-regime early flags went 0/9 in backtests' : ': early-season prior less reliable'}`)
                }
              }
              // Line movement vs the week's opener (spread picks only).
              const open_ = boardOpenLines ? boardOpenLines[`${d.away_team}@${d.home_team}`] : null
              if (!isTotalPick && open_ != null && d.market_spread != null && Math.abs(d.market_spread - open_) >= 0.5) {
                const pickedHome = d.spread_gap > 0
                const mv = d.market_spread - open_
                const withUs = pickedHome ? mv > 0 : mv < 0
                const f = (x) => (x > 0 ? `home -${Math.abs(x).toFixed(1)}` : `home +${Math.abs(x).toFixed(1)}`)
                chips.push(`Line: opened ${f(open_)} → now ${f(d.market_spread)} (${withUs ? 'moving our way' : 'moving against us'})`)
              }
              if (!chips.length) return null
              return <div className="card-chips">{chips.map((c) => <span key={c} className={`card-chip ${c.includes('wind') ? 'warn' : ''} ${c.includes('against') ? 'warn' : ''}`}>{c}</span>)}</div>
            })()}

            <ConfidenceMeter drivers={drivers} />

            <div className="bet-stats">
              <div>
                <span className="bet-stat-label">Est. cover</span>
                <span className="bet-stat-value">{formatPercent(prob)}</span>
              </div>
              <div>
                <span className="bet-stat-label">Edge</span>
                <span className="bet-stat-value">{Math.abs(gap).toFixed(1)} pts</span>
              </div>
              <div>
                <span className="bet-stat-label">Stake</span>
                <span className="bet-stat-value">
                  {stake.units > 0 ? `$${stake.dollars.toLocaleString()}` : '—'}
                  {stake.units > 0 && <span className="stake-units"> {stake.units}u</span>}
                </span>
              </div>
            </div>

            <p className="bet-reason">
              {describeReason(d, market)}
              {d.fpi_home_prob != null && d.market_win_prob_home_fair != null && (
                <span className="fpi-ref">
                  {' '}· FPI {Math.round(d.fpi_home_prob * 100)}% / market {Math.round(d.market_win_prob_home_fair * 100)}% home
                </span>
              )}
            </p>
            {d.best_prices && (() => {
              const bp = d.best_prices
              const pickBest = market === 'spread'
                ? (d.spread_gap > 0 ? bp.home_spread : bp.away_spread)
                : (d.total_gap > 0 ? bp.over : bp.under)
              if (!pickBest || pickBest.price == null) return null
              const shownPoint = market === 'spread'
                ? formatSigned(d.spread_gap > 0 ? -pickBest.point : pickBest.point, 1)
                : `${d.total_gap > 0 ? 'over' : 'under'} ${pickBest.point.toFixed(1)}`
              return (
                <p className="best-price-line">
                  Best price: {shownPoint} at {pickBest.price > 0 ? `+${pickBest.price}` : pickBest.price}
                  {' '}({pickBest.book}, {bp.n_books} books checked)
                </p>
              )
            })()}
            {d.sharp_anchor && d.sharp_anchor.stale_books && d.sharp_anchor.stale_books.length > 0 && (
              <p className="sharp-line">
                {d.sharp_anchor.book} has {formatSigned(-d.sharp_anchor.spread, 1)} —{' '}
                {d.sharp_anchor.stale_books
                  .slice(0, 2)
                  .map((s) => `${s.book} stale at ${formatSigned(-s.point, 1)} (${s.value_side === 'home' ? d.home_team : d.away_team} value)`)
                  .join('; ')}
              </p>
            )}
            <GameContext d={d} qb1Map={qb1Map} />
            {market === 'spread' && marginDist && (
              <AltLines d={d} marginDist={marginDist} />
            )}
            {settings.mode !== 'flat' && stake.units > 0 && (
              <p className="kelly-line">
                Kelly at −110: full {formatPercent(stake.fullKelly)} → applied {formatPercent(stake.applied)} of bankroll, capped at 2u
              </p>
            )}

            <button
              className="log-bet-btn"
              disabled={capBlocked}
              onClick={() =>
                logBet({
                  label: pick,
                  market,
                  ou: market === 'total' ? (d.total_gap > 0 ? 'over' : 'under') : null,
                  line: pickLine,
                  price: DEFAULT_PRICE,
                  stakeUnits: stake.units || 1,
                  stakeDollars: stake.dollars || Math.round(unitDollars),
                  season,
                  week,
                })
              }
            >
              {capBlocked
                ? 'Over weekly cap'
                : stake.units > 0
                ? `Log bet · ${stake.units}u at −110`
                : 'Edge doesn\u2019t clear the vig — log 1u anyway'}
            </button>
          </div>
        )
      })}

      {passed.length > 0 && (
        <>
          <button className="toggle-passed" onClick={() => setShowPassed(!showPassed)}>
            {showPassed ? 'Hide' : 'Show'} {passed.length} passed games
          </button>
          {showPassed && (
            <div className="passed-list">
              {passed.map((d) => (
                <div className="passed-row" key={`${d.away_team}-${d.home_team}`}>
                  <span className="passed-game">
                    {d.away_team} at {d.home_team} · gap {Math.max(Math.abs(d.spread_gap), Math.abs(d.total_gap)).toFixed(1)} pts
                  </span>
                  <span className="passed-why">{describePass(d)}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

/* ---------------- Track record ---------------- */

function useMarginDist() {
  const [dist, setDist] = useState(null)
  useEffect(() => {
    fetch('/data/margin_dist.json')
      .then((res) => { if (!res.ok) throw new Error('none'); return res.json() })
      .then(setDist)
      .catch(() => {})
  }, [])
  return dist
}

function usePerformance(league) {
  const [state, setState] = useState({ data: null, loading: true })

  useEffect(() => {
    setState({ data: null, loading: true })
    const file = league === 'CFB' ? '/data/cfb_performance.json' : league === 'MLB' ? '/data/mlb_performance.json' : '/data/performance.json'
    fetch(file)
      .then((res) => { if (!res.ok) throw new Error('none'); return res.json() })
      .then((data) => setState({ data, loading: false }))
      .catch(() => setState({ data: null, loading: false }))
  }, [league])

  return state
}

function KpiStrip({ perf }) {
  const atsPct = perf && perf.ats_wins + perf.ats_losses > 0
    ? perf.ats_wins / (perf.ats_wins + perf.ats_losses)
    : null

  const kpis = [
    {
      label: 'Flagged plays ATS',
      value: perf ? `${perf.ats_wins}–${perf.ats_losses}` : '—',
      note: atsPct !== null ? `${formatPercent(atsPct)} · breakeven 52.4%` : 'Tracking starts week 1',
      tone: atsPct === null ? '' : atsPct >= 0.524 ? 'up' : 'down',
    },
    {
      label: 'Avg CLV per play',
      value: perf && perf.avg_clv != null ? `${formatSigned(perf.avg_clv, 1)} pts` : '—',
      note: perf ? `${perf.n_clv_bets ?? 0} plays measured vs close` : 'Needs live line snapshots',
      tone: perf && perf.avg_clv > 0 ? 'up' : perf && perf.avg_clv < 0 ? 'down' : '',
    },
    {
      label: 'Units (flat stakes)',
      value: perf && perf.units != null ? formatSigned(perf.units, 1) : '—',
      note: perf && perf.roi != null ? `ROI ${formatSigned(perf.roi * 100, 1)}%` : 'Graded after each week',
      tone: perf && perf.units > 0 ? 'up' : perf && perf.units < 0 ? 'down' : '',
    },
    {
      label: 'Model vs market error',
      value: perf && perf.model_mae != null ? `${perf.model_mae.toFixed(1)} / ${perf.market_mae.toFixed(1)}` : '—',
      note: 'Mean abs. error, points',
      tone: '',
    },
  ]

  return (
    <div className="kpi-grid">
      {kpis.map((k) => (
        <div className="kpi" key={k.label}>
          <span className="kpi-label">{k.label}</span>
          <div className={`kpi-value ${k.tone}`}>{k.value}</div>
          <span className="kpi-note">{k.note}</span>
        </div>
      ))}
    </div>
  )
}

function useClvReport(league) {
  const [state, setState] = useState({ games: null, loading: true, error: null })

  useEffect(() => {
    setState({ games: null, loading: true, error: null })
    const key = league === 'CFB' ? 'cfb_divergence' : league === 'MLB' ? 'mlb_divergence' : 'divergence'
    fetch('/data/manifest.json')
      .then((res) => { if (!res.ok) throw new Error('no manifest'); return res.json() })
      .then((manifest) => {
        const files = manifest[key] || []
        if (files.length < 2) {
          setState({ games: null, loading: false, error: new Error('need snapshots') })
          return
        }
        return Promise.all(files.map((f) => fetch(`/data/${league === 'CFB' ? 'cfb_divergence' : league === 'MLB' ? 'mlb_divergence' : 'divergence'}/${f}`).then((r) => r.json()))).then(
          (snapshots) => {
            snapshots.sort((a, b) => new Date(a.computed_at) - new Date(b.computed_at))
            const gameKeys = new Set()
            snapshots.forEach((snap) => (snap.divergences || []).forEach((d) => gameKeys.add(`${d.away_team}@${d.home_team}`)))

            const games = []
            gameKeys.forEach((key) => {
              const [away, home] = key.split('@')
              const appearances = snapshots
                .map((snap) => (snap.divergences || []).find((d) => d.away_team === away && d.home_team === home))
                .filter(Boolean)
              if (appearances.length < 2) return

              const earliest = appearances[0]
              const latest = appearances[appearances.length - 1]
              const modelSpread = earliest.market_spread + earliest.spread_gap
              const divergenceDirection = modelSpread - earliest.market_spread
              const marketMovement = latest.market_spread - earliest.market_spread
              const clvScore = divergenceDirection === 0 ? 0 : marketMovement * (divergenceDirection > 0 ? 1 : -1)

              games.push({
                away, home,
                openingMarketSpread: earliest.market_spread,
                closingMarketSpread: latest.market_spread,
                validated: clvScore > 0,
              })
            })
            setState({ games, loading: false, error: null })
          }
        )
      })
      .catch((error) => setState({ games: null, loading: false, error }))
  }, [league])

  return state
}

function TrackRecord({ league }) {
  const perf = usePerformance(league)
  const clv = useClvReport(league)

  return (
    <div>
      <section>
        <h2 className="section-heading">Season scorecard — {league}</h2>
        <p className="section-sub">
          Every flagged play is graded against the closing line and the final score, wins and losses
          alike. If the numbers here go red, you'll see it before we do anything about it.
          {league === 'CFB' && ' CFB and NFL records are tracked separately; the CFB record begins with its first graded week.'}
        </p>
        {perf.loading ? <p className="section-sub">Loading…</p> : <KpiStrip perf={perf.data} />}
        {perf.data && perf.data.plays && perf.data.plays.length > 0 && (
          <div className="graded-plays">
            {perf.data.plays.slice(0, 25).map((p, i) => (
              <div className="log-row" key={i}>
                <div className="log-main">
                  <span className="log-pick">{p.label}</span>
                  <span className="log-detail">
                    Week {p.week} · {p.tier} · edge {p.edge} pts
                    {p.clv != null && ` · CLV ${p.clv > 0 ? '+' : ''}${p.clv}`}
                  </span>
                </div>
                <span className={`result-badge ${p.result}`}>
                  {p.result === 'win' ? 'W' : p.result === 'loss' ? 'L' : 'P'}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="section-heading">Line movement on our plays</h2>
        <p className="section-sub">
          When the market moves toward the model's number after we flag a game, the model saw
          something real. Movement away means the edge was likely noise.
        </p>
        {clv.loading && <p className="section-sub">Loading…</p>}
        {(clv.error || !clv.games || clv.games.length === 0) && !clv.loading && (
          <div className="empty-state">
            <strong>No line movement measured yet</strong>
            This needs at least two odds checks on the same game — it fills in automatically during
            game weeks.
          </div>
        )}
        {clv.games && clv.games.map((g) => (
          <div className="clv-row" key={`${g.away}@${g.home}`}>
            <span className="clv-matchup">
              {g.away} <span className="clv-at">at</span> {g.home}
            </span>
            <span className="clv-detail">
              Open {formatSigned(g.openingMarketSpread, 1)} → Close {formatSigned(g.closingMarketSpread, 1)}
            </span>
            <span className={`clv-badge ${g.validated ? 'clv-toward' : 'clv-away'}`}>
              {g.validated ? 'Moved toward model' : 'Moved away'}
            </span>
          </div>
        ))}
      </section>
    </div>
  )
}

/* ---------------- Ratings + team research ---------------- */


function DivisionStandings({ siteTeams, ratingsByTeam, onSelectTeam, allRatings }) {
  const teams = siteTeams.teams
  const byDiv = {}
  Object.values(teams).forEach((t) => { (byDiv[t.division] = byDiv[t.division] || []).push(t) })
  const order = (a, b) => (b.record.w - b.record.l) - (a.record.w - a.record.l) || (b.record.pf - b.record.pa) - (a.record.pf - a.record.pa)
  return (
    <div className="divisions-grid">
      {NFL_DIVISION_ORDER.map((div) => (
        <div key={div} className="division-block">
          <h3 className="division-heading">{div}</h3>
          <table className="ratings-table division-table">
            <thead><tr><th>Team</th><th className="numeric">W-L</th><th className="numeric">PF</th><th className="numeric">PA</th><th className="numeric">Rating</th><th className="numeric">Rem SOS</th></tr></thead>
            <tbody>
              {(byDiv[div] || []).sort(order).map((t) => {
                const r = allRatings ? allRatings.find((x) => x.team === t.abbr) : null
                return (
                  <tr key={t.abbr} className="clickable-row" onClick={() => r && onSelectTeam(r)}>
                    <td className="team-cell"><TeamMark league="NFL" team={t.abbr} /> {t.nickname}</td>
                    <td className="numeric">{t.record.w}-{t.record.l}{t.record.t ? `-${t.record.t}` : ''}</td>
                    <td className="numeric">{t.record.pf}</td>
                    <td className="numeric">{t.record.pa}</td>
                    <td className="numeric">{r ? formatSigned(r.total_rating * 100, 1) : '—'}</td>
                    <td className="numeric">{t.remaining_sos != null ? formatSigned(t.remaining_sos * 100, 1) : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

function ScheduleTab({ siteTeams }) {
  const [week, setWeek] = useState(null)
  if (!siteTeams.data) return <p className="section-sub">{siteTeams.loading ? 'Loading…' : 'Schedule publishes with the weekly data refresh.'}</p>
  const teams = siteTeams.data.teams
  const weeks = {}
  Object.values(teams).forEach((t) => t.schedule.forEach((g) => {
    const key = g.home ? `${g.opp}@${t.abbr}` : `${t.abbr}@${g.opp}`
    ;(weeks[g.week] = weeks[g.week] || {})[key] = { ...g, away: key.split('@')[0], home_t: key.split('@')[1] }
  }))
  const weekNums = Object.keys(weeks).map(Number).sort((a, b) => a - b)
  const current = week || weekNums.find((w) => Object.values(weeks[w]).some((g) => !g.result)) || weekNums[0]
  return (
    <section>
      <h2 className="section-heading">Season schedule — NFL</h2>
      <div className="week-chips">
        {weekNums.map((w) => (
          <button key={w} className={`week-chip ${w === current ? 'active' : ''}`} onClick={() => setWeek(w)}>W{w}</button>
        ))}
      </div>
      <div className="schedule-list">
        {Object.entries(weeks[current]).sort((a, b) => (a[1].kickoff || '').localeCompare(b[1].kickoff || '')).map(([key, g]) => (
          <div key={key} className="schedule-row">
            <span className="sched-teams">
              <TeamMark league="NFL" team={g.away} size={18} /> {g.away}
              <span className="matchup-at"> @ </span>
              <TeamMark league="NFL" team={g.home_t} size={18} /> {g.home_t}
            </span>
            <span className="sched-when">{g.result ? g.result : (g.kickoff ? new Date(g.kickoff).toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' }) : 'TBD')}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function PlayersTab({ playerLeaders, manifest }) {
  const propsFiles = (manifest && manifest.props) || []
  const propsState = useJson(propsFiles.length ? `/data/props/${propsFiles[propsFiles.length - 1]}` : '/data/props/none.json', [propsFiles.length])
  const [openGame, setOpenGame] = useState(null)
  const MARKET_LABELS = { player_pass_yds: 'Passing yards', player_rush_yds: 'Rushing yards', player_reception_yds: 'Receiving yards', player_anytime_td: 'Anytime TD' }
  const fmtPrice = (x) => (x == null ? '—' : x > 0 ? `+${x}` : `${x}`)
  const usageRank = {}
  if (playerLeaders.data) playerLeaders.data.high_usage.forEach((r, i) => { usageRank[r.player] = i + 1 })
  return (
    <section>
      <h2 className="section-heading">Players</h2>
      {!playerLeaders.data ? <p className="section-sub">{playerLeaders.loading ? 'Loading…' : 'Leaders publish with the weekly data refresh.'}</p> : (
        <>
          <p className="section-sub">League leaders — {playerLeaders.data.label}.</p>
          <div className="leaders-grid">
            {Object.entries(playerLeaders.data.categories).map(([cat, rows]) => (
              <div key={cat} className="leader-block">
                <h3 className="division-heading">{cat}</h3>
                <table className="ratings-table leader-table"><tbody>
                  {rows.slice(0, 10).map((r, i) => (
                    <tr key={r.player}><td className="rank-cell">{i + 1}</td>
                      <td className="team-cell"><TeamMark league="NFL" team={r.team} size={16} /> {r.player}</td>
                      <td className="numeric">{Math.round(r.value)}</td></tr>
                  ))}
                </tbody></table>
              </div>
            ))}
            <div className="leader-block">
              <h3 className="division-heading">High usage (touches)</h3>
              <table className="ratings-table leader-table"><tbody>
                {playerLeaders.data.high_usage.map((r, i) => (
                  <tr key={r.player}><td className="rank-cell">{i + 1}</td>
                    <td className="team-cell"><TeamMark league="NFL" team={r.team} size={16} /> {r.player}</td>
                    <td className="numeric">{Math.round(r.value)}</td></tr>
                ))}
              </tbody></table>
            </div>
          </div>
        </>
      )}
      <h2 className="section-heading" style={{ marginTop: 28 }}>Player props — NFL</h2>
      {!propsState.data ? (
        <p className="section-sub">{propsFiles.length ? 'Loading…' : 'Props publish weekly with the first Thursday odds run.'}</p>
      ) : (
        <>
          <p className="props-note">{propsState.data.note}</p>
          {(() => {
            const rows = []
            for (const [gm, gdata] of Object.entries(propsState.data.games)) {
              for (const [mk, players] of Object.entries(gdata.markets)) {
                for (const [pl, r] of Object.entries(players)) {
                  if (r.edge && r.edge.ev_pct != null) rows.push({ gm, mk, pl, r, kickoff: gdata.kickoff })
                }
              }
            }
            rows.sort((a, b) => b.r.edge.ev_pct - a.r.edge.ev_pct)
            const top = rows.filter((x) => x.r.edge.ev_pct >= 1.0).slice(0, 12)
            const offMarket = []
            for (const [gm, gdata] of Object.entries(propsState.data.games)) {
              for (const [mk, players] of Object.entries(gdata.markets)) {
                for (const [pl, r] of Object.entries(players)) {
                  for (const om of r.off_market || []) offMarket.push({ gm, mk, pl, ...om })
                }
              }
            }
            offMarket.sort((a, b) => Math.abs(b.vs_consensus) - Math.abs(a.vs_consensus))
            return (
              <>
                <h3 className="division-heading">Best price edges — vs de-vigged consensus</h3>
                {top.length === 0 ? <p className="section-sub">No prop currently beats the multi-book consensus by ≥1% EV. That is a finding, not a failure — most weeks most books agree.</p> : (
                  <div className="props-edge-list">
                    {top.map(({ gm, mk, pl, r, kickoff }) => (
                      <div key={gm + mk + pl} className="bet-card props-edge-card">
                        <div className="bet-card-top">
                          <div className="matchup-block">
                            <p className="matchup-line">{pl} <span className="matchup-at">·</span> {MARKET_LABELS[mk] || mk}{r.edge.side !== 'yes' ? ` ${r.edge.side} ${r.line}` : ''}</p>
                            <p className="kickoff-line">
                              <TeamMark league="NFL" team={gm.split('@')[0]} size={15} /> {gm.replace('@', ' @ ')} {formatKickoff(kickoff) ? '· ' + formatKickoff(kickoff) : ''}
                            </p>
                          </div>
                          <span className={`verdict ${r.edge.ev_pct >= 3 ? 'play' : 'lean'}`}>+{r.edge.ev_pct}% EV</span>
                        </div>
                        <div className="card-chips">
                          <span className="card-chip">Best: {r.edge.price > 0 ? `+${r.edge.price}` : r.edge.price} ({r.edge.book}) · consensus fair {(r.edge.fair_prob * 100).toFixed(1)}% across {r.edge.n_books} books</span>
                          <span className="card-chip">{r.edge.basis}</span>
                          {r.model && (
                            <span className="card-chip">Engine (watch): {r.model.kind === 'score' ? `${Math.round(r.model.p_score * 100)}% to score` : `${Math.round(r.model.p_over * 100)}% over · median ${r.model.median}`}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {offMarket.length > 0 && (
                  <>
                    <h3 className="division-heading" style={{ marginTop: 16 }}>Off-market lines</h3>
                    <div className="card-chips" style={{ marginBottom: 14 }}>
                      {offMarket.slice(0, 8).map((o) => (
                        <span key={o.gm + o.mk + o.pl + o.book} className="card-chip warn">{o.pl} {MARKET_LABELS[o.mk] || o.mk}: {o.book} hangs {o.line} ({o.vs_consensus > 0 ? '+' : ''}{o.vs_consensus} vs consensus)</span>
                      ))}
                    </div>
                  </>
                )}
                <h3 className="division-heading">Full board</h3>
              </>
            )
          })()}
          {Object.entries(propsState.data.games).map(([gm, gdata]) => (
            <div key={gm} className="props-game">
              <button className="props-game-head" onClick={() => setOpenGame(openGame === gm ? null : gm)}>
                <TeamMark league="NFL" team={gm.split('@')[0]} size={18} /> {gm.replace('@', ' @ ')} <TeamMark league="NFL" team={gm.split('@')[1]} size={18} />
                <span className="props-toggle">{openGame === gm ? '−' : '+'}</span>
              </button>
              {openGame === gm && Object.entries(gdata.markets).map(([mk, players]) => (
                <div key={mk} className="props-market">
                  <h4 className="props-market-title">{MARKET_LABELS[mk] || mk}</h4>
                  <table className="ratings-table props-table">
                    <thead><tr><th>Player</th><th className="numeric">Line</th><th className="numeric">Best Over</th><th className="numeric">Best Under / Yes</th><th className="numeric">Engine</th></tr></thead>
                    <tbody>
                      {Object.entries(players).sort((a, b) => (b[1].line || 0) - (a[1].line || 0)).map(([pl, row]) => (
                        <tr key={pl}>
                          <td className="team-cell">{pl}{usageRank[pl] ? <span className="usage-badge">#{usageRank[pl]} touches</span> : null}</td>
                          <td className="numeric">{row.line != null ? row.line : '—'}</td>
                          <td className="numeric">{row.over ? `${fmtPrice(row.over.price)} (${row.over.book})` : '—'}</td>
                          <td className="numeric">{row.under ? `${fmtPrice(row.under.price)} (${row.under.book})` : row.yes ? `${fmtPrice(row.yes.price)} (${row.yes.book})` : '—'}</td>
                          <td className="numeric">{row.model ? (row.model.kind === 'score' ? `${Math.round(row.model.p_score * 100)}%` : `${Math.round(row.model.p_over * 100)}% o`) : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          ))}
        </>
      )}
    </section>
  )
}

function TeamSiteSections({ site, teamAbbr }) {
  if (!site) return null
  const POS_ORDER = ['QB', 'RB', 'WR', 'TE']
  return (
    <>
      <section>
        <h2 className="section-heading">Depth chart</h2>
        <div className="depth-grid">
          {POS_ORDER.filter((p2) => site.depth_chart[p2]).map((pos) => (
            <div key={pos} className="depth-slot">
              <span className="depth-pos">{pos}</span>
              {site.depth_chart[pos].map((pl, i) => (
                <span key={pl} className={`depth-player ${i === 0 ? 'starter' : ''}`}>{pl}</span>
              ))}
            </div>
          ))}
        </div>
      </section>
      <section>
        <h2 className="section-heading">Injury report</h2>
        {site.injuries.length === 0 ? <p className="section-sub">No players with a disclosed designation.</p> : (
          <ul className="injury-list">
            {site.injuries.map((r) => (
              <li key={r.player}><span className={`inj-status ${String(r.status).toLowerCase()}`}>{r.status}</span> {r.player} <span className="inj-pos">{r.position}</span></li>
            ))}
          </ul>
        )}
        <p className="section-sub">Team-page injuries refresh weekly; bet cards refresh hourly on game days.</p>
      </section>
      <section>
        <h2 className="section-heading">Schedule & strength</h2>
        <p className="section-sub">
          Remaining SOS {site.remaining_sos != null ? formatSigned(site.remaining_sos * 100, 1) : '—'} · Played SOS {site.played_sos != null ? formatSigned(site.played_sos * 100, 1) : '—'} (mean opponent rating; higher = harder)
        </p>
        <div className="schedule-list">
          {site.schedule.map((g) => (
            <div key={g.week} className="schedule-row">
              <span className="sched-week">W{g.week}</span>
              <span className="sched-teams"><TeamMark league="NFL" team={g.opp} size={16} /> {g.home ? 'vs' : '@'} {g.opp}</span>
              <span className="sched-when">{g.result || (g.kickoff ? new Date(g.kickoff).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : 'TBD')}</span>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

function RatingsTable({ ratings, onSelectTeam, league = 'NFL', cfbLogos = null }) {
  const [sortKey, setSortKey] = useState('total_rating')
  const [sortDir, setSortDir] = useState('desc')
  const hasPlayoffPct = ratings.some((t) => t.playoff_pct != null)

  const sorted = useMemo(() => {
    const copy = [...ratings]
    copy.sort((a, b) => {
      const diff = a[sortKey] - b[sortKey]
      return sortDir === 'desc' ? -diff : diff
    })
    return copy
  }, [ratings, sortKey, sortDir])

  function handleSort(key) {
    if (key === sortKey) setSortDir(sortDir === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  function headerProps(key) {
    return { className: `numeric${sortKey === key ? ' sorted' : ''}`, onClick: () => handleSort(key) }
  }

  return (
    <table className="ratings-table">
      <thead>
        <tr>
          <th></th>
          <th>Team</th>
          <th {...headerProps('total_rating')}>Rating</th>
          <th {...headerProps('offense_voa')}>Offense</th>
          <th {...headerProps('defense_voa')}>Defense</th>
          {hasPlayoffPct && <th {...headerProps('playoff_pct')}>Playoff %</th>}
        </tr>
      </thead>
      <tbody>
        {sorted.map((team, i) => (
          <tr key={team.team} className="clickable-row" onClick={() => onSelectTeam(team)}>
            <td className="rank-cell">{i + 1}</td>
            <td className="team-cell"><TeamMark league={league} team={team.team} cfbLogos={cfbLogos} size={18} /> {team.team}</td>
            <td className="numeric">
              <span className={`rating-value ${team.total_rating >= 0 ? 'positive' : 'negative'}`}>
                {formatSigned(team.total_rating * 100, 1)}
              </span>
            </td>
            <td className="numeric">{formatSigned(team.offense_voa * 100, 1)}</td>
            <td className="numeric">{formatSigned(team.defense_voa * 100, 1)}</td>
            {hasPlayoffPct && (
              <td className="numeric">{team.playoff_pct != null ? `${(team.playoff_pct * 100).toFixed(0)}%` : '—'}</td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function RatingTrendChart({ history, currentTeam }) {
  if (!history || history.length < 2) {
    return <p className="section-sub">Not enough weekly snapshots yet to show a trend.</p>
  }

  const width = 680
  const height = 160
  const padding = 24
  const values = history.map((h) => h.total_rating)
  const minVal = Math.min(...values, 0)
  const maxVal = Math.max(...values, 0)
  const range = maxVal - minVal || 1

  const xStep = (width - padding * 2) / (history.length - 1)
  const yFor = (v) => height - padding - ((v - minVal) / range) * (height - padding * 2)
  const zeroY = yFor(0)
  const points = history.map((h, i) => `${padding + i * xStep},${yFor(h.total_rating)}`).join(' ')

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${currentTeam} rating trend by week`}>
      <line x1={padding} y1={zeroY} x2={width - padding} y2={zeroY} stroke="rgba(232,238,241,0.15)" strokeWidth="1" />
      <polyline points={points} fill="none" stroke="#f0a93b" strokeWidth="2" />
      {history.map((h, i) => (
        <circle key={h.week} cx={padding + i * xStep} cy={yFor(h.total_rating)} r="3" fill="#f0a93b" />
      ))}
      {history.map((h, i) => (
        <text key={`label-${h.week}`} x={padding + i * xStep} y={height - 4} fontSize="11" fill="#8c97a0" textAnchor="middle">
          Wk {h.week}
        </text>
      ))}
    </svg>
  )
}

function TeamProfilePage({ team, onBack, league = 'NFL', siteInfo = null, cfbLogos = null }) {
  const grade = Math.max(0, Math.min(100, Math.round(50 + team.total_rating * 125)))
  const gradeClass = grade >= 60 ? 'positive' : grade <= 40 ? 'negative' : ''
  const ratingsHistory = useRatingsHistory(league === 'CFB' ? 'cfb_ratings' : 'ratings')
  const teamHistory = ratingsHistory.history ? ratingsHistory.history[team.team] : null

  const site = siteInfo  // teams.json entry (NFL only): schedule, depth, injuries, SOS
  const tiles = [
    { label: 'EPA / play', primary: formatSigned(team.epa_per_play_offense, 3), rows: [['Allowed', formatSigned(team.epa_per_play_allowed, 3)]] },
    { label: 'Success rate', primary: formatPercent(team.success_rate_offense), rows: [['Allowed', formatPercent(team.success_rate_allowed)]] },
    {
      label: 'Rating (opponent-adjusted)',
      primary: formatSigned(team.total_rating != null ? team.total_rating * 100 : null, 1),
      rows: [
        ['Offense', formatSigned(team.offense_voa != null ? team.offense_voa * 100 : null, 1)],
        ['Defense', formatSigned(team.defense_voa != null ? team.defense_voa * 100 : null, 1)],
        ...(team.rating_p05 != null && team.rating_p95 != null
          ? [['90% range', `${formatSigned(team.rating_p05 * 100, 1)} to ${formatSigned(team.rating_p95 * 100, 1)}`]] : []),
        ...(team.total_rating_last_4 != null ? [['Last 4 weeks', formatSigned(team.total_rating_last_4 * 100, 1)]] : []),
      ],
    },
    { label: 'Special teams', primary: formatSigned(team.special_teams_voa != null ? team.special_teams_voa * 100 : null, 1), rows: [] },
    {
      label: 'Elo rating',
      primary: team.elo_rating != null ? Math.round(team.elo_rating).toString() : '—',
      rows: [['vs 1500 baseline', formatSigned(team.elo_rating != null ? team.elo_rating - 1500 : null, 0)]],
    },
    {
      label: 'Red zone pts / trip',
      primary: formatNumber(team.red_zone_points_per_trip),
      rows: [['TD rate', formatPercent(team.red_zone_td_pct)], ['Trips', team.red_zone_trips ?? '—']],
    },
    {
      label: 'Turnover margin',
      primary: formatSigned(team.turnover_margin, 0),
      rows: [['Takeaways', team.takeaways ?? '—'], ['Giveaways', team.giveaways ?? '—']],
    },
  ]

  return (
    <div>
      <button className="back-link" onClick={onBack}>&larr; All teams</button>

      <div className="profile-header">
        <div>
          <h1 className="profile-team-name">{team.team}</h1>
          <p className="profile-meta">Season profile</p>
        </div>
        <div className={`grade-badge ${gradeClass}`}>{grade}</div>
      </div>

      <div className="tile-grid">
        {tiles.map((tile) => (
          <div className="stat-tile" key={tile.label}>
            <span className="tile-label">{tile.label}</span>
            <span className="tile-primary">{tile.primary}</span>
            {tile.rows.map(([label, value]) => (
              <div className="tile-row" key={label}>
                <span>{label}</span>
                <span>{value}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <section>
        <h2 className="section-heading">Rating trend</h2>
        {ratingsHistory.loading && <p className="section-sub">Loading…</p>}
        {teamHistory && <RatingTrendChart history={teamHistory} currentTeam={team.team} />}
        {!ratingsHistory.loading && !teamHistory && (
          <p className="section-sub">No trend data yet for {team.team}.</p>
        )}
      </section>
      <TeamSiteSections site={siteInfo} teamAbbr={team.team} />
    </div>
  )
}

function PlayerGradesSection({ grades }) {
  const positions = [
    { key: 'QB', label: 'Quarterbacks' },
    { key: 'WR_TE', label: 'Wide receivers / tight ends' },
    { key: 'RB', label: 'Running backs' },
  ]

  return (
    <div className="player-grades-grid">
      {positions.map(({ key, label }) => (
        <div key={key} className="player-grades-column">
          <h3 className="player-grades-heading">{label}</h3>
          {(grades[key] || []).slice(0, 8).map((p, i) => (
            <div className="player-grade-row" key={p.player}>
              <span className="player-grade-rank">{i + 1}</span>
              <span className="player-grade-name">{p.player}</span>
              <span className={`player-grade-value ${p.grade >= 60 ? 'positive' : p.grade <= 40 ? 'negative' : ''}`}>
                {p.grade.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

/* ---------------- App shell ---------------- */

const TABS = [
  { id: 'board', label: 'This week' },
  { id: 'schedule', label: 'Schedule' },
  { id: 'record', label: 'Track record' },
  { id: 'ratings', label: 'Teams' },
  { id: 'players', label: 'Players' },
  { id: 'book', label: 'My book' },
]

const NFL_ESPN_ABBR = { WAS: 'wsh', LA: 'lar' }
const NFL_DIVISION_ORDER = ['AFC East', 'AFC North', 'AFC South', 'AFC West', 'NFC East', 'NFC North', 'NFC South', 'NFC West']

function nflLogo(abbr) {
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${(NFL_ESPN_ABBR[abbr] || abbr).toLowerCase()}.png`
}

function TeamMark({ league, team, cfbLogos, size = 20 }) {
  const src = league === 'CFB' ? (cfbLogos ? cfbLogos[team] : null) : nflLogo(team)
  if (!src) return null
  return <img className="team-logo" src={src} width={size} height={size} alt="" loading="lazy" onError={(e) => { e.target.style.display = 'none' }} />
}

function useJson(url, deps = []) {
  const [state, setState] = useState({ data: null, loading: true })
  useEffect(() => {
    setState({ data: null, loading: true })
    fetch(url).then((r) => { if (!r.ok) throw new Error('none'); return r.json() })
      .then((data) => setState({ data, loading: false }))
      .catch(() => setState({ data: null, loading: false }))
  }, deps)
  return state
}

const ESPN_SB = {
  NFL: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
  CFB: 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=120',
}
const ESPN_TO_NFLVERSE = { WSH: 'WAS', LAR: 'LA' }

function useOpeningLines(league, manifest, season, week) {
  // Opening line per game: the FIRST snapshot of the current week.
  // One extra fetch; powers the per-card line-movement readout.
  const fam = league === 'CFB' ? 'cfb_divergence' : 'divergence'
  const files = (manifest && manifest[fam]) || []
  const prefix = season != null && week != null ? `${season}-week-${String(week).padStart(2, '0')}` : null
  const first = prefix ? files.find((f) => f.startsWith(prefix)) : null
  const st = useJson(first ? `/data/${fam}/${first}` : '/data/none.json', [first])
  const map = {}
  if (st.data) st.data.divergences.forEach((d) => { map[`${d.away_team}@${d.home_team}`] = d.market_spread })
  return map
}

function useLiveScores(league, active, cfbNames) {
  // Client-side ESPN polling (their API allows browser origins --
  // verified from this site's own origin). Only runs while the board
  // tab is visible; 45s cadence; maps back to our team keys.
  const [scores, setScores] = useState({})
  useEffect(() => {
    if (!active) return undefined
    let stop = false
    const strip = (n) => n.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim()
    const deriv = new Set(['tech', 'state', 'am', 'southern', 'christian', 'wesleyan', 'baptist', 'international', 'central'])
    const cfbCanon = {}
    if (cfbNames) cfbNames.forEach((n) => { cfbCanon[strip(n)] = n })
    const resolveCfb = (name) => {
      const w = strip(name).split(' ')
      for (const k of [1, 2]) {
        if (w.length <= k) break
        if (k === 2 && deriv.has(w[w.length - 2])) continue
        const cand = cfbCanon[w.slice(0, -k).join(' ')]
        if (cand) return cand
      }
      return null
    }
    async function poll() {
      try {
        const j = await (await fetch(ESPN_SB[league])).json()
        const next = {}
        for (const ev of j.events || []) {
          const comp = ev.competitions && ev.competitions[0]
          if (!comp) continue
          const home = comp.competitors.find((c) => c.homeAway === 'home')
          const away = comp.competitors.find((c) => c.homeAway === 'away')
          if (!home || !away) continue
          let hk, ak
          if (league === 'CFB') {
            hk = resolveCfb(home.team.displayName); ak = resolveCfb(away.team.displayName)
          } else {
            hk = ESPN_TO_NFLVERSE[home.team.abbreviation] || home.team.abbreviation
            ak = ESPN_TO_NFLVERSE[away.team.abbreviation] || away.team.abbreviation
          }
          if (!hk || !ak) continue
          const st = comp.status || {}
          next[`${ak}@${hk}`] = {
            hs: Number(home.score), as: Number(away.score),
            state: st.type && st.type.state, detail: st.type && st.type.shortDetail,
          }
        }
        if (!stop) setScores(next)
      } catch (e) { /* soft-fail: scores are decoration */ }
    }
    poll()
    const t = setInterval(() => { if (document.visibilityState === 'visible') poll() }, 45000)
    return () => { stop = true; clearInterval(t) }
  }, [league, active, cfbNames && cfbNames.length])
  return scores
}

export default function App() {
  const [league, setLeague] = useState('NFL')
  const [tab, setTab] = useState('board')
  const [selectedTeam, setSelectedTeam] = useState(null)
  const manifestState = useJson('/data/manifest.json')
  const siteTeams = useJson('/data/site/teams.json')
  const playerLeaders = useJson('/data/site/player_leaders.json')
  const cfbLogosState = useJson('/data/site/cfb_logos.json')
  const cfbLogos = cfbLogosState.data ? cfbLogosState.data.logos : null
  const liveScores = useLiveScores(league, tab === 'board', cfbLogos ? Object.keys(cfbLogos) : null)

  const account = useAccount()
  const book = useBook(account)
  const perf = usePerformance()
  const marginDist = useMarginDist()

  const ratingsState = useLatestSnapshot('ratings')
  const cfbRatingsState = useLatestSnapshot('cfb_ratings')
  const divergenceState = useLatestSnapshot('divergence')
  const cfbDivergenceState = useLatestSnapshot('cfb_divergence')
  const mlbDivergenceState = useLatestSnapshot('mlb_divergence')
  const activeDivData = league === 'CFB' ? (cfbDivergenceState.data || null) : (divergenceState.data || null)
  const openingLines = useOpeningLines(league, manifestState.data, activeDivData && activeDivData.season, activeDivData && activeDivData.week)
  const playerGradesState = useLatestSnapshot('player_grades')

  const activeRatingsState = league === 'CFB' ? cfbRatingsState : ratingsState

  const ratingsByTeam = useMemo(() => {
    if (!ratingsState.data) return null
    const byTeam = {}
    ratingsState.data.ratings.forEach((t) => { byTeam[t.team] = t })
    return byTeam
  }, [ratingsState.data])

  const cfbRatingsByTeam = useMemo(() => {
    if (!cfbRatingsState.data) return null
    const byTeam = {}
    cfbRatingsState.data.ratings.forEach((t) => { byTeam[t.team] = t })
    return byTeam
  }, [cfbRatingsState.data])

  if (selectedTeam) {
    return (
      <div className="page">
        <TeamProfilePage team={selectedTeam} onBack={() => setSelectedTeam(null)} league={league} siteInfo={league === 'NFL' && siteTeams.data ? siteTeams.data.teams[selectedTeam.team] : null} cfbLogos={cfbLogos} />
      </div>
    )
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-row">
          <h1 className="brand">Cover<em>line</em></h1>
          <div className="masthead-right">
            <AccountChip account={account} />
            <div className="league-toggle" role="tablist" aria-label="League">
              {['NFL', 'CFB', 'MLB'].map((l) => (
                <button key={l} className={league === l ? 'active' : ''} onClick={() => setLeague(l)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
        </div>
        {activeRatingsState.data && (
          <p className="stamp">
            {activeRatingsState.data.season} season · week {activeRatingsState.data.week} · updated{' '}
            {new Date(activeRatingsState.data.computed_at).toLocaleDateString()}
          </p>
        )}
        <nav className="tabs" aria-label="Sections">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === 'board' && league === 'MLB' && (
        <MlbBoard snap={mlbDivergenceState.data} />
      )}
      {tab === 'board' && league !== 'MLB' && (
        <section>
          {league === 'NFL' ? (
            <>
              {divergenceState.loading && <p className="section-sub">Loading…</p>}
              {divergenceState.error && (
                <div className="empty-state">
                  <strong>No lines posted yet</strong>
                  Sportsbooks open lines gradually as kickoff approaches — the board fills in
                  automatically on game weeks.
                </div>
              )}
              {divergenceState.data && (
                <EdgeBoard
                  divergences={divergenceState.data.divergences}
                  note={divergenceState.data.note}
                  season={divergenceState.data.season}
                  week={divergenceState.data.week}
                  book={book}
                  ratingsByTeam={ratingsByTeam}
                  perf={perf.data}
                  marginDist={marginDist}
                  qb1Map={divergenceState.data.qb1_map}
                  boardLeague="NFL"
                  boardScores={liveScores}
                  boardOpenLines={openingLines}
                  boardSiteTeams={siteTeams.data}
                />
              )}
            </>
          ) : (
            <>
              {cfbDivergenceState.loading && <p className="section-sub">Loading…</p>}
              {cfbDivergenceState.error && (
                <div className="empty-state">
                  <strong>No CFB lines gathered yet</strong>
                  The CFB odds watch fills this board automatically once it runs against live NCAAF
                  odds. Team ratings are already live under the Ratings tab.
                </div>
              )}
              {cfbDivergenceState.data && (
                <EdgeBoard
                  divergences={cfbDivergenceState.data.divergences}
                  note={cfbDivergenceState.data.note}
                  season={cfbDivergenceState.data.season}
                  week={cfbDivergenceState.data.week}
                  book={book}
                  ratingsByTeam={cfbRatingsByTeam}
                  perf={null}
                  marginDist={null}
                  /* Weeks 1-4 cap at Lean: the held-out backtest graded
                     early-season flags BELOW breakeven (48.3%) -- ratings
                     are data-starved before week 5. Evidence-backed
                     demotion, unlike the QB case where the data said
                     annotate-only. */
                  playGap={(cfbDivergenceState.data.week ?? 5) <= 4 ? 999 : 5}
                  leanGap={3}
                  /* CFB-specific calibration: logistic fit to the
                     held-out 2023 bucket table (574 games). A 22-pt
                     carryover gap is ~60% cover, not the ~94% the NFL
                     normal approximation was displaying. */
                  edgeCoefOverride={0.01828}
                  boardLeague="CFB"
                  boardCfbLogos={cfbLogos}
                  boardScores={liveScores}
                  boardOpenLines={openingLines}
                />
              )}
            </>
          )}
        </section>
      )}

      {tab === 'record' && <TrackRecord league={league} />}

      {tab === 'book' && <MyBook book={book} account={account} />}

      {tab === 'schedule' && league === 'MLB' && (
        <section><h2 className="section-heading">Season schedule — MLB</h2><p className="section-sub">MLB schedule view arrives with the 2027 board launch.</p></section>
      )}
      {tab === 'schedule' && league !== 'MLB' && (league === 'NFL'
        ? <ScheduleTab siteTeams={siteTeams} />
        : <section><h2 className="section-heading">Season schedule — CFB</h2><p className="section-sub">The CFB slate lives on the This week board; a full 136-team schedule view is on the roadmap.</p></section>)}

      {tab === 'players' && league === 'MLB' && (
        <section><h2 className="section-heading">Players — MLB</h2><p className="section-sub">MLB player surfaces arrive with the 2027 board launch.</p></section>
      )}
      {tab === 'players' && league !== 'MLB' && (league === 'NFL'
        ? <PlayersTab playerLeaders={playerLeaders} manifest={manifestState.data} />
        : <section><h2 className="section-heading">Players — CFB</h2><p className="section-sub">Player surfaces are NFL-only for now (college player data volume is a different animal).</p></section>)}

      {tab === 'ratings' && league === 'MLB' && (
        <section><h2 className="section-heading">Teams — MLB</h2><p className="section-sub">MLB team ratings publish with the 2027 board launch; the model runs and is graded before anything shows here.</p></section>
      )}
      {tab === 'ratings' && league !== 'MLB' && (
        <>
          <section>
            <h2 className="section-heading">Team ratings</h2>
            <p className="section-sub">
              Opponent-adjusted efficiency, the engine behind every number on the board. Tap a team
              for its full profile.
            </p>
            {activeRatingsState.loading && <p className="section-sub">Loading…</p>}
            {activeRatingsState.error && (
              <div className="empty-state">
                <strong>No {league} ratings published yet</strong>
                Ratings publish weekly once the season starts.
              </div>
            )}
            {league === 'NFL' && siteTeams.data && activeRatingsState.data && (
              <DivisionStandings siteTeams={siteTeams.data} ratingsByTeam={ratingsByTeam}
                onSelectTeam={setSelectedTeam} allRatings={activeRatingsState.data.ratings} />
            )}
            {activeRatingsState.data && (league !== 'NFL' || !siteTeams.data) && (
              <RatingsTable ratings={activeRatingsState.data.ratings} onSelectTeam={setSelectedTeam}
                league={league} cfbLogos={cfbLogos} />
            )}
          </section>

          {league === 'NFL' && (
            <section>
              <h2 className="section-heading">Player grades</h2>
              <p className="section-sub">
                From real player-tracking data — accuracy over expectation for QBs, yards after catch
                over expectation for receivers, rushing yards over expected for backs.
              </p>
              {playerGradesState.data ? (
                <PlayerGradesSection grades={playerGradesState.data.grades} />
              ) : (
                <div className="empty-state">
                  <strong>No player grades yet</strong>
                  Grades need a few weeks of in-season tracking data.
                </div>
              )}
            </section>
          )}
        </>
      )}

      <footer className="footnote">
        <p>
          Every number here is a model estimate, not a guarantee, and nothing on this site is betting
          advice. Estimated cover probabilities are approximations and real results will vary.
        </p>
        <p>
          Bet only what you can afford to lose. If gambling stops being fun, call or text 1-800-GAMBLER.
          21+ where required.
        </p>
      </footer>
    </div>
  )
}
