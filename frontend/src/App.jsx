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

function EdgeBoard({ divergences, note, season, week, book, ratingsByTeam, perf, marginDist, playGap = PLAY_GAP, leanGap = LEAN_GAP }) {
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
        const edgeCoef = marginDist ? marginDist.edge_calibration?.edge_coef : null
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
              <span className="bet-pick">{pick}</span>
              <span className={`verdict ${verdict}`}>{verdict === 'play' ? 'Play' : 'Lean'}</span>
            </div>

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

            <p className="bet-reason">{describeReason(d, market)}</p>
            {d.best_prices && (() => {
              const bp = d.best_prices
              const pickBest = market === 'spread'
                ? (d.spread_gap > 0 ? bp.home_spread : bp.away_spread)
                : (d.total_gap > 0 ? bp.over : bp.under)
              if (!pickBest || pickBest.price == null) return null
              const shownPoint = market === 'spread' && d.spread_gap > 0 ? -pickBest.point : pickBest.point
              return (
                <p className="best-price-line">
                  Best price: {formatSigned(shownPoint, 1)} at {pickBest.price > 0 ? `+${pickBest.price}` : pickBest.price}
                  {' '}({pickBest.book}, {bp.n_books} books checked)
                </p>
              )
            })()}
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

function usePerformance() {
  const [state, setState] = useState({ data: null, loading: true })

  useEffect(() => {
    fetch('/data/performance.json')
      .then((res) => { if (!res.ok) throw new Error('none'); return res.json() })
      .then((data) => setState({ data, loading: false }))
      .catch(() => setState({ data: null, loading: false }))
  }, [])

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

function useClvReport() {
  const [state, setState] = useState({ games: null, loading: true, error: null })

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => { if (!res.ok) throw new Error('no manifest'); return res.json() })
      .then((manifest) => {
        const files = manifest.divergence || []
        if (files.length < 2) {
          setState({ games: null, loading: false, error: new Error('need snapshots') })
          return
        }
        return Promise.all(files.map((f) => fetch(`/data/divergence/${f}`).then((r) => r.json()))).then(
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
  }, [])

  return state
}

function TrackRecord() {
  const perf = usePerformance()
  const clv = useClvReport()

  return (
    <div>
      <section>
        <h2 className="section-heading">Season scorecard</h2>
        <p className="section-sub">
          Every flagged play is graded against the closing line and the final score, wins and losses
          alike. If the numbers here go red, you'll see it before we do anything about it.
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

function RatingsTable({ ratings, onSelectTeam }) {
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
            <td className="team-cell">{team.team}</td>
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

function TeamProfilePage({ team, onBack, league = 'NFL' }) {
  const grade = Math.max(0, Math.min(100, Math.round(50 + team.total_rating * 125)))
  const gradeClass = grade >= 60 ? 'positive' : grade <= 40 ? 'negative' : ''
  const ratingsHistory = useRatingsHistory(league === 'CFB' ? 'cfb_ratings' : 'ratings')
  const teamHistory = ratingsHistory.history ? ratingsHistory.history[team.team] : null

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
  { id: 'record', label: 'Track record' },
  { id: 'ratings', label: 'Ratings' },
  { id: 'book', label: 'My book' },
]

export default function App() {
  const [league, setLeague] = useState('NFL')
  const [tab, setTab] = useState('board')
  const [selectedTeam, setSelectedTeam] = useState(null)

  const account = useAccount()
  const book = useBook(account)
  const perf = usePerformance()
  const marginDist = useMarginDist()

  const ratingsState = useLatestSnapshot('ratings')
  const cfbRatingsState = useLatestSnapshot('cfb_ratings')
  const divergenceState = useLatestSnapshot('divergence')
  const cfbDivergenceState = useLatestSnapshot('cfb_divergence')
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
        <TeamProfilePage team={selectedTeam} onBack={() => setSelectedTeam(null)} league={league} />
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
              {['NFL', 'CFB'].map((l) => (
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

      {tab === 'board' && (
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
                  playGap={5}
                  leanGap={3}
                />
              )}
            </>
          )}
        </section>
      )}

      {tab === 'record' && <TrackRecord />}

      {tab === 'book' && <MyBook book={book} account={account} />}

      {tab === 'ratings' && (
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
            {activeRatingsState.data && (
              <RatingsTable ratings={activeRatingsState.data.ratings} onSelectTeam={setSelectedTeam} />
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
