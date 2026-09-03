import { useEffect, useState, useMemo } from 'react'

function useJson(path) {
  const [state, setState] = useState({ data: null, loading: true, error: null })

  useEffect(() => {
    fetch(path)
      .then((res) => {
        if (!res.ok) throw new Error('not found')
        return res.json()
      })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((error) => setState({ data: null, loading: false, error }))
  }, [path])

  return state
}

/**
 * Fetches every available ratings snapshot (not just the latest) to
 * build a week-over-week trend per team — now possible thanks to the
 * immutable snapshot architecture, which never had a technical reason
 * to stop at "just the current one."
 */
function useRatingsHistory() {
  const [state, setState] = useState({ history: null, loading: true, error: null })

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => {
        if (!res.ok) throw new Error('no manifest')
        return res.json()
      })
      .then((manifest) => {
        const files = manifest.ratings || []
        if (files.length === 0) {
          setState({ history: null, loading: false, error: new Error('no snapshots yet') })
          return
        }
        return Promise.all(files.map((f) => fetch(`/data/ratings/${f}`).then((r) => r.json()))).then(
          (snapshots) => {
            // Real bug found via visual review: sorting by week alone
            // silently merges different SEASONS into one misleading
            // line if snapshots from more than one season ever coexist
            // (which has genuinely happened this project -- 2023 demo
            // data sitting alongside real 2026 data). Only trend the
            // most recent season present, sorted by (season, week).
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
  }, [])

  return state
}

function RatingTrendChart({ history, currentTeam }) {
  if (!history || history.length < 2) {
    return (
      <p className="section-sub">Not enough weekly snapshots yet to show a trend — check back after a few more weeks.</p>
    )
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
      <line x1={padding} y1={zeroY} x2={width - padding} y2={zeroY} stroke="var(--hairline)" strokeWidth="1" />
      <polyline points={points} fill="none" stroke="var(--amber)" strokeWidth="2" />
      {history.map((h, i) => (
        <circle key={h.week} cx={padding + i * xStep} cy={yFor(h.total_rating)} r="3" fill="var(--amber)" />
      ))}
      {history.map((h, i) => (
        <text
          key={`label-${h.week}`}
          x={padding + i * xStep}
          y={height - 4}
          fontSize="11"
          fill="var(--chalk-dim)"
          textAnchor="middle"
        >
          Wk {h.week}
        </text>
      ))}
    </svg>
  )
}

/**
 * Ratings and divergence are now stored as immutable per-week/per-check
 * snapshot files rather than one overwritten file (fixes the git
 * friction repeated overwrites were causing). The manifest — generated
 * at build time, not committed to git — lists what's available, so the
 * site can find and load the latest one.
 */
function useLatestSnapshot(kind) {
  const [state, setState] = useState({ data: null, loading: true, error: null, filename: null })

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => {
        if (!res.ok) throw new Error('no manifest')
        return res.json()
      })
      .then((manifest) => {
        const files = manifest[kind] || []
        if (files.length === 0) {
          setState({ data: null, loading: false, error: new Error('no snapshots yet'), filename: null })
          return
        }
        // Filenames sort correctly as strings since weeks are zero-padded
        // (2023-week-05.json < 2023-week-18.json) — the last one is latest.
        const latest = files[files.length - 1]
        return fetch(`/data/${kind}/${latest}`)
          .then((res) => res.json())
          .then((data) => setState({ data, loading: false, error: null, filename: latest }))
      })
      .catch((error) => setState({ data: null, loading: false, error, filename: null }))
  }, [kind])

  return state
}

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
    if (key === sortKey) {
      setSortDir(sortDir === 'desc' ? 'asc' : 'desc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  function headerProps(key) {
    return {
      className: `numeric${sortKey === key ? ' sorted' : ''}`,
      onClick: () => handleSort(key),
    }
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

function DivergenceSection({ divergences }) {
  if (divergences.length === 0) {
    return (
      <div className="empty-state">
        <strong>No lines posted yet</strong>
        Sportsbooks open lines gradually as kickoff approaches — check back closer to game day.
      </div>
    )
  }

  return (
    <div className="matchup-list">
      {divergences.map((d) => {
        const modelSpread = d.market_spread + d.spread_gap
        const modelTotal = d.market_total + d.total_gap
        const anyFlagged = d.spread_flagged || d.total_flagged || d.win_prob_flagged

        return (
          <div className={`matchup${anyFlagged ? ' flagged' : ''}`} key={`${d.home_team}-${d.away_team}`}>
            <div className="teams">
              {d.away_team}
              <span className="at">at</span>
              {d.home_team}
            </div>
            <div className="line-compare">
              <div>
                <span className="label">Market spread</span>
                <span className="value">{formatSigned(d.market_spread)}</span>
              </div>
              <div>
                <span className="label">Model spread</span>
                <span className="value model">{formatSigned(modelSpread)}</span>
              </div>
              <div>
                <span className="label">Market total</span>
                <span className="value">{d.market_total.toFixed(1)}</span>
              </div>
              <div>
                <span className="label">Model total</span>
                <span className="value model">{modelTotal.toFixed(1)}</span>
              </div>
            </div>
            {anyFlagged && <span className="flag-badge">Diverges from market</span>}
          </div>
        )
      })}
    </div>
  )
}

function TeamProfilePage({ team, onBack }) {
  const grade = Math.max(0, Math.min(100, Math.round(50 + team.total_rating * 125)))
  const gradeClass = grade >= 60 ? 'positive' : grade <= 40 ? 'negative' : ''
  const ratingsHistory = useRatingsHistory()
  const teamHistory = ratingsHistory.history ? ratingsHistory.history[team.team] : null

  const tiles = [
    {
      label: 'EPA / Play',
      primary: formatSigned(team.epa_per_play_offense, 3),
      rows: [['Allowed', formatSigned(team.epa_per_play_allowed, 3)]],
    },
    {
      label: 'Success rate',
      primary: formatPercent(team.success_rate_offense),
      rows: [['Allowed', formatPercent(team.success_rate_allowed)]],
    },
    {
      label: 'DVOA (opponent-adjusted)',
      primary: formatSigned(team.total_rating != null ? team.total_rating * 100 : null, 1),
      rows: [
        ['Offense', formatSigned(team.offense_voa != null ? team.offense_voa * 100 : null, 1)],
        ['Defense', formatSigned(team.defense_voa != null ? team.defense_voa * 100 : null, 1)],
        ...(team.rating_p05 != null && team.rating_p95 != null
          ? [['90% range', `${formatSigned(team.rating_p05 * 100, 1)} to ${formatSigned(team.rating_p95 * 100, 1)}`]]
          : []),
      ],
    },
    {
      label: 'Special teams',
      primary: formatSigned(team.special_teams_voa != null ? team.special_teams_voa * 100 : null, 1),
      rows: [],
    },
    {
      label: 'Red zone: pts / trip',
      primary: formatNumber(team.red_zone_points_per_trip),
      rows: [
        ['TD rate', formatPercent(team.red_zone_td_pct)],
        ['Trips', team.red_zone_trips ?? '—'],
      ],
    },
    {
      label: 'Turnover margin',
      primary: formatSigned(team.turnover_margin, 0),
      rows: [
        ['Takeaways', team.takeaways ?? '—'],
        ['Giveaways', team.giveaways ?? '—'],
      ],
    },
  ]

  return (
    <div>
      <button className="back-link" onClick={onBack}>
        &larr; All teams
      </button>

      <div className="profile-header">
        <div>
          <h1 className="profile-team-name">{team.team}</h1>
          <p className="meta">Season profile</p>
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

      <h2 className="section-heading">Rating trend</h2>
      <p className="section-sub">Total rating (DVOA) across every published weekly snapshot.</p>
      {ratingsHistory.loading && <p className="section-sub">Loading…</p>}
      {teamHistory && <RatingTrendChart history={teamHistory} currentTeam={team.team} />}
      {!ratingsHistory.loading && !teamHistory && (
        <p className="section-sub">No trend data yet for {team.team}.</p>
      )}

      <p className="section-sub" style={{ marginTop: '32px' }}>
        Recent-form time windows (last 4/8 games, etc.) still aren't broken out separately from the
        season-long rating shown above — the snapshots now exist to support this, it just hasn't been
        built as a distinct view yet.
      </p>
    </div>
  )
}

/**
 * Client-side mirror of model/clv_tracking.py's compute_clv_for_game --
 * same logic, same sign convention, kept in sync deliberately rather
 * than duplicated by accident. Fetches every divergence snapshot (not
 * just the latest) to measure real line movement between the earliest
 * and latest check for each game.
 */
function useClvReport() {
  const [state, setState] = useState({ games: null, loading: true, error: null })

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => {
        if (!res.ok) throw new Error('no manifest')
        return res.json()
      })
      .then((manifest) => {
        const files = manifest.divergence || []
        if (files.length < 2) {
          setState({ games: null, loading: false, error: new Error('need at least 2 snapshots to measure movement') })
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
              const openingMarketSpread = earliest.market_spread
              const closingMarketSpread = latest.market_spread
              const divergenceDirection = modelSpread - openingMarketSpread
              const marketMovement = closingMarketSpread - openingMarketSpread
              const clvScore = divergenceDirection === 0 ? 0 : marketMovement * (divergenceDirection > 0 ? 1 : -1)

              games.push({
                away, home, nSnapshots: appearances.length,
                openingMarketSpread, closingMarketSpread, marketMovement,
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

function ClvSection() {
  const { games, loading, error } = useClvReport()

  if (loading) return <p className="section-sub">Loading…</p>
  if (error || !games || games.length === 0) {
    return (
      <div className="empty-state">
        <strong>No line-movement data yet</strong>
        Closing-line value needs at least two odds checks on the same game — check back once the
        odds-watch job has run more than once this week.
      </div>
    )
  }

  return (
    <div className="clv-list">
      {games.map((g) => (
        <div className="clv-row" key={`${g.away}@${g.home}`}>
          <span className="clv-matchup">
            {g.away} <span className="clv-at">at</span> {g.home}
          </span>
          <span className="clv-detail">
            Open {formatSigned(g.openingMarketSpread, 1)} → Close {formatSigned(g.closingMarketSpread, 1)}
          </span>
          <span className={`clv-badge ${g.validated ? 'clv-toward' : 'clv-away'}`}>
            {g.validated ? 'Moved toward model' : 'Moved away from model'}
          </span>
        </div>
      ))}
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

export default function App() {
  const ratingsState = useLatestSnapshot('ratings')
  const divergenceState = useLatestSnapshot('divergence')
  const playerGradesState = useLatestSnapshot('player_grades')
  const [selectedTeam, setSelectedTeam] = useState(null)

  if (selectedTeam) {
    return (
      <div className="page">
        <TeamProfilePage team={selectedTeam} onBack={() => setSelectedTeam(null)} />
      </div>
    )
  }

  return (
    <div className="page">
      <header className="masthead">
        <h1>
          Power <span className="accent">Ratings</span>
        </h1>
        {ratingsState.data && (
          <p className="meta">
            <strong>
              {ratingsState.data.season} season, week {ratingsState.data.week}
            </strong>{' '}
            — updated {new Date(ratingsState.data.computed_at).toLocaleString()}
            {ratingsState.data.methodology_version && (
              <span> · methodology v{ratingsState.data.methodology_version}</span>
            )}
          </p>
        )}
      </header>

      <section>
        <h2 className="section-heading">Team ratings</h2>
        <p className="section-sub">
          Opponent-adjusted efficiency, DVOA-style. Click a column to sort, click a team for its full
          profile.
        </p>
        {ratingsState.loading && <p className="section-sub">Loading…</p>}
        {ratingsState.error && (
          <div className="empty-state">
            <strong>No ratings published yet</strong>
            The weekly ratings job runs every Tuesday once the season starts — check back after the
            first week's games.
          </div>
        )}
        {ratingsState.data && (
          <RatingsTable ratings={ratingsState.data.ratings} onSelectTeam={setSelectedTeam} />
        )}
      </section>

      <section>
        <h2 className="section-heading">This week: model vs. market</h2>
        <p className="section-sub">
          The model's own prediction, shown alongside the market's de-vigged line. A flagged game means
          the two disagree by more than a small threshold — not a recommendation, just a gap worth
          looking at.
        </p>
        {divergenceState.loading && <p className="section-sub">Loading…</p>}
        {divergenceState.error && (
          <div className="empty-state">
            <strong>No comparison data yet</strong>
            The odds-watch job only runs on real game days, and only after the weekly ratings job has
            produced this week's ratings.
          </div>
        )}
        {divergenceState.data && <DivergenceSection divergences={divergenceState.data.divergences} />}
      </section>

      <section>
        <h2 className="section-heading">Line movement</h2>
        <p className="section-sub">
          Did the market move toward or away from where the model originally diverged? A line moving
          toward the model's view is a real signal it saw something; moving away suggests the
          divergence was more likely noise.
        </p>
        <ClvSection />
      </section>

      <section>
        <h2 className="section-heading">Player grades</h2>
        <p className="section-sub">
          Real player-tracking data (Next Gen Stats) — completion accuracy above expectation for QBs,
          yards-after-catch above expectation for WR/TE, rushing yards over expected for RBs. A
          different lens than the team ratings above, not a restatement of them.
        </p>
        {playerGradesState.loading && <p className="section-sub">Loading…</p>}
        {playerGradesState.error && (
          <div className="empty-state">
            <strong>No player grades yet</strong>
            Player grades need real in-season tracking data — check back after a few weeks of games.
          </div>
        )}
        {playerGradesState.data && <PlayerGradesSection grades={playerGradesState.data.grades} />}
      </section>

      <footer className="footnote">
        Model output only — not betting advice. Spreads and totals shown are the model's own estimate,
        not a guarantee.
      </footer>
    </div>
  )
}
