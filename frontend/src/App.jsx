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
            // Build { team: [{ week, total_rating }, ...] }, sorted by week
            const history = {}
            snapshots
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
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

function RatingsTable({ ratings, onSelectTeam }) {
  const [sortKey, setSortKey] = useState('total_rating')
  const [sortDir, setSortDir] = useState('desc')

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
      primary: `${(team.success_rate_offense * 100).toFixed(1)}%`,
      rows: [['Allowed', `${(team.success_rate_allowed * 100).toFixed(1)}%`]],
    },
    {
      label: 'DVOA (opponent-adjusted)',
      primary: formatSigned(team.total_rating * 100, 1),
      rows: [
        ['Offense', formatSigned(team.offense_voa * 100, 1)],
        ['Defense', formatSigned(team.defense_voa * 100, 1)],
      ],
    },
    {
      label: 'Red zone: pts / trip',
      primary: team.red_zone_points_per_trip.toFixed(2),
      rows: [
        ['TD rate', `${(team.red_zone_td_pct * 100).toFixed(1)}%`],
        ['Trips', team.red_zone_trips],
      ],
    },
    {
      label: 'Turnover margin',
      primary: formatSigned(team.turnover_margin, 0),
      rows: [
        ['Takeaways', team.takeaways],
        ['Giveaways', team.giveaways],
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

export default function App() {
  const ratingsState = useLatestSnapshot('ratings')
  const divergenceState = useLatestSnapshot('divergence')
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

      <footer className="footnote">
        Model output only — not betting advice. Spreads and totals shown are the model's own estimate,
        not a guarantee.
      </footer>
    </div>
  )
}
