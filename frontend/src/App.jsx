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

function formatSigned(value, digits = 1) {
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

function RatingsTable({ ratings }) {
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
          <tr key={team.team}>
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

export default function App() {
  const ratingsState = useJson('/data/ratings.json')
  const divergenceState = useJson('/data/divergence.json')

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
        <p className="section-sub">Opponent-adjusted efficiency, DVOA-style. Click a column to sort.</p>
        {ratingsState.loading && <p className="section-sub">Loading…</p>}
        {ratingsState.error && (
          <div className="empty-state">
            <strong>No ratings published yet</strong>
            The weekly ratings job runs every Tuesday once the season starts — check back after the
            first week's games.
          </div>
        )}
        {ratingsState.data && <RatingsTable ratings={ratingsState.data.ratings} />}
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
