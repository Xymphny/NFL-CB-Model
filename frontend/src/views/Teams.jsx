import { LEAGUE_NAME } from '../format'

/* What this league's model actually rates, as the board exported it. Each
 * sport gets its own table because each model uses different things. */

const SPEC = {
  nfl: {
    title: 'Team ratings, as priced this week',
    note: 'Total rating is what the live model reads (rating-only vector); the interval is the bootstrap 5th to 95th percentile.',
    cols: [
      ['team', 'Team'], ['total_rating', 'Rating', 3], ['offense_voa', 'Offence', 3],
      ['defense_voa', 'Defence', 3], ['special_teams_voa', 'Special teams', 3],
      ['rating_p05', '5th pct', 3], ['rating_p95', '95th pct', 3],
    ],
  },
  cfb: {
    title: 'Team ratings, as priced this week',
    note: 'Opponent-adjusted efficiency from this season\'s play-by-play, rated on games before this week. The live model reads the total alone (DVOA-only vector).',
    cols: [['team', 'Team'], ['total_rating', 'Rating', 3], ['offense_voa', 'Offence', 3], ['defense_voa', 'Defence', 3]],
  },
  nba: {
    title: 'Team ratings, as priced',
    note: 'Points of margin against an average team, walked forward game by game and regressed halfway between seasons. Team-level only: no injuries or minutes.',
    cols: [['team', 'Team'], ['rating', 'Rating (points)', 2]],
  },
  nhl: {
    title: 'Attack and defence rates, this season',
    note: 'Log goal rates with both goalies on the ice, walked forward from a league-average start each season. Higher attack scores more; higher defence concedes more.',
    cols: [['team', 'Team'], ['attack', 'Attack', 3], ['defence', 'Defence', 3]],
  },
  mlb: {
    title: "Today's probable starters and parks",
    note: 'Runs allowed per 27 outs for each listed starter, shrunk toward the league, and the park factor the model applies.',
    cols: [['pitcher', 'Starter'], ['team', 'Team'], ['ra27', 'RA/27', 2], ['park', 'Park'], ['park_factor', 'Park factor', 3]],
  },
}

export default function Teams({ league, board }) {
  const spec = SPEC[league]
  const rows = board.teams || []
  if (!spec || !rows.length) return null
  /* The first figure column is also drawn as a bar around zero, scaled to the
   * largest value in the table: a position of the published number, no new
   * figure. MLB's table is not centred on zero, so it has no bar. */
  const barKey = league === 'mlb' ? null : spec.cols.find(([, , d]) => d != null)?.[0]
  const maxAbs = barKey ? Math.max(...rows.map((r) => Math.abs(Number(r[barKey]) || 0)), 1e-9) : 1
  return (
    <section className="panel">
      <header className="panel-head">
        <h1>{spec.title}</h1>
        <p>{spec.note}</p>
      </header>
      <table className={`data data-narrow${barKey ? ' with-bar' : ''}`}>
        <thead>
          <tr>
            {spec.cols.map(([k, label, d]) => <th key={k} className={d != null ? 'num' : undefined}>{label}</th>)}
            {barKey && <th aria-hidden="true" />}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.team}-${r.pitcher || i}`}>
              {spec.cols.map(([k, , d]) => (
                <td key={k} className={d != null ? 'num' : undefined}>
                  {d != null ? (r[k] == null ? '—' : Number(r[k]).toFixed(d)) : r[k] ?? '—'}
                </td>
              ))}
              {barKey && (
                <td className="bar-cell" aria-hidden="true">
                  <span className="dbar">
                    <span className={Number(r[barKey]) >= 0 ? 'pos' : 'neg'}
                          style={{ width: `${(Math.abs(Number(r[barKey]) || 0) / maxAbs) * 50}%` }} />
                  </span>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="panel-foot">{rows.length} rows from the {LEAGUE_NAME[league]} board export.</p>
    </section>
  )
}
