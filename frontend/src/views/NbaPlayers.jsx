import { useNbaPlayers } from '../data'
import { age, dateLabel } from '../format'

/* NBA availability and minutes, from data/site/players_nba.json
 * (scripts/export_players.py). NOT a model input: the NBA model rates teams,
 * and this is what it cannot see. The view shows the export and computes
 * nothing -- no adjusted line, no implied edge.
 *
 * Teams on the current slate come first, in tip-off order, because that is
 * where a status change can overturn a suggestion. Status is always a word,
 * never colour alone. */

const STATUS_CLASS = {
  Out: 'st-out', Doubtful: 'st-doubtful', Questionable: 'st-questionable',
  Probable: 'st-probable', 'Day-To-Day': 'st-dtd',
}

/* A return date can be a season away, so it carries the month and year,
 * not a weekday. */
function backBy(iso) {
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number)
  return new Date(Date.UTC(y, m - 1, d, 12)).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC',
  })
}

function slateTeams(board) {
  const seen = []
  for (const g of [...(board?.games || [])].sort((a, b) => (a.start || '').localeCompare(b.start || ''))) {
    for (const t of [g.away, g.home]) if (t && !seen.includes(t)) seen.push(t)
  }
  return seen
}

function Injuries({ rows }) {
  if (!rows.length) return <p className="pl-none">No one listed on the injury report.</p>
  return (
    <ul className="pl-inj">
      {rows.map((r) => (
        <li key={r.player}>
          <span className={`pl-status ${STATUS_CLASS[r.status] || ''}`}>{r.status}</span>
          <span className="pl-name">{r.player}{r.position && <small>{r.position}</small>}</span>
          <span className="pl-why">
            {r.injury || 'Not specified'}
            {r.return_date && <small>expected back {backBy(r.return_date)}</small>}
          </span>
        </li>
      ))}
    </ul>
  )
}

function Rotation({ team, window: n }) {
  if (!team.rotation.length) return null
  return (
    <table className="data data-narrow pl-rot">
      <caption>Last {team.games_in_window} game{team.games_in_window === 1 ? '' : 's'}</caption>
      <thead>
        <tr><th>Player</th><th className="num">Min / game</th><th className="num">Played</th>
          <th className="num">Starts</th><th className="num">Missed</th></tr>
      </thead>
      <tbody>
        {team.rotation.map((r) => (
          <tr key={r.player}>
            <td>{r.player}{r.position && <small>{r.position}</small>}</td>
            <td className="num">{r.min_per_game.toFixed(1)}</td>
            <td className="num">{r.played} / {r.of}</td>
            <td className="num">{r.starts}</td>
            <td className="num">{r.missed || '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function TeamBlock({ team, window: n, onSlate }) {
  return (
    <article className={`pl-team${onSlate ? ' on-slate' : ''}`}>
      <header>
        <h3>{team.name}</h3>
        <span className="pl-code">{team.team}</span>
        {onSlate && <span className="pl-flag">On the slate</span>}
      </header>
      <Injuries rows={team.injuries} />
      <Rotation team={team} window={n} />
    </article>
  )
}

export default function NbaPlayers({ board }) {
  const { data, loading, error } = useNbaPlayers()
  const onSlate = slateTeams(board)
  const byCode = Object.fromEntries((data?.teams || []).map((t) => [t.team, t]))
  const first = onSlate.map((c) => byCode[c]).filter(Boolean)
  const rest = (data?.teams || [])
    .filter((t) => !onSlate.includes(t.team) && (t.injuries.length || t.rotation.length))

  return (
    <section className="panel">
      <header className="panel-head">
        <h1>Players</h1>
        <p>Who is available and who carries the minutes, team by team.</p>
      </header>
      <div className="stamp stamp-cap pl-banner" role="note">
        <p><b>Not a model input.</b> {data?.note?.replace(/^Not a model input\.\s*/, '') ||
          'The NBA model rates teams, not players. This is what it cannot see.'} When the market
          moves on news and the model does not, that gap is the news, not an edge.</p>
      </div>

      {loading && <p className="panel-foot">Loading the injury report…</p>}
      {!loading && (error || !data) && (
        <div className="empty">
          <h2>No player file yet</h2>
          <p>It is written by the live-inputs job twice a day. If this persists, that job has not
            run since this tab was added.</p>
        </div>
      )}

      {data && (
        <>
          <p className="panel-lede pl-asof">
            Injury report as of {data.injuries_as_of ? `${dateLabel(data.injuries_as_of.slice(0, 10))} (${age(data.injuries_as_of)})` : '—'}
            {' · '}
            {data.minutes_through
              ? `minutes through ${dateLabel(data.minutes_through.slice(0, 10))}`
              : data.minutes_note}
          </p>
          {data.unmatched_team_names?.length > 0 && (
            <div className="stamp stamp-refused" role="status">
              <p>ESPN listed team names the site does not recognise, so their players are missing
                here: {data.unmatched_team_names.join(', ')}.</p>
            </div>
          )}

          {first.length > 0 && (
            <>
              <h2 className="panel-sub">On the current slate</h2>
              <div className="pl-grid">
                {first.map((t) => <TeamBlock key={t.team} team={t} window={data.window} onSlate />)}
              </div>
            </>
          )}
          {!first.length && board?.next_slate && (
            <p className="panel-foot">No NBA games on the current slate. Next slate:{' '}
              {dateLabel(board.next_slate.date)}, {board.next_slate.games} game{board.next_slate.games === 1 ? '' : 's'}.</p>
          )}

          <h2 className="panel-sub">{first.length ? 'Everyone else' : 'By team'}</h2>
          {rest.length === 0
            ? <p className="pl-none">No other team has anyone listed.</p>
            : <div className="pl-grid">
                {rest.map((t) => <TeamBlock key={t.team} team={t} window={data.window} />)}
              </div>}
        </>
      )}
    </section>
  )
}
