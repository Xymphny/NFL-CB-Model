import { LEAGUES } from '../data'
import { LEAGUE_CODE, LEAGUE_NAME, book as bookName, line as fmtLine, pct, pts } from '../format'

/* The record is the paper-trading ledger, exported by scripts/export_record.py:
 * one row per game, the model's preferred side at the price it saw, settled
 * against the final score. Paper trades, never stakes -- until a league has
 * 150 of them and a grade against the market says otherwise. */

export default function Record({ league, record }) {
  const { data, loading, error } = record
  if (loading) return <p className="panel-foot">Loading the record…</p>
  if (error || !data) {
    return <div className="empty"><h2>The record could not be loaded</h2><p>It is exported with each board; reload after the next deploy.</p></div>
  }
  const r = data.leagues[league]
  return (
    <section className="panel">
      <header className="panel-head">
        <h1>The record</h1>
        <p>Paper trades: every game the core priced, the model's preferred side at the price it
          saw, settled against the final score. Nothing here was staked. A league's weight is
          regraded from these once it has {data.floor} settled games.</p>
        <p>CLV is the game's earliest trade against its closing line, in probability points,
          after the vig. When the line moved, the close is valued at the trade's line on the
          league's own margin distribution, key numbers included, so beating a close by a point
          counts as the point it was.</p>
      </header>

      <table className="data">
        <thead>
          <tr><th>League</th><th className="num">Settled</th><th>Toward the floor</th>
            <th className="num">Preferred side won</th><th className="num">Break-even</th>
            <th className="num">Mean CLV</th><th className="num">Market grade</th></tr>
        </thead>
        <tbody>
          {LEAGUES.map((l) => {
            const x = data.leagues[l]
            return (
              <tr key={l} className={l === league ? 'here' : undefined}>
                <th scope="row">{LEAGUE_CODE[l]}</th>
                <td className="num">{x.settled_games}</td>
                <td>
                  <span className="meter meter-inline" role="meter" aria-valuemin={0} aria-valuemax={data.floor}
                        aria-valuenow={x.settled_games} aria-label={`${x.settled_games} of ${data.floor}`}>
                    <span style={{ width: `${x.progress_to_floor * 100}%` }} />
                  </span>
                  <small>{x.settled_games}/{data.floor}</small>
                </td>
                <td className="num">{x.preferred_side_hit_rate != null ? `${pct(x.preferred_side_hit_rate)} (${x.preferred_side_wins}/${x.settled_games})` : '—'}</td>
                <td className="num">{pct(x.break_even_at_minus_110)}</td>
                <td className="num">{x.mean_clv_prob_points != null ? `${pts(x.mean_clv_prob_points, 2)} pts` : '—'}</td>
                <td className="num">{x.grade.weight.toFixed(2)}{x.grade.source ? ` (${x.grade.source})` : ''}</td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <h2 className="panel-sub">{LEAGUE_NAME[league]}: most recent settled games</h2>
      {!r.recent?.length ? (
        <div className="empty">
          <h2>Nothing settled yet</h2>
          <p>Paper trades start when the odds capture runs; each settles the morning after its
            game. The first {LEAGUE_NAME[league]} rows appear here then.</p>
        </div>
      ) : (
        <table className="data">
          <thead>
            <tr><th>Priced</th><th>Pick</th><th className="num">Price</th><th className="num">Model</th>
              <th className="num">Market</th><th>Result</th><th className="num">Score</th><th className="num">CLV</th></tr>
          </thead>
          <tbody>
            {r.recent.map((x, i) => (
              <tr key={i}>
                <td>{String(x.at).slice(0, 10)}</td>
                <td>{x.selection} {x.market === 'h2h' ? 'ML' : fmtLine(x.line)}</td>
                <td className="num">{x.price_decimal ? `${x.price_decimal.toFixed(2)}` : '—'} <small>{bookName(x.book)}</small></td>
                <td className="num">{pct(x.p_model)}</td>
                <td className="num">{pct(x.p_market)}</td>
                <td><span className={`result r-${x.result}`}>{x.result === 'win' ? 'Won' : 'Lost'}</span></td>
                <td className="num">{x.score}</td>
                <td className="num">{x.clv_prob_points != null ? pts(x.clv_prob_points, 2) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

