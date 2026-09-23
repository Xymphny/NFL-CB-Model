import { usePropLedger } from '../data'
import { pct } from '../format'

/* NFL is the only league with a player model: the props projection engine.
 * It runs in watch mode -- its opinions are graded here and never sized --
 * so this view is its ledger, read from /data/prop_grades/summary.json
 * (deploy/grade_props.py). Claimed and actual are shown side by side; the
 * site does not compute a gap between them. */

export default function Players() {
  const { data, loading, error } = usePropLedger()
  return (
    <section className="panel">
      <header className="panel-head">
        <h1>Player props engine</h1>
        <p>Watch mode. The engine publishes an opinion on every prop it can price, and none of
          them is sized. They count only once graded here.</p>
      </header>
      {loading && <p className="panel-foot">Loading the ledger…</p>}
      {!loading && (error || !data?.n_claims) && (
        <div className="empty">
          <h2>No graded claims yet</h2>
          <p>The first ledger lands the Tuesday after a completed week.</p>
        </div>
      )}
      {data?.n_claims > 0 && <EngineLedger data={data} />}
    </section>
  )
}

export function EngineLedger({ data }) {
  return (
    <>
      <p className="panel-lede">
        {data.n_claims} claims graded across week{data.weeks_graded.length > 1 ? 's' : ''} {data.weeks_graded.join(', ')}.
        Claimed is what the engine said would happen; actual is how often it did. A market that
        claims well above what it hits is over-confident, which is what keeps it in watch mode.
      </p>
      <table className="data">
        <thead>
          <tr><th>Market</th><th>Engine</th><th className="num">Claims</th><th className="num">Claimed</th>
            <th className="num">Actual</th><th className="num">Log loss</th><th className="num">Brier</th></tr>
        </thead>
        <tbody>
          {(data.by_market || []).map((m) => (
            <tr key={`${m.engine}-${m.market}`}>
              <td>{m.market.replaceAll('_', ' ')}</td>
              <td>{m.engine || 'baseline'}</td>
              <td className="num">{m.n}</td>
              <td className="num">{pct(m.mean_claimed)}</td>
              <td className="num">{pct(m.actual_rate)}</td>
              <td className="num">{m.log_loss?.toFixed(3)}</td>
              <td className="num">{m.brier?.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {(data.buckets || []).length > 0 && (
        <>
          <h2 className="panel-sub">Calibration by claimed probability</h2>
          <table className="data">
            <thead><tr><th>Claimed band</th><th className="num">Claims</th><th className="num">Claimed</th><th className="num">Actual</th></tr></thead>
            <tbody>
              {data.buckets.map((b) => (
                <tr key={b.lo}>
                  <td>{pct(b.lo, 0)} to {pct(Math.min(b.hi, 1), 0)}</td>
                  <td className="num">{b.n}</td>
                  <td className="num">{pct(b.claimed)}</td>
                  <td className="num">{pct(b.actual)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  )
}
