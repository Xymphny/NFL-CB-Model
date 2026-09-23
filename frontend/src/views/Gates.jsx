import { StateGlyph } from '../Glyph'
import { LEAGUE_CODE, TIER_LABEL, pct, signed } from '../format'

/* Every check that decides what ships, from data/site/gates.json -- read
 * from the fitted artifacts' own grades and docs/decisions by
 * scripts/export_record.py. Nothing on this page is written by hand. */

export default function Gates({ league, gates }) {
  const { data, loading, error } = gates
  if (loading) return <p className="panel-foot">Loading the gates…</p>
  if (error || !data) return <div className="empty"><h2>The gates could not be loaded</h2><p>They are exported with each board.</p></div>
  const mine = (x) => x.league === league
  return (
    <section className="panel">
      <header className="panel-head">
        <h1>Gates</h1>
        <p>What each piece of the model had to pass before it could price anything, read from the
          artifacts that graded it. A gate that fails is refused by the code, and shows here the same way.</p>
      </header>

      <h2 className="panel-sub">Stakes: graded against closing prices</h2>
      <table className="data">
        <thead><tr><th>League</th><th className="num">Staking weight</th><th className="num">Measured</th>
          <th className="num">Games</th><th>Tier bands</th><th>Preferred side won, by band</th></tr></thead>
        <tbody>
          {data.market.map((m) => (
            <tr key={m.league} className={mine(m) ? 'here' : undefined}>
              <th scope="row">{LEAGUE_CODE[m.league]}</th>
              <td className="num">{m.staking_weight.toFixed(2)}</td>
              <td className="num">{m.w_hat != null ? `${signed(m.w_hat, 3)} ± ${m.se.toFixed(3)}` : 'not graded'}</td>
              <td className="num">{m.n?.toLocaleString() ?? '—'}</td>
              <td>{m.tiers_source ? `${m.tiers_provisional ? 'Provisional' : 'Calibrated'}, ${m.tiers_source.replace('borrowed:', 'borrowed from ').replace(/from (\w+)$/, (_, x) => `from ${x.toUpperCase()}`)}` : '—'}</td>
              <td className="bands">
                {m.tiers_by_band
                  ? Object.entries(m.tiers_by_band).map(([t, b]) => (
                      <span key={t}>{TIER_LABEL[t]} {pct(b.preferred_side_hit_rate)} <small>n {b.n}</small></span>
                    ))
                  : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="panel-foot">Break-even at −110 is {pct(data.market[0]?.break_even_at_minus_110)}. The staking weight is the
        one-sided 95% lower bound of the measured weight: zero unless the data rule zero out (ADR 0024).</p>

      <h2 className="panel-sub">Fitted components</h2>
      <table className="data">
        <thead><tr><th>League</th><th>Component</th><th>Held-out grade</th><th className="num">t</th><th>Source</th></tr></thead>
        <tbody>
          {data.fitted.map((f) => (
            <tr key={f.file} className={mine(f) ? 'here' : undefined}>
              <th scope="row">{LEAGUE_CODE[f.league]}</th>
              <td>{f.gate}</td>
              <td><span className="verdict"><StateGlyph state={f.passed ? 'up' : 'refused'} />{f.passed ? 'Passed' : f.missing ? 'Missing' : 'Failed; refused by its loader'}</span></td>
              <td className="num">{f.t != null ? f.t.toFixed(2) : '—'}</td>
              <td><code>{f.file}</code></td>
            </tr>
          ))}
          {data.validation.map((v) => (
            <tr key={v.file} className={mine(v) ? 'here' : undefined}>
              <th scope="row">{LEAGUE_CODE[v.league]}</th>
              <td>{v.gate}</td>
              <td><span className="verdict"><StateGlyph state={v.supported ? 'up' : 'refused'} />{v.supported ? 'Supported' : (v.action === 'WITHHELD' ? 'Withheld' : v.action)}</span></td>
              <td className="num">—</td>
              <td><code>{v.file}</code></td>
            </tr>
          ))}
        </tbody>
      </table>

      {(data.rules || []).length > 0 && (
        <>
          <h2 className="panel-sub">Rules applied to the board</h2>
          <ul className="rules">
            {data.rules.map((r) => (
              <li key={r.rule}><b>{LEAGUE_CODE[r.league]} · {r.rule}.</b> {r.text} <code>{r.file}</code></li>
            ))}
          </ul>
        </>
      )}

      <h2 className="panel-sub">Decision records</h2>
      <ol className="decisions">
        {data.decisions.slice().reverse().map((d) => (
          <li key={d.id}>
            <span className="d-id">{d.id}</span>
            <span className="d-title">{d.title.replace(/^\d{4}\.\s*/, '')}</span>
            <span className="d-meta">{d.status}{d.superseded_by ? `, superseded by ${d.superseded_by.slice(0, 4)}` : ''} · {d.date}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
