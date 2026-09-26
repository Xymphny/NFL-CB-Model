import { line as fmtLine, pct } from '../format'
import { StateGlyph } from '../Glyph'

/* Per-game context from the export (scripts/board_context.py and
 * board_detail.py). Everything here formats and lays out: which injuries
 * show, how many, in what order, and whether each item is in the price all
 * arrive decided. A missing source renders as "no report" or "unknown",
 * never as an empty list that reads healthy. */

const STATUS_CLASS = {
  Out: 'st-out', 'Injured Reserve': 'st-out', IR: 'st-out', '60-Day IL': 'st-out',
  '15-Day IL': 'st-out', '10-Day IL': 'st-out', '7-Day IL': 'st-out', Suspended: 'st-out',
  Doubtful: 'st-doubtful', Questionable: 'st-questionable', Probable: 'st-probable',
  'Day-To-Day': 'st-dtd',
}

export function InPrice({ yes }) {
  return <span className={`inprice ${yes ? 'yes' : 'no'}`}>{yes ? 'In the price' : 'Not in the price'}</span>
}

/* "NOT IN THE PRICE · <TEAM> <ROLE>", one sentence, and where it came from.
 * On any card in any tier; nothing when there is nothing to flag. */
export function KeyPlayerStrip({ items, onPlayers }) {
  if (!items?.length) return null
  return (
    <div className="kp-strips">
      {items.map((k) => (
        <p className="kp-strip" key={`${k.team}-${k.text}`} role="note">
          <StateGlyph state="degraded" size={11} />
          <b>NOT IN THE PRICE · {k.team} {k.role.toUpperCase()}</b>
          <span>{k.text}.</span>
          <small>Source: {k.source}.{onPlayers && k.role === 'starter' && (
            <> <button className="linkish" onClick={onPlayers}>Players tab</button></>)}</small>
        </p>
      ))}
    </div>
  )
}

function TeamInjuries({ team, t }) {
  if (!t || t.status === 'no report') {
    return <div className="inj-team"><h4>{team}</h4><p className="inj-none">No report: unknown, not healthy.</p></div>
  }
  const counts = Object.entries(t.counts || {})
  return (
    <div className="inj-team">
      <h4>{team}</h4>
      {t.shown?.length > 0 && (
        <ul>
          {t.shown.map((p) => (
            <li key={p.player}>
              <span className="inj-name">{p.player}{p.position && <small>{p.position}</small>}</span>
              <span className={`inj-status ${STATUS_CLASS[p.status] || ''}`}>{p.status}</span>
            </li>
          ))}
        </ul>
      )}
      {t.more > 0 && <p className="inj-more">+{t.more} more</p>}
      {!t.shown?.length && counts.length > 0 && (
        <p className="inj-counts">{counts.map(([s, n]) => `${s} ${n}`).join(' · ')}</p>
      )}
      {!counts.length && <p className="inj-none">No one listed</p>}
    </div>
  )
}

export function InjuryReport({ g }) {
  const inj = g.context?.injuries
  if (!inj) return null
  return (
    <section className="inj" aria-label="Injury report">
      <h3>Injury report <InPrice yes={inj.in_price} />{inj.source && <small>{inj.source}</small>}</h3>
      {inj.status === 'no report' ? (
        <p className="inj-none">No report available for this game. Unknown, not healthy.</p>
      ) : (
        <div className="inj-cols">
          <TeamInjuries team={g.away} t={inj.away} />
          <TeamInjuries team={g.home} t={inj.home} />
        </div>
      )}
      {inj.note && <p className="ctx-note">{inj.note}</p>}
    </section>
  )
}

function weatherText(w) {
  if (!w || w.status === 'no report') return 'No report'
  if (w.status === 'indoors') return 'Indoors'
  if (w.status === 'forecast') {
    return [w.temp_f != null && `${Math.round(w.temp_f)}°`, w.wind_mph != null && `wind ${Math.round(w.wind_mph)} mph`,
            w.precip_prob != null && `${w.precip_prob}% rain`].filter(Boolean).join(' · ') || 'No report'
  }
  return [w.condition, w.temp_f != null && `${w.temp_f}°`, w.wind].filter(Boolean).join(' · ')
}

function venueText(v) {
  if (!v) return 'Venue unknown'
  if (v.neutral && !v.name) return 'Neutral site'
  const place = [v.city, v.state].filter(Boolean).join(', ')
  const turf = [v.roof && v.roof !== 'outdoors' ? v.roof : v.roof === 'outdoors' ? 'open air' : null,
                v.surface].filter(Boolean).join(' · ')
  return [v.name || 'Venue unknown', place, turf, v.neutral ? 'neutral site' : null].filter(Boolean).join(' · ')
}

function restText(r, g) {
  if (!r) return null
  if (r.home_days != null && r.away_days != null) {
    return r.home_days === r.away_days ? `Both on ${r.home_days} days`
      : `${g.away} ${r.away_days} days · ${g.home} ${r.home_days} days`
  }
  const bits = []
  ;['away', 'home'].forEach((s) => {
    const team = s === 'home' ? g.home : g.away
    if (r[`${s}_back_to_back`]) bits.push(`${team} on a back-to-back`)
    else if (r[`${s}_rest_days`] != null) bits.push(`${team} ${r[`${s}_rest_days`]} days' rest`)
    if (r[`${s}_day_after_night`]) bits.push(`${team} day game after a night game`)
  })
  return bits.join(' · ') || null
}

/* Venue, weather, rest and travel: one line each, each tagged. */
export function ContextLines({ g }) {
  const c = g.context || {}
  if (!('venue' in c) && !('weather' in c)) return null
  const rest = restText(c.rest, g)
  return (
    <dl className="ctx-lines">
      <dt>Venue</dt><dd>{venueText(c.venue)}{c.venue && <InPrice yes={c.venue.in_price} />}</dd>
      <dt>Weather</dt><dd>{weatherText(c.weather)}{c.weather && <InPrice yes={c.weather.in_price} />}</dd>
      {rest && <><dt>Rest</dt><dd>{rest}<InPrice yes={c.rest.in_price} /></dd></>}
      {c.travel && (
        <><dt>Travel</dt><dd>{c.travel.away_miles != null ? `${g.away} ~${c.travel.away_miles.toLocaleString('en-US')} mi` : 'Unknown'}
          <InPrice yes={c.travel.in_price} /></dd></>
      )}
      {c.key_player_note && <><dt>Goalies</dt><dd>{c.key_player_note}</dd></>}
    </dl>
  )
}

/* Line since open. Positions only: the points are the export's, in the
 * order they were captured; nothing between them is drawn in. */
export function LineSpark({ g, market }) {
  const lh = g.line_history
  if (!lh) return null
  const ml = market?.market === 'moneyline' || g.fair_line == null
  const pts = (lh.points || []).map((p) => (ml ? p.p_market : p.line)).filter((v) => v != null)
  const fair = ml ? g.fair_p : g.fair_line
  if (!pts.length) return <p className="spark-none">No capture holds this game yet.</p>
  const all = fair != null ? [...pts, fair] : pts
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1
  const W = 200, H = 40
  const y = (v) => (4 + ((hi - v) / span) * (H - 8)).toFixed(1)
  const x = (i) => (pts.length === 1 ? W / 2 : (i / (pts.length - 1)) * W).toFixed(1)
  const fmt = (v) => (ml ? pct(v) : fmtLine(v))
  const closeIdx = lh.close_at ? lh.points.findIndex((p) => p.captured_at === lh.close_at) : -1
  return (
    <figure className="spark" aria-label={`Line since open: ${fmt(pts[0])} to ${fmt(pts[pts.length - 1])}${fair != null ? `; model ${fmt(fair)}` : ''}`}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        {fair != null && <line x1="0" x2={W} y1={y(fair)} y2={y(fair)} className="spark-fair" />}
        {pts.length > 1
          ? <polyline points={pts.map((v, i) => `${x(i)},${y(v)}`).join(' ')} className="spark-line" />
          : <circle cx={x(0)} cy={y(pts[0])} r="2.5" className="spark-dot" />}
        {closeIdx >= 0 && <circle cx={x(closeIdx)} cy={y(pts[closeIdx])} r="3" className="spark-close" />}
      </svg>
      <figcaption>
        <span>open {fmt(pts[0])}</span>
        {fair != null && <span className="spark-model">model {fmt(fair)}</span>}
        <span>{lh.frozen ? 'close' : 'now'} {fmt(pts[pts.length - 1])}</span>
        <small>{lh.captures === 1 ? '1 capture so far' : `${lh.captures} captures`}
          {lh.missed_windows?.length ? ` · ${lh.missed_windows.length} missed window${lh.missed_windows.length > 1 ? 's' : ''}, not filled in` : ''}</small>
      </figcaption>
    </figure>
  )
}

export function matchupPath(league, gid) {
  return `/${league}/game/${encodeURIComponent(gid)}`
}

export function OpenMatchup({ league, g, onOpen }) {
  return (
    <a className="open-matchup" href={matchupPath(league, g.game_id)}
       onClick={(e) => { e.preventDefault(); onOpen?.(g.game_id) }}>Open matchup</a>
  )
}
