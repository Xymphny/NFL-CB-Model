import { useTeamPage } from '../data'
import {
  LEAGUE_NAME, MARKET_LABEL, TIER_LABEL, american, book as bookName, line as fmtLine, pct, pts,
  startDay, startTime,
} from '../format'
import { StateGlyph } from '../Glyph'
import { ContextLines, InjuryReport, KeyPlayerStrip, LineSpark } from './GameContext'

/* Every number the board has on one game (/<league>/game/<game_id>).
 * The distribution is the export's model.margin_pmf -- bars drawn from its
 * mass, never a pmf computed here. Colouring the bars that cover the line
 * is layout; every figure in the text is read off the export. */

function MarginBars({ g, m }) {
  const pm = g.model?.margin_pmf
  if (!pm) return <p className="panel-foot">No distribution was exported for this game.</p>
  const max = Math.max(...pm.mass)
  const homeLine = m?.market === 'spread' ? (m.side === 'home' ? m.line : -m.line) : 0
  const covers = (k) => (m?.side === 'away' ? k + homeLine < 0 : k + homeLine > 0)
  const at = (k) => pm.mass[k - pm.from]
  const keys = pm.key_numbers || []
  return (
    <figure className="pmf">
      <div className="pmf-bars" role="img"
           aria-label={`Distribution of ${g.home}'s final margin from ${pm.from} to ${pm.to}.${m?.status === 'priced' ? ` The pick covers in ${pct(m.p_model)} of outcomes.` : ''}`}>
        {pm.mass.map((v, i) => {
          const k = pm.from + i
          return (
            <span key={k} title={`${g.home} ${k > 0 ? 'by ' + k : k < 0 ? 'loses by ' + -k : 'ties'}: ${pct(v)}`}
                  className={`pmf-bar${m?.status === 'priced' && covers(k) ? ' covers' : ''}${keys.includes(k) ? ' key' : ''}${k === 0 ? ' zero' : ''}`}
                  style={{ height: `${max ? (v / max) * 100 : 0}%` }} />
          )
        })}
      </div>
      <figcaption>
        <span>{g.away} by {-pm.from}{pm.tail_below > 0.0005 ? '+' : ''}</span>
        <span>even</span>
        <span>{g.home} by {pm.to}{pm.tail_above > 0.0005 ? '+' : ''}</span>
      </figcaption>
      <p className="pmf-notes">
        {g.model?.p_home_win != null && <>{g.home} win: {pct(g.model.p_home_win)}. </>}
        {m?.status === 'priced' && m.market === 'spread' && <>Highlighted: the pick covers, {pct(m.p_model)}. </>}
        {keys.filter((k) => k > 0).map((k) => `Exactly ${k}: ${pct(at(k))}`).join(' · ')}
        {pm.binned && ' Modelled continuously; mass shown per whole point.'}
      </p>
    </figure>
  )
}

function Markets({ g, league }) {
  const markets = Object.values(g.markets || {})
  const totals = g.model?.totals
  return (
    <table className="data d-markets">
      <thead><tr><th>Market</th><th>Side</th><th className="num">Line</th><th>Best price</th>
        <th className="num">Market</th><th className="num">Model</th><th className="num">Gap</th><th>Tier</th></tr></thead>
      <tbody>
        {markets.map((v) => {
          if (v.status !== 'priced') {
            return <tr key={v.market}><th scope="row">{MARKET_LABEL[v.market] || v.market}</th>
              <td colSpan={7}><StateGlyph state="refused" /> {v.refusal}</td></tr>
          }
          const who = v.side === 'home' ? g.home : v.side === 'away' ? g.away : v.side
          return (
            <tr key={v.market}>
              <th scope="row">{MARKET_LABEL[v.market] || v.market}</th>
              <td>{who}</td>
              <td className="num">{v.line == null ? 'ML' : fmtLine(v.line)}</td>
              <td>{american(v.best_price_american)} <small>{bookName(v.best_book)} · {v.books} books</small></td>
              <td className="num">{pct(v.p_market)}</td>
              <td className="num">{pct(v.p_model)}</td>
              <td className="num">{pts(v.edge)}</td>
              <td>{TIER_LABEL[v.tier]}</td>
            </tr>
          )
        })}
        {totals && !totals.priced && (
          <tr><th scope="row">Total</th><td colSpan={7} className="muted">{totals.reason}</td></tr>
        )}
        {!markets.length && <tr><td colSpan={8} className="muted">No market price captured for {LEAGUE_NAME[league]} on this game yet.</td></tr>}
      </tbody>
    </table>
  )
}

/* Side by side from the team page's exported ranks. */
const SIDE_ROWS = {
  nfl: [['Model rating', (t) => t.rating], ['Offence VOA', (t) => t.model?.offense_voa],
        ['Defence VOA', (t) => t.model?.defense_voa], ['Special teams VOA', (t) => t.model?.special_teams_voa],
        ['EPA per play', (t) => t.efficiency?.epa_per_play], ['EPA allowed', (t) => t.efficiency?.epa_per_play_allowed],
        ['Success rate', (t) => t.efficiency?.success_rate], ['Success allowed', (t) => t.efficiency?.success_rate_allowed],
        ['Turnover margin', (t) => t.ball_security?.turnover_margin],
        ['Red-zone pts/trip', (t) => t.ball_security?.red_zone_points_per_trip]],
  nba: [['Net per game', (t) => t.net], ['Points scored', (t) => t.pts_for],
        ['Points allowed', (t) => t.pts_against], ['Model rating', (t) => t.rating]],
}

function fmtStat(s) {
  if (!s || s.value == null) return '—'
  const v = s.value
  return Math.abs(v) < 1 && v !== 0 ? v.toFixed(3) : (Math.round(v * 10) / 10).toString()
}

function SideBySide({ league, g }) {
  const page = useTeamPage(league)
  const rows = SIDE_ROWS[league]
  if (!rows) return null
  const teams = page.data?.teams || []
  const find = (code) => teams.find((t) => (t.abbr || t.team) === code)
  const a = find(g.away), h = find(g.home)
  if (!a || !h) return <p className="panel-foot">Team ranks are not available for this matchup.</p>
  const of = league === 'nfl' ? 32 : 30
  return (
    <table className="data side-by-side">
      <caption>Side by side · ranks among {of} · only the model rating moves the price</caption>
      <thead><tr><th className="num">{g.away}</th><th className="num">Rank</th><th>Stat</th><th className="num">Rank</th><th className="num">{g.home}</th></tr></thead>
      <tbody>
        {rows.map(([label, get]) => {
          const av = get(a), hv = get(h)
          const better = av?.rank != null && hv?.rank != null ? (av.rank < hv.rank ? 'a' : hv.rank < av.rank ? 'h' : null) : null
          return (
            <tr key={label}>
              <td className={`num${better === 'a' ? ' better' : ''}`}>{fmtStat(av)}</td>
              <td className="num muted">{av?.rank ?? '—'}</td>
              <th scope="row">{label}</th>
              <td className="num muted">{hv?.rank ?? '—'}</td>
              <td className={`num${better === 'h' ? ' better' : ''}`}>{fmtStat(hv)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export default function Matchup({ league, board, gameId, onBack }) {
  const g = (board?.games || []).find((x) => x.game_id === gameId)
  if (!board) return <p className="panel-foot">Loading the board…</p>
  if (!g) {
    return (
      <section className="panel">
        <p><button className="linkish" onClick={onBack}>← {LEAGUE_NAME[league]} board</button></p>
        <div className="empty"><h2>This game is not on the current board</h2>
          <p>It may have finished, or the board has moved on to a new slate.</p></div>
      </section>
    )
  }
  const m = g.markets?.[g.headline]
  return (
    <section className="panel matchup">
      <p><button className="linkish" onClick={onBack}>← {LEAGUE_NAME[league]} board</button></p>
      <header className="panel-head">
        <h1>{g.away_name || g.away} @ {g.home_name || g.home}</h1>
        <p>{startDay(g.start)} {startTime(g.start)}{g.context?.venue?.name ? ` · ${g.context.venue.name}` : ''}</p>
      </header>
      <KeyPlayerStrip items={g.context?.key_player} />
      <div className="mu-grid">
        <div>
          <h2 className="panel-sub">What the model sees: {g.home}'s final margin</h2>
          <MarginBars g={g} m={m} />
        </div>
        <div className="mu-pick">
          {m?.status === 'priced' ? (
            <>
              <p className="mu-tier">{TIER_LABEL[m.tier]}<small> · the model's suggestion, {m.unsized_reason ? 'unsized, paper-traded' : 'sized'}</small></p>
              <p className="mu-sel">{m.side === 'home' ? g.home : g.away} {m.line == null ? 'ML' : fmtLine(m.line)}</p>
              <p className="muted">{american(m.best_price_american)} · {bookName(m.best_book)} · best of {m.books} books</p>
            </>
          ) : <p className="muted">{g.started ? 'Started. Nothing is priced after kickoff.' : g.refusal || 'Not priced against the market.'}</p>}
        </div>
      </div>
      <Markets g={g} league={league} />
      <SideBySide league={league} g={g} />
      <div className="d-v2">
        <div className="d-v2-col"><h3>Line since open</h3><LineSpark g={g} market={m} /><ContextLines g={g} /></div>
        <div className="d-v2-col"><InjuryReport g={g} /></div>
      </div>
    </section>
  )
}
