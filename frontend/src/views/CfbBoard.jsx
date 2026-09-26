import { TIER_LABEL, line as fmtLine, pct, startTime } from '../format'
import { TeamLogo } from './GameContext'

/* The college board's own parts (dashboard v2 item 8), around the shared
 * board: the early-season banner, the slate by kickoff window, and the
 * not-priced panel. Windows, reasons and counts are the export's
 * (scripts/export_board.py cfb_window, not_priced); the site groups by the
 * label it is given and counts nothing. */

export function EarlySeasonBanner({ board }) {
  if (!board.early_season) return null
  const through = board.freshness?.ratings_through_week
  return (
    <div className="stamp stamp-cap early" role="note">
      <b>EARLY SEASON</b>
      <p>College ratings have {through != null ? `${through} week${through === 1 ? '' : 's'}` : 'only a few weeks'} of
        games behind them, and 130+ teams to spread them across. Spreads run wider and the model disagrees with
        the market by more than in any other league, so its tier bands are wider too: a Play here needs
        a {board.tiers?.play != null ? (board.tiers.play * 100).toFixed(1) : '—'}-point gap in cover probability. Every
        suggestion is paper-traded; nothing is sized.</p>
    </div>
  )
}

const WINDOW_ORDER = ['Friday night', 'Saturday noon', 'Saturday afternoon', 'Saturday prime time', 'Saturday late']

export function ByWindow({ board }) {
  const games = (board.games || []).filter((g) => (g.markets?.[g.headline] || {}).status === 'priced')
  if (!games.length) return null
  const names = [...WINDOW_ORDER, ...new Set(games.map((g) => g.window).filter((w) => w && !WINDOW_ORDER.includes(w)))]
  return (
    <section className="group by-window" aria-label="By kickoff window">
      <header className="group-head"><h2>By kickoff window</h2>
        <span className="group-note">games with a line · Eastern time</span></header>
      <table className="data">
        <thead><tr><th>Kickoff</th><th>Matchup</th><th>Market</th><th className="num">Model</th>
          <th className="num">Cover · model</th><th>Read</th></tr></thead>
        {names.map((w) => {
          const rows = games.filter((g) => g.window === w)
          if (!rows.length) return null
          return (
            <tbody key={w}>
              <tr className="win-head"><th colSpan={6} scope="rowgroup">{w}</th></tr>
              {rows.map((g) => {
                const m = g.markets[g.headline]
                const who = m.side === 'home' ? g.home : g.away
                return (
                  <tr key={g.game_id}>
                    <td>{startTime(g.start)}</td>
                    <td><TeamLogo src={g.away_logo} size={16} />{g.away} @ <TeamLogo src={g.home_logo} size={16} />{g.home}{g.check_flag && <small className="flag-word"> · check</small>}</td>
                    <td>{who} {fmtLine(m.line)}</td>
                    <td className="num">{g.fair_line != null ? `${who} ${fmtLine(g.fair_line)}` : '—'}</td>
                    <td className="num">{pct(m.p_model)} <small>vs {pct(m.p_market)}</small></td>
                    <td>{TIER_LABEL[m.tier]}</td>
                  </tr>
                )
              })}
            </tbody>
          )
        })}
      </table>
    </section>
  )
}

export function CfbRail({ board }) {
  const np = board.status?.not_priced || []
  return (
    <aside className="rail rail-extra" aria-label="Not priced, with the reason">
      <h2>Not priced, with the reason</h2>
      <dl className="np-list">
        {np.map((x) => (
          <div key={x.key} className="np-row">
            <dt>{x.label}</dt><dd><b>{x.games}</b></dd>
            <p className="rail-note">{x.examples?.length ? `${x.examples.join(', ')}${x.games > x.examples.length ? ' and more' : ''}. ` : ''}{x.note}</p>
          </div>
        ))}
      </dl>
      <h2>What the college model does not see</h2>
      <p className="rail-note">Quarterback changes, roster churn from the transfer portal, injuries and weather. A
        starting QB who is out or returning gets a Not in the price flag on his game, from the hand-kept
        override file. ESPN's injury report covers only some college teams, so a missing report means
        unknown, not healthy.</p>
    </aside>
  )
}
