import { useState } from 'react'
import { useOutlook, useTeamPage } from '../data'
import { dateLabel, pct, startDay, startTime } from '../format'

/* Team pages from scripts/export_teams.py and export_outlook.py. Every rank
 * is exported; a null stat shows "--" with no rank. The site orders nothing
 * except the list, by the exported rank. */

function Stat({ label, s, fmt = (v) => v }) {
  const has = s && s.value != null
  return (
    <div className="tp-stat">
      <dt>{label}</dt>
      <dd><b>{has ? fmt(s.value) : '—'}</b>{has && s.rank != null && <small>#{s.rank}</small>}</dd>
    </div>
  )
}

const f3 = (v) => (v > 0 ? '+' : '') + v.toFixed(3)
const f1 = (v) => v.toFixed(1)
const pc = (v) => pct(v, 1)

/* "The model on this team": numbers only from the floor up. */
function ModelOnTeam({ rec, team }) {
  const r = rec?.teams?.[team]
  const floor = rec?.team_floor ?? 10
  const n = r?.settled ?? 0
  return (
    <section className="tp-block">
      <h3>The model on this team</h3>
      {n >= floor ? (
        <dl className="tp-stats">
          <div className="tp-stat"><dt>Settled paper trades</dt><dd><b>{n}</b></dd></div>
          <div className="tp-stat"><dt>Preferred side won</dt><dd><b>{r.wins} of {n}</b></dd></div>
          <div className="tp-stat"><dt>Mean CLV after vig</dt><dd><b>{r.mean_clv == null ? '—' : pct(r.mean_clv, 2)}</b></dd></div>
        </dl>
      ) : <p className="muted">{n} of {floor} settled. Numbers appear at {floor}.</p>}
    </section>
  )
}

function TeamList({ items, sel, onSel, label, sub }) {
  return (
    <ol className="tp-list" aria-label={label}>
      {items.map((t) => (
        <li key={t.key}>
          <button className={`tp-row${sel === t.key ? ' on' : ''}`} onClick={() => onSel(t.key)} aria-pressed={sel === t.key}>
            <span className="tp-rank">{t.rank ?? '—'}</span>
            <span className="tp-name">{t.name}</span>
            <span className="tp-sub">{sub(t)}</span>
          </button>
        </li>
      ))}
    </ol>
  )
}

export function NflTeams({ record }) {
  const page = useTeamPage('nfl')
  const outlook = useOutlook()
  const teams = page.data?.teams || []
  const [sel, setSel] = useState(null)
  if (page.loading) return <p className="panel-foot">Loading teams…</p>
  if (!teams.length) return <div className="empty"><h2>No team page yet</h2><p>It is written with each ratings snapshot.</p></div>
  const sorted = [...teams].sort((a, b) => (a.rating.rank ?? 99) - (b.rating.rank ?? 99))
  const t = teams.find((x) => x.abbr === (sel || sorted[0].abbr))
  const po = outlook.data?.playoff_pct?.[t.abbr]
  const rec = record?.data?.leagues?.nfl
  return (
    <section className="panel teams-v2">
      <header className="panel-head"><h1>Teams</h1>
        <p>Ratings snapshot {page.data.ratings_version} · ranked of 32 · click a team.</p></header>
      <div className="tp-grid">
        <TeamList label="NFL teams by model rating" sel={t.abbr} onSel={setSel}
                  items={sorted.map((x) => ({ key: x.abbr, rank: x.rating.rank, name: x.name || x.abbr }))}
                  sub={(x) => { const r = teams.find((y) => y.abbr === x.key).record; return r ? `${r.w}-${r.l}${r.t ? `-${r.t}` : ''}` : '' }} />
        <article className="tp-panel" aria-label={`${t.name} detail`}>
          <header>
            <h2>{t.name}</h2>
            <p>{t.division} · {t.division_place ? `${t.division_place} in the division` : ''} · QB1 {t.qb1 || 'unknown'}</p>
            <p className="tp-rating">Model rating <b>{t.rating.value == null ? '—' : f3(t.rating.value)}</b>
              {t.rating.rank && <small>#{t.rating.rank} of {t.rating.of}</small>}
              {t.rating.p05 != null && <small>90% range {f3(t.rating.p05)} to {f3(t.rating.p95)}</small>}</p>
          </header>
          <section className="tp-block"><h3>Model rating</h3><dl className="tp-stats">
            <Stat label="Offence VOA" s={t.model.offense_voa} fmt={f3} />
            <Stat label="Defence VOA (lower is better)" s={t.model.defense_voa} fmt={f3} />
            <Stat label="Special teams VOA" s={t.model.special_teams_voa} fmt={f3} /></dl></section>
          <section className="tp-block"><h3>Efficiency</h3><dl className="tp-stats">
            <Stat label="EPA per play" s={t.efficiency.epa_per_play} fmt={f3} />
            <Stat label="EPA allowed" s={t.efficiency.epa_per_play_allowed} fmt={f3} />
            <Stat label="Success rate" s={t.efficiency.success_rate} fmt={pc} />
            <Stat label="Success allowed" s={t.efficiency.success_rate_allowed} fmt={pc} /></dl></section>
          <section className="tp-block"><h3>Ball security and schedule</h3><dl className="tp-stats">
            <Stat label="Turnover margin" s={t.ball_security.turnover_margin} fmt={(v) => (v > 0 ? '+' : '') + v} />
            <Stat label="Red-zone points per trip" s={t.ball_security.red_zone_points_per_trip} fmt={f1} />
            <Stat label="Schedule played (1 = hardest)" s={t.schedule.played} fmt={f3} />
            <Stat label="Schedule remaining (1 = hardest)" s={t.schedule.remaining} fmt={f3} /></dl></section>
          <section className="tp-block"><h3>Season outlook</h3>
            {po != null ? <p><b>{pct(po, 0)}</b> to make the playoffs. <small className="muted">{outlook.data.label}</small></p>
              : <p className="muted">No outlook yet: it runs with each weekly ratings snapshot.</p>}
          </section>
          <section className="tp-block"><h3>Results and next</h3>
            <ul className="tp-games">
              {t.results.map((g) => <li key={`r${g.week}`}>Wk {g.week} {g.home ? 'vs' : '@'} {g.opp} · {g.result}</li>)}
              {t.next.map((g) => <li key={`n${g.week}`} className="muted">Wk {g.week} {g.home ? 'vs' : '@'} {g.opp} · {dateLabel(g.kickoff.slice(0, 10))}</li>)}
            </ul>
            {Object.keys(t.injury_counts || {}).length > 0 && (
              <p className="muted">Injury report: {Object.entries(t.injury_counts).map(([s, n]) => `${s} ${n}`).join(' · ')}</p>)}
          </section>
          <ModelOnTeam rec={rec} team={t.abbr} />
          <p className="ctx-note">{page.data.footnote}</p>
        </article>
      </div>
    </section>
  )
}

export function NbaTeams({ record }) {
  const page = useTeamPage('nba')
  const teams = page.data?.teams || []
  const [sel, setSel] = useState(null)
  if (page.loading) return <p className="panel-foot">Loading teams…</p>
  if (!teams.length) return <div className="empty"><h2>No team page yet</h2></div>
  const sorted = [...teams].sort((a, b) => (a.rating.rank ?? 99) - (b.rating.rank ?? 99))
  const t = teams.find((x) => x.team === (sel || sorted[0].team))
  const rec = record?.data?.leagues?.nba
  return (
    <section className="panel teams-v2">
      <header className="panel-head"><h1>Teams</h1>
        <p>{page.data.season - 1}-{String(page.data.season).slice(2)} season, the file the model reads · ranked of {page.data.of}.</p></header>
      <div className="tp-grid">
        <TeamList label="NBA teams by final model rating" sel={t.team} onSel={setSel}
                  items={sorted.map((x) => ({ key: x.team, rank: x.rating.rank, name: x.team }))}
                  sub={(x) => { const r = teams.find((y) => y.team === x.key).record; return `${r.w}-${r.l}` }} />
        <article className="tp-panel" aria-label={`${t.team} detail`}>
          <header><h2>{t.team}</h2>
            <p>{t.conference} · {t.conference_place} in the conference · {t.record.w}-{t.record.l} (#{t.record.rank})</p></header>
          <section className="tp-block"><h3>Season</h3><dl className="tp-stats">
            <Stat label="Net per game" s={t.net} fmt={(v) => (v > 0 ? '+' : '') + f1(v)} />
            <Stat label="Points scored" s={t.pts_for} fmt={f1} />
            <Stat label="Points allowed (lower is better)" s={t.pts_against} fmt={f1} />
            <Stat label="Final model rating" s={t.rating} fmt={(v) => (v > 0 ? '+' : '') + f1(v)} /></dl></section>
          <section className="tp-block"><h3>Splits</h3><dl className="tp-stats">
            <div className="tp-stat"><dt>Home</dt><dd><b>{t.home}</b></dd></div>
            <div className="tp-stat"><dt>Road</dt><dd><b>{t.road}</b></dd></div>
            <div className="tp-stat"><dt>Last 10</dt><dd><b>{t.last_10}</b></dd></div>
            <div className="tp-stat"><dt>Within 5 points</dt><dd><b>{t.within_5}</b></dd></div></dl>
            <p className="muted">Opened the season rated {t.opening_rating == null ? '—' : f1(t.opening_rating)} · first game {t.first_game ? dateLabel(t.first_game) : '—'}.</p>
          </section>
          <section className="tp-block"><h3>Season outlook</h3><p className="muted">{page.data.outlook}</p></section>
          <ModelOnTeam rec={rec} team={t.team} />
        </article>
      </div>
    </section>
  )
}

export function MlbPitchersParks({ record }) {
  const page = useTeamPage('mlb')
  const teams = page.data?.teams || []
  const [sel, setSel] = useState(null)
  if (page.loading) return <p className="panel-foot">Loading…</p>
  if (!teams.length) return <div className="empty"><h2>No page yet</h2></div>
  const sorted = [...teams].sort((a, b) => a.offense.rank - b.offense.rank)
  const t = teams.find((x) => x.team === (sel || sorted[0].team))
  return (
    <section className="panel teams-v2">
      <header className="panel-head"><h1>Pitchers &amp; parks</h1>
        <p>The walk-forward the model prices with, replayed to {dateLabel(page.data.as_of)} · league {page.data.league_rpg} runs per team-game.</p></header>
      <div className="tp-grid">
        <TeamList label="MLB teams by offence" sel={t.team} onSel={setSel}
                  items={sorted.map((x) => ({ key: x.team, rank: x.offense.rank, name: x.team }))}
                  sub={(x) => `${teams.find((y) => y.team === x.key).offense.value} R/G`} />
        <article className="tp-panel">
          <header><h2>{t.team}</h2><p>Home park {t.park}</p></header>
          <dl className="tp-stats">
            <Stat label="Offence, runs per game" s={t.offense} fmt={(v) => v.toFixed(2)} />
            <Stat label="Bullpen RA/27 (lower is better)" s={t.pen_ra27} fmt={(v) => v.toFixed(2)} />
            <div className="tp-stat"><dt>Bullpen: share that is its own record</dt><dd><b>{pct(t.pen_ra27.credibility, 0)}</b></dd></div>
            <Stat label="Park factor" s={t.park_factor} fmt={(v) => v.toFixed(3)} />
            <div className="tp-stat"><dt>Relief outs, last 3 days</dt><dd><b>{t.bullpen_workload.relief_outs_last_3_days}</b>
              <small className="inprice no">Not in the price</small></dd></div>
          </dl>
          <p className="ctx-note">{page.data.note}</p>
        </article>
      </div>
      <h2 className="panel-sub">Today's probables</h2>
      {page.data.probables?.length ? (
        <table className="data">
          <thead><tr><th>Start</th><th>Team</th><th>Probable</th><th className="num">RA/27</th><th className="num">Own record</th></tr></thead>
          <tbody>
            {page.data.probables.map((p) => (
              <tr key={`${p.game_key}-${p.team}`} className={p.status === 'refused' ? 'refused' : undefined}>
                <td>{startDay(p.start_utc)} {startTime(p.start_utc)}</td>
                <td>{p.team_name || p.team}</td>
                <td>{p.status === 'refused' ? <>TBD · refused</> : p.pitcher}</td>
                <td className="num">{p.ra27 ?? '—'}</td>
                <td className="num">{p.own_record_share == null ? '—' : pct(p.own_record_share, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">No slate today.</p>}
    </section>
  )
}

export function NhlAttackDefence() {
  const page = useTeamPage('nhl')
  const teams = page.data?.teams || []
  const [sel, setSel] = useState(null)
  if (page.loading) return <p className="panel-foot">Loading…</p>
  if (!teams.length) return <div className="empty"><h2>No page yet</h2></div>
  const sorted = [...teams].sort((a, b) => a.points.rank - b.points.rank)
  const t = teams.find((x) => x.team === (sel || sorted[0].team))
  return (
    <section className="panel teams-v2">
      <header className="panel-head"><h1>Attack &amp; defence</h1></header>
      <p className="stamp stamp-cap" role="note"><b>{page.data.banner}</b></p>
      <div className="tp-grid">
        <TeamList label="NHL teams by points" sel={t.team} onSel={setSel}
                  items={sorted.map((x) => ({ key: x.team, rank: x.points.rank, name: x.team }))}
                  sub={(x) => `${teams.find((y) => y.team === x.key).points.value} pts`} />
        <article className="tp-panel">
          <header><h2>{t.team}</h2><p>{t.gp} games, {page.data.season - 1}-{String(page.data.season).slice(2)}</p></header>
          <dl className="tp-stats">
            <Stat label="No-pull goals for per game" s={t.nopull_gf} fmt={(v) => v.toFixed(2)} />
            <Stat label="No-pull goals against (lower is better)" s={t.nopull_ga} fmt={(v) => v.toFixed(2)} />
            <div className="tp-stat"><dt>Pulled-goalie goals for / against</dt><dd><b>{t.pulled_goalie_for} / {t.pulled_goalie_against}</b></dd></div>
            <Stat label="Points" s={t.points} />
            <div className="tp-stat"><dt>Games past regulation</dt><dd><b>{t.past_regulation}</b></dd></div>
          </dl>
          <p className="ctx-note">{page.data.rates_note}</p>
        </article>
      </div>
    </section>
  )
}
