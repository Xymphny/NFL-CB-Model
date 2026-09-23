/* Your own book: bankroll, exposure cap, bet log, personal CLV. Settings and
 * the log live in the browser and sync through the account (store.js,
 * account.js -- unchanged). The stake on any pick comes from the core; this
 * page only sets the bankroll it is a fraction of. */

function signedFmt(value, digits = 1) {
  if (value === null || value === undefined) return '—'
  return `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}`
}

export function AccountChip({ account }) {
  if (!account.enabled) return null
  if (!account.user) {
    return (
      <button className="account" onClick={account.signIn} disabled={account.busy}>
        {account.busy ? 'Signing in…' : 'Sign in with Discord'}
      </button>
    )
  }
  return (
    <button className="account in" onClick={account.signOut} title="Sign out">
      <span className="avatar" aria-hidden="true">{account.user.username.slice(0, 2).toUpperCase()}</span>
      {account.user.username}
    </button>
  )
}

const SYNC = {
  pulling: 'Syncing…', pushing: 'Syncing…', synced: 'Synced across devices',
  error: 'Sync unavailable; saved on this device',
}

function Settings({ book }) {
  const { settings, setSettings, syncStatus } = book
  const unit = Math.round((settings.bankroll * settings.unitPct) / 100)
  return (
    <section className="panel-block">
      <header className="block-head">
        <h2>Bankroll and exposure</h2>
        <span className="sync">{SYNC[syncStatus] || 'Saved on this device'}</span>
      </header>
      <div className="fields">
        <label className="field">
          <span>Bankroll</span>
          <input type="number" min="0" step="100" inputMode="numeric" value={settings.bankroll}
                 onChange={(e) => setSettings({ bankroll: Math.max(0, Number(e.target.value) || 0) })} />
        </label>
        <label className="field">
          <span>Unit, {settings.unitPct}% of bankroll ({unit.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })})</span>
          <input type="range" min="0.5" max="3" step="0.5" value={settings.unitPct}
                 onChange={(e) => setSettings({ unitPct: Number(e.target.value) })} />
        </label>
        <label className="field">
          <span>Weekly exposure cap, {settings.weeklyCapUnits} units</span>
          <input type="range" min="5" max="20" step="1" value={settings.weeklyCapUnits}
                 onChange={(e) => setSettings({ weeklyCapUnits: Number(e.target.value) })} />
        </label>
      </div>
      <p className="block-note">
        Stakes are sized by the core from each league's grade against closing prices, as a
        fraction of this bankroll. Every league is ungraded today, so every pick is unsized and the
        log records it as 0 units. The cap blocks logging once a week's logged units would pass it.
      </p>
    </section>
  )
}

function BetLog({ book }) {
  const { betLog, updateBet, deleteBet, settings } = book
  const unit = (settings.bankroll * settings.unitPct) / 100 || 1
  const graded = betLog.filter((b) => b.result === 'w' || b.result === 'l')
  const wins = graded.filter((b) => b.result === 'w').length
  const units = betLog.reduce((sum, b) => {
    const stake = b.stakeUnits || 0
    const price = b.price || -110
    const payout = price > 0 ? price / 100 : 100 / -price
    if (b.result === 'w') return sum + stake * payout
    if (b.result === 'l') return sum - stake
    return sum
  }, 0)
  /* Your CLV on your own logged bets: your line against the close you
   * logged. Spread: line − close. Totals: over gains when the close rises. */
  const clvBets = betLog.filter((b) => b.closeLine != null && b.line != null)
  const clv = (b) => (b.market === 'total' ? (b.ou === 'under' ? b.line - b.closeLine : b.closeLine - b.line) : b.line - b.closeLine)
  const avgClv = clvBets.length ? clvBets.reduce((s, b) => s + clv(b), 0) / clvBets.length : null

  return (
    <section className="panel-block">
      <header className="block-head">
        <h2>Bet log</h2>
        <span className="sync">{betLog.length} logged</span>
      </header>
      {betLog.length > 0 && (
        <dl className="log-sum">
          <div><dt>Record</dt><dd>{wins}–{graded.length - wins}</dd></div>
          <div><dt>Result</dt><dd>{signedFmt(units, 1)} u <small>({signedFmt(units * unit, 0)} $)</small></dd></div>
          <div><dt>Your CLV</dt><dd>{avgClv != null ? `${signedFmt(avgClv, 1)} pts` : 'log closing lines to see it'}</dd></div>
        </dl>
      )}
      {betLog.length === 0 ? (
        <div className="empty">
          <h2>Nothing logged yet</h2>
          <p>Open any priced game on the board and use “Log this pick”. Bets you take elsewhere
            belong here too: the log is your record, not the model's.</p>
        </div>
      ) : (
        <ul className="bets">
          {betLog.map((b) => (
            <li key={b.id}>
              <span className="bet-main">
                <b>{b.label}</b>
                <small>
                  {b.price != null ? (b.price > 0 ? `+${b.price}` : `−${Math.abs(b.price)}`) : ''}
                  {b.book ? ` at ${b.book}` : ''} · {b.stakeUnits ? `${b.stakeUnits} u` : 'unsized'}
                  {b.closeLine != null ? ` · closed ${signedFmt(b.closeLine, 1)}` : ''}
                </small>
              </span>
              <span className="bet-grade" role="group" aria-label="Result">
                {[['w', 'Won'], ['l', 'Lost'], ['p', 'Push']].map(([r, label]) => (
                  <button key={r} className={`seg${b.result === r ? ' on' : ''}`} aria-pressed={b.result === r}
                          onClick={() => updateBet(b.id, { result: b.result === r ? null : r })}>{label}</button>
                ))}
                <button className="seg del" onClick={() => deleteBet(b.id)} aria-label={`Delete ${b.label}`}>Delete</button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default function Book({ book, account }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h1>My book</h1>
        <p>{account.enabled && !account.user
          ? 'Saved in this browser. Sign in with Discord to carry it across devices.'
          : 'Your bankroll, your cap, and what you actually bet.'}</p>
      </header>
      <Settings book={book} />
      <BetLog book={book} />
    </section>
  )
}
