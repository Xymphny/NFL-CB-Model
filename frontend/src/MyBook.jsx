import { STAKING_MODES } from './staking'

function formatSigned(value, digits = 1) {
  if (value === null || value === undefined) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

export function AccountChip({ account }) {
  if (!account.enabled) return null
  if (!account.user) {
    return (
      <button className="account-chip" onClick={account.signIn} disabled={account.busy}>
        {account.busy ? 'Signing in…' : 'Sign in with Discord'}
      </button>
    )
  }
  return (
    <button className="account-chip signed-in" onClick={account.signOut} title="Sign out">
      <span className="account-avatar">{account.user.username.slice(0, 2).toUpperCase()}</span>
      {account.user.username}
    </button>
  )
}

function SyncBadge({ status }) {
  const map = {
    off: null,
    idle: null,
    pulling: 'Syncing…',
    pushing: 'Syncing…',
    synced: 'Synced across devices',
    error: 'Sync unavailable — saved on this device',
  }
  const text = map[status]
  if (!text) return <span className="sync-badge">Saved on this device</span>
  return <span className="sync-badge">{text}</span>
}

export function SettingsPanel({ book }) {
  const { settings, setSettings, syncStatus } = book
  const unitDollars = Math.round((settings.bankroll * settings.unitPct) / 100)

  return (
    <div className="settings-panel">
      <div className="settings-head">
        <h2 className="section-heading">Bankroll and staking</h2>
        <SyncBadge status={syncStatus} />
      </div>

      <label className="field">
        <span className="field-label">Bankroll ($)</span>
        <input
          type="number" min="0" step="100" value={settings.bankroll}
          onChange={(e) => setSettings({ bankroll: Math.max(0, Number(e.target.value) || 0) })}
        />
      </label>

      <label className="field">
        <span className="field-label">Unit size — {settings.unitPct}% of bankroll (${unitDollars})</span>
        <input
          type="range" min="0.5" max="3" step="0.5" value={settings.unitPct}
          onChange={(e) => setSettings({ unitPct: Number(e.target.value) })}
        />
      </label>

      <label className="field">
        <span className="field-label">Staking mode</span>
        <select value={settings.mode} onChange={(e) => setSettings({ mode: e.target.value })}>
          <option value="flat">Flat — every play is 1 unit</option>
          <option value="qk">Quarter Kelly — sized to edge, conservative</option>
          <option value="hk">Half Kelly — sized to edge, aggressive</option>
        </select>
      </label>

      <label className="field">
        <span className="field-label">Weekly cap — {settings.weeklyCapUnits} units</span>
        <input
          type="range" min="5" max="20" step="1" value={settings.weeklyCapUnits}
          onChange={(e) => setSettings({ weeklyCapUnits: Number(e.target.value) })}
        />
      </label>

      <p className="settings-note">
        {settings.mode === 'flat' && 'Flat staking is the safest mode and the easiest to audit — recommended until the model has a graded winning sample.'}
        {settings.mode === 'qk' && 'Quarter Kelly caps any single play at 2 units regardless of edge. Kelly sizing is only as good as the model probabilities behind it.'}
        {settings.mode === 'hk' && 'Half Kelly roughly doubles both growth and drawdowns versus quarter Kelly. Only appropriate with a proven, calibrated model — the track record tab is the judge.'}
      </p>
    </div>
  )
}

function gradeButton(bet, result, updateBet) {
  const active = bet.result === result
  return (
    <button
      key={result}
      className={`grade-btn ${result}${active ? ' active' : ''}`}
      onClick={() => updateBet(bet.id, { result: active ? null : result })}
    >
      {result.toUpperCase()}
    </button>
  )
}

export function BetLog({ book }) {
  const { betLog, updateBet, deleteBet, settings } = book
  const unitDollars = (settings.bankroll * settings.unitPct) / 100 || 1

  const graded = betLog.filter((b) => b.result === 'w' || b.result === 'l')
  const wins = graded.filter((b) => b.result === 'w').length
  const losses = graded.length - wins
  const units = betLog.reduce((sum, b) => {
    const stake = b.stakeUnits || 1
    const price = b.price || -110
    const winPayout = price > 0 ? price / 100 : 100 / -price
    if (b.result === 'w') return sum + stake * winPayout
    if (b.result === 'l') return sum - stake
    return sum
  }, 0)
  /* CLV sign: positive means you beat the close.
     Spread (your side's own line): line - close. Taking -3.5 that
     closes -4.5 -> +1. Taking +7 that closes +6 -> +1.
     Totals: over beats close when it rises (close - line), under when
     it falls (line - close). */
  const clvBets = betLog.filter((b) => b.closeLine != null && b.line != null)
  const betClv = (b) => {
    if (b.market === 'total') return b.ou === 'under' ? b.line - b.closeLine : b.closeLine - b.line
    return b.line - b.closeLine
  }
  const avgClv = clvBets.length ? clvBets.reduce((s, b) => s + betClv(b), 0) / clvBets.length : null

  return (
    <div>
      <div className="settings-head">
        <h2 className="section-heading">Bet log</h2>
        <span className="sync-badge">{betLog.length} plays</span>
      </div>
      <p className="section-sub">
        Log every play at the price you actually got. Grade results after the game — your record and
        your closing-line value are computed from what you logged, not from the model's picks.
      </p>

      {betLog.length > 0 && (
        <div className="log-summary">
          <span>{wins}–{losses}</span>
          <span className={units >= 0 ? 'pos' : 'neg'}>{formatSigned(units, 1)}u (${formatSigned(units * unitDollars, 0)})</span>
          <span>{avgClv != null ? `CLV ${formatSigned(avgClv, 1)} pts` : 'CLV — log closing lines'}</span>
        </div>
      )}

      {betLog.length === 0 && (
        <div className="empty-state">
          <strong>Nothing logged yet</strong>
          Use the log button on any play card, or plays you take elsewhere still count — the log is
          about your discipline, not our picks.
        </div>
      )}

      {betLog.map((bet) => (
        <div className="log-row" key={bet.id}>
          <div className="log-main">
            <span className="log-pick">{bet.label}</span>
            <span className="log-detail">
              {bet.line != null ? formatSigned(bet.line, 1) : ''} at {bet.price > 0 ? `+${bet.price}` : bet.price} · {bet.stakeUnits}u
              {bet.closeLine != null && ` · closed ${formatSigned(bet.closeLine, 1)}`}
            </span>
          </div>
          <div className="log-actions">
            {['w', 'l', 'p'].map((r) => gradeButton(bet, r, updateBet))}
            <button className="grade-btn del" onClick={() => deleteBet(bet.id)} aria-label="Delete">×</button>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function MyBook({ book, account }) {
  return (
    <div>
      {account.enabled && !account.user && (
        <div className="empty-state" style={{ marginTop: 16 }}>
          <strong>You're using device-only mode</strong>
          Settings and your bet log save to this browser. Sign in with Discord to sync them across
          devices.
        </div>
      )}
      <section><SettingsPanel book={book} /></section>
      <section><BetLog book={book} /></section>
    </div>
  )
}
