/* Staking math and confidence grading -- pure functions, no state.
 *
 * HONESTY NOTES baked into the design:
 * - Cover probability is a normal approximation from the model/market
 *   gap (sigma ~13.86 ATS). It is NOT yet calibrated -- the walk-forward
 *   backtest has not shown flagged plays clearing 52.4%. Kelly stakes
 *   built on these probabilities are therefore capped hard (quarter
 *   Kelly default, 2u max) and flat staking is offered as the safest mode.
 * - Confidence pips are computable drivers only. A driver whose input
 *   is missing renders as unknown, never as a filled pip.
 */

export const ATS_SIGMA = 13.86
export const PLAY_GAP = 4.0
export const LEAN_GAP = 2.5
export const MAX_STAKE_UNITS = 2.0
export const DEFAULT_PRICE = -110

export function normCdf(z) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z))
  const d = 0.3989423 * Math.exp((-z * z) / 2)
  let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
  if (z > 0) p = 1 - p
  return p
}

/* Calibrated cover probability. edgeCoef comes from margin_dist.json's
 * logistic fit on held-out backtest games -- the honest mapping (a
 * 4-pt edge -> ~51%, NOT the ~61% the normal approximation claims).
 * Falls back to the normal approximation only when the calibration
 * file hasn't loaded. */
export function coverProb(gap, edgeCoef = null) {
  if (edgeCoef) return 1 / (1 + Math.exp(-edgeCoef * Math.abs(gap)))
  return normCdf(Math.abs(gap) / ATS_SIGMA)
}

export function breakevenProb(americanPrice) {
  if (americanPrice > 0) return 100 / (americanPrice + 100)
  return -americanPrice / (-americanPrice + 100)
}

/* Net decimal payout per 1 staked (the "b" in Kelly). */
export function netPayout(americanPrice) {
  if (americanPrice > 0) return americanPrice / 100
  return 100 / -americanPrice
}

export function fullKellyFraction(p, americanPrice) {
  const b = netPayout(americanPrice)
  const f = (p * (b + 1) - 1) / b
  return Math.max(0, f)
}

export const STAKING_MODES = {
  flat: { label: 'Flat', multiplier: 0 },
  qk: { label: 'Quarter Kelly', multiplier: 0.25 },
  hk: { label: 'Half Kelly', multiplier: 0.5 },
}

/* Returns { units, dollars, fullKelly, applied } for one play. */
export function sizeStake({ prob, price = DEFAULT_PRICE, settings }) {
  const { bankroll, unitPct, mode } = settings
  const unitDollars = (bankroll * unitPct) / 100
  const fullK = fullKellyFraction(prob, price)

  if (mode === 'flat' || fullK <= 0) {
    const units = fullK > 0 || mode === 'flat' ? 1 : 0
    return { units, dollars: units * unitDollars, fullKelly: fullK, applied: mode === 'flat' ? 0 : fullK }
  }

  const mult = STAKING_MODES[mode] ? STAKING_MODES[mode].multiplier : 0.25
  const applied = fullK * mult
  let units = (applied * bankroll) / unitDollars
  units = Math.min(units, MAX_STAKE_UNITS)
  units = Math.round(units * 20) / 20
  return { units, dollars: Math.round(units * unitDollars), fullKelly: fullK, applied }
}

/* Does the segment between the market number and the model number cross
 * a key margin (3 or 7, either sign)? Crossing means the extra points
 * the model sees are partly "spent" hopping the most common margins. */
export function crossesKeyNumber(marketSpread, modelSpread) {
  const keys = [3, 7, -3, -7]
  const lo = Math.min(marketSpread, modelSpread)
  const hi = Math.max(marketSpread, modelSpread)
  return keys.some((k) => k > lo && k < hi)
}

/* Confidence drivers. Each returns true / false / null (unknown).
 * ratingsByTeam: { TEAM: { rating_std } } or null.
 * tierStats: from performance.json, { play_tier_ats_pct } or null. */
export function confidenceDrivers(d, market, ratingsByTeam, tierStats) {
  const gap = market === 'spread' ? d.spread_gap : d.total_gap
  const modelSpread = d.market_spread + d.spread_gap

  const drivers = [
    { key: 'edge', label: `Edge ${Math.abs(gap).toFixed(1)} pts`, ok: Math.abs(gap) >= PLAY_GAP },
    {
      key: 'movement',
      label: d.moved_toward_model === true ? 'Line moving our way' : d.moved_toward_model === false ? 'Line moving against us' : 'Line movement unknown',
      ok: d.moved_toward_model === true ? true : d.moved_toward_model === false ? false : null,
    },
  ]

  if (market === 'spread') {
    const crosses = crossesKeyNumber(d.market_spread, modelSpread)
    drivers.push({ key: 'keynum', label: crosses ? 'Crosses key number' : 'Clear of key numbers', ok: !crosses })
  } else {
    drivers.push({ key: 'keynum', label: 'Total market', ok: null })
  }

  let stability = null
  if (ratingsByTeam) {
    const h = ratingsByTeam[d.home_team]
    const a = ratingsByTeam[d.away_team]
    if (h && a && h.rating_std != null && a.rating_std != null) {
      const stds = Object.values(ratingsByTeam)
        .map((t) => t.rating_std)
        .filter((v) => v != null)
        .sort((x, y) => x - y)
      const p75 = stds[Math.floor(stds.length * 0.75)] ?? Infinity
      stability = h.rating_std <= p75 && a.rating_std <= p75
    }
  }
  drivers.push({
    key: 'stability',
    label: stability === true ? 'Ratings stable' : stability === false ? 'Ratings uncertain' : 'Rating stability unknown',
    ok: stability,
  })

  let tierOk = null
  let tierLabel = 'Tier record: no sample yet'
  if (tierStats && tierStats.n_plays >= 30 && tierStats.ats_pct != null) {
    tierOk = tierStats.ats_pct >= 0.524
    tierLabel = `Tier hit ${(tierStats.ats_pct * 100).toFixed(0)}% over ${tierStats.n_plays} plays`
  }
  drivers.push({ key: 'tier', label: tierLabel, ok: tierOk })

  return drivers
}

export function confidenceScore(drivers) {
  const filled = drivers.filter((dr) => dr.ok === true).length
  const known = drivers.filter((dr) => dr.ok !== null).length
  const label = filled >= 4 ? 'High' : filled >= 3 ? 'Solid' : filled >= 2 ? 'Moderate' : 'Thin'
  return { filled, known, total: drivers.length, label }
}

export function probToAmerican(p) {
  p = Math.min(Math.max(p, 1e-6), 1 - 1e-6)
  return p >= 0.5 ? -Math.round((100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p)
}

/* Fair prices at half-point alt lines around the market number, from
 * the empirical residual pmf (margin_dist.json). Answers "what juice
 * is this alt spread actually worth given OUR number" -- compare to
 * the book's alt menu to find mispriced rungs. */
export function altLineFairPrices(modelMargin, marketLine, residualPmf, span = 2) {
  const entries = Object.entries(residualPmf).map(([r, p]) => [parseFloat(r), p])
  const rows = []
  for (let off = -span; off <= span + 1e-9; off += 0.5) {
    const line = Math.round((marketLine + off) * 2) / 2
    const need = line - modelMargin
    let cover = 0, push = 0
    for (const [r, p] of entries) {
      if (r > need) cover += p
      else if (r === need) push += p
    }
    const prob = push < 1 ? cover / (1 - push) : 0.5
    rows.push({ line, prob, fair: probToAmerican(prob) })
  }
  return rows
}
