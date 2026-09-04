/* Settings + bet log: localStorage first (always works, offline, no
 * login), synced to the Render web service when signed in.
 *
 * Sync model: whole-state document per user, last-write-wins on
 * settings, union-by-id on bet log entries (so two devices logging
 * different bets both survive a merge). VITE_SYNC_URL unset -> sync is
 * silently disabled and everything stays device-local.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

const SYNC_URL = import.meta.env.VITE_SYNC_URL || ''

export const DEFAULT_SETTINGS = {
  bankroll: 1000,
  unitPct: 1,
  mode: 'flat',
  weeklyCapUnits: 10,
  updatedAt: 0,
}

function storageKey(user, kind) {
  const who = user ? user.id : 'local'
  return `cl_${kind}_${who}`
}

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    if (raw) return JSON.parse(raw)
  } catch { /* corrupted -> fallback */ }
  return fallback
}

function mergeState(local, remote) {
  if (!remote) return local
  const settings =
    (remote.settings?.updatedAt || 0) > (local.settings?.updatedAt || 0)
      ? remote.settings
      : local.settings
  const byId = new Map()
  ;[...(remote.betLog || []), ...(local.betLog || [])].forEach((b) => {
    const existing = byId.get(b.id)
    if (!existing || (b.updatedAt || 0) > (existing.updatedAt || 0)) byId.set(b.id, b)
  })
  const betLog = [...byId.values()].sort((a, b) => b.ts - a.ts)
  return { settings, betLog }
}

async function pullRemote(accessToken) {
  const res = await fetch(`${SYNC_URL}/v1/state`, {
    headers: { Authorization: `Bearer ${accessToken}` },
  })
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`sync pull failed: ${res.status}`)
  return res.json()
}

async function pushRemote(accessToken, state) {
  const res = await fetch(`${SYNC_URL}/v1/state`, {
    method: 'PUT',
    headers: { Authorization: `Bearer ${accessToken}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(state),
  })
  if (!res.ok) throw new Error(`sync push failed: ${res.status}`)
}

export function useBook(account) {
  const user = account?.user || null
  const accessToken = account?.accessToken || null
  const canSync = Boolean(SYNC_URL && accessToken)

  const [settings, setSettingsState] = useState(() =>
    loadJson(storageKey(user, 'settings'), DEFAULT_SETTINGS)
  )
  const [betLog, setBetLogState] = useState(() => loadJson(storageKey(user, 'betlog'), []))
  const [syncStatus, setSyncStatus] = useState(canSync ? 'idle' : 'off')
  const pushTimer = useRef(null)

  useEffect(() => {
    setSettingsState(loadJson(storageKey(user, 'settings'), DEFAULT_SETTINGS))
    setBetLogState(loadJson(storageKey(user, 'betlog'), []))
  }, [user && user.id])

  useEffect(() => {
    if (!canSync) { setSyncStatus('off'); return }
    setSyncStatus('pulling')
    pullRemote(accessToken)
      .then((remote) => {
        const merged = mergeState(
          { settings: loadJson(storageKey(user, 'settings'), DEFAULT_SETTINGS), betLog: loadJson(storageKey(user, 'betlog'), []) },
          remote
        )
        localStorage.setItem(storageKey(user, 'settings'), JSON.stringify(merged.settings))
        localStorage.setItem(storageKey(user, 'betlog'), JSON.stringify(merged.betLog))
        setSettingsState(merged.settings)
        setBetLogState(merged.betLog)
        setSyncStatus('synced')
      })
      .catch(() => setSyncStatus('error'))
  }, [canSync, accessToken, user && user.id])

  const schedulePush = useCallback(
    (nextSettings, nextLog) => {
      if (!canSync) return
      clearTimeout(pushTimer.current)
      pushTimer.current = setTimeout(() => {
        setSyncStatus('pushing')
        pushRemote(accessToken, { settings: nextSettings, betLog: nextLog })
          .then(() => setSyncStatus('synced'))
          .catch(() => setSyncStatus('error'))
      }, 800)
    },
    [canSync, accessToken]
  )

  const setSettings = useCallback(
    (patch) => {
      setSettingsState((prev) => {
        const next = { ...prev, ...patch, updatedAt: Date.now() }
        localStorage.setItem(storageKey(user, 'settings'), JSON.stringify(next))
        setBetLogState((log) => { schedulePush(next, log); return log })
        return next
      })
    },
    [user && user.id, schedulePush]
  )

  const logBet = useCallback(
    (bet) => {
      setBetLogState((prev) => {
        const entry = { id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, ts: Date.now(), updatedAt: Date.now(), result: null, ...bet }
        const next = [entry, ...prev]
        localStorage.setItem(storageKey(user, 'betlog'), JSON.stringify(next))
        setSettingsState((s) => { schedulePush(s, next); return s })
        return next
      })
    },
    [user && user.id, schedulePush]
  )

  const updateBet = useCallback(
    (id, patch) => {
      setBetLogState((prev) => {
        const next = prev.map((b) => (b.id === id ? { ...b, ...patch, updatedAt: Date.now() } : b))
        localStorage.setItem(storageKey(user, 'betlog'), JSON.stringify(next))
        setSettingsState((s) => { schedulePush(s, next); return s })
        return next
      })
    },
    [user && user.id, schedulePush]
  )

  const deleteBet = useCallback(
    (id) => {
      setBetLogState((prev) => {
        const next = prev.filter((b) => b.id !== id)
        localStorage.setItem(storageKey(user, 'betlog'), JSON.stringify(next))
        setSettingsState((s) => { schedulePush(s, next); return s })
        return next
      })
    },
    [user && user.id, schedulePush]
  )

  return { settings, setSettings, betLog, logBet, updateBet, deleteBet, syncStatus }
}
