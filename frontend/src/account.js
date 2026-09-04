/* Discord OAuth2 with PKCE -- the flow built for apps that cannot hold
 * a client secret, which a static site cannot. No backend involvement:
 * the browser exchanges the authorization code directly with Discord
 * using the code_verifier it generated.
 *
 * Config: VITE_DISCORD_CLIENT_ID (Render env var on the static site).
 * If unset, useAccount() reports { enabled: false } and the UI hides
 * sign-in entirely -- the app is fully usable local-only.
 */

import { useEffect, useState, useCallback } from 'react'

const CLIENT_ID = import.meta.env.VITE_DISCORD_CLIENT_ID || ''
const TOKEN_KEY = 'cl_discord_token'
const USER_KEY = 'cl_discord_user'
const VERIFIER_KEY = 'cl_pkce_verifier'

function b64url(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)))
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

async function makePkcePair() {
  const verifierBytes = new Uint8Array(32)
  crypto.getRandomValues(verifierBytes)
  const verifier = b64url(verifierBytes)
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return { verifier, challenge: b64url(digest) }
}

export async function beginLogin() {
  const { verifier, challenge } = await makePkcePair()
  sessionStorage.setItem(VERIFIER_KEY, verifier)
  const params = new URLSearchParams({
    client_id: CLIENT_ID,
    response_type: 'code',
    redirect_uri: window.location.origin,
    scope: 'identify',
    code_challenge: challenge,
    code_challenge_method: 'S256',
  })
  window.location.assign(`https://discord.com/oauth2/authorize?${params}`)
}

async function exchangeCode(code) {
  const verifier = sessionStorage.getItem(VERIFIER_KEY)
  if (!verifier) throw new Error('missing pkce verifier')
  const body = new URLSearchParams({
    client_id: CLIENT_ID,
    grant_type: 'authorization_code',
    code,
    redirect_uri: window.location.origin,
    code_verifier: verifier,
  })
  const res = await fetch('https://discord.com/api/oauth2/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!res.ok) throw new Error('token exchange failed')
  const tok = await res.json()
  return { accessToken: tok.access_token, expiresAt: Date.now() + tok.expires_in * 1000 }
}

async function fetchMe(accessToken) {
  const res = await fetch('https://discord.com/api/users/@me', {
    headers: { Authorization: `Bearer ${accessToken}` },
  })
  if (!res.ok) throw new Error('identity fetch failed')
  const u = await res.json()
  return { id: u.id, username: u.global_name || u.username, avatar: u.avatar }
}

function loadStoredSession() {
  try {
    const tok = JSON.parse(localStorage.getItem(TOKEN_KEY))
    const user = JSON.parse(localStorage.getItem(USER_KEY))
    if (tok && user && tok.expiresAt > Date.now() + 60_000) return { token: tok, user }
  } catch { /* fall through */ }
  return null
}

export function useAccount() {
  const enabled = CLIENT_ID.length > 0
  const [session, setSession] = useState(() => (enabled ? loadStoredSession() : null))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!enabled) return
    const params = new URLSearchParams(window.location.search)
    const code = params.get('code')
    if (!code) return
    window.history.replaceState({}, '', window.location.pathname)
    setBusy(true)
    exchangeCode(code)
      .then(async (token) => {
        const user = await fetchMe(token.accessToken)
        localStorage.setItem(TOKEN_KEY, JSON.stringify(token))
        localStorage.setItem(USER_KEY, JSON.stringify(user))
        sessionStorage.removeItem(VERIFIER_KEY)
        setSession({ token, user })
      })
      .catch((e) => setError(e))
      .finally(() => setBusy(false))
  }, [enabled])

  const signOut = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    setSession(null)
  }, [])

  return {
    enabled,
    busy,
    error,
    user: session ? session.user : null,
    accessToken: session ? session.token.accessToken : null,
    signIn: beginLogin,
    signOut,
  }
}
