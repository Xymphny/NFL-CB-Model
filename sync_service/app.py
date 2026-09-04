"""
Coverline sync service -- a deliberately tiny Render web service that
stores one JSON state document (settings + bet log) per Discord user.

Auth model: the static site obtains a Discord access token client-side
via OAuth2 PKCE (no secret involved anywhere). Every request here
carries that token as a Bearer header; this service verifies it by
asking Discord who the token belongs to (GET /users/@me) and uses the
returned user id as the storage key. Verification results are cached
in-memory for 10 minutes so normal usage costs ~1 Discord call per
user per session, well under Discord's rate limits.

Storage: SQLite at $DATA_DIR/state.db. DATA_DIR must point at a Render
persistent disk mount -- a bare web service's filesystem is EPHEMERAL
and wiped on every deploy, which would silently destroy user bet logs.
The render.yaml disk block is not optional.

Size guard: state documents are capped at 256KB -- a full season of
logged bets is a few KB, so anything near the cap is abuse, not usage.
"""

import json
import os
import sqlite3
import time
import threading

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = os.environ.get("DATA_DIR", "./data")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
MAX_STATE_BYTES = 256 * 1024
TOKEN_CACHE_TTL = 600

app = FastAPI(title="coverline-sync", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["GET", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

_db_lock = threading.Lock()
_token_cache = {}
_token_cache_lock = threading.Lock()


def db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(os.path.join(DATA_DIR, "state.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS user_state ("
        "  user_id TEXT PRIMARY KEY,"
        "  state TEXT NOT NULL,"
        "  updated_at REAL NOT NULL"
        ")"
    )
    return conn


async def verify_discord_token(authorization: str) -> str:
    """Returns the Discord user id for a valid Bearer token, 401 otherwise."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[7:]

    now = time.time()
    with _token_cache_lock:
        cached = _token_cache.get(token)
        if cached and cached[1] > now:
            return cached[0]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://discord.com/api/users/@me",
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Discord token")

    user_id = resp.json()["id"]
    with _token_cache_lock:
        _token_cache[token] = (user_id, now + TOKEN_CACHE_TTL)
        if len(_token_cache) > 5000:
            expired = [k for k, v in _token_cache.items() if v[1] <= now]
            for k in expired:
                del _token_cache[k]
    return user_id


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/state")
async def get_state(authorization: str = Header(default="")):
    user_id = await verify_discord_token(authorization)
    with _db_lock:
        conn = db()
        try:
            row = conn.execute(
                "SELECT state FROM user_state WHERE user_id = ?", (user_id,)
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="No state yet")
    return json.loads(row[0])


@app.put("/v1/state")
async def put_state(request: Request, authorization: str = Header(default="")):
    user_id = await verify_discord_token(authorization)
    body = await request.body()
    if len(body) > MAX_STATE_BYTES:
        raise HTTPException(status_code=413, detail="State too large")
    try:
        state = json.loads(body)
        assert isinstance(state, dict)
    except (json.JSONDecodeError, AssertionError):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    with _db_lock:
        conn = db()
        try:
            conn.execute(
                "INSERT INTO user_state (user_id, state, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET state = excluded.state, updated_at = excluded.updated_at",
                (user_id, json.dumps(state), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    return {"ok": True}
