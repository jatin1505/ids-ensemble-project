"""
FastAPI entry point.

Run from the project root (the folder containing both backend/ and
shared/):

    uvicorn backend.main:app --reload

Then visit:
    http://127.0.0.1:8000/health   -- plain health check
    http://127.0.0.1:8000/docs     -- FastAPI's auto-generated API docs,
                                       useful for poking at endpoints
                                       without writing a frontend first
    ws://127.0.0.1:8000/ws         -- the live event stream, now
                                       requiring ?token=<app JWT> -- see
                                       backend/auth.py / auth_routes.py
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# MUST run before importing anything that reads os.environ at import
# time -- backend.auth (imported indirectly via auth_routes below)
# raises AuthConfigError immediately if APP_JWT_SECRET isn't set yet.
# Resolved relative to this file, not the terminal's cwd, so it doesn't
# matter which directory you happen to launch uvicorn from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
print(f"[debug] looking for .env at: {Path(__file__).resolve().parent.parent / '.env'}")
print(f"[debug] that path exists: {(Path(__file__).resolve().parent.parent / '.env').exists()}")
import os
print(f"[debug] APP_JWT_SECRET is now: {os.environ.get('APP_JWT_SECRET')!r}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.replay_engine import run_replay
from backend.websocket_manager import router as websocket_router
from backend.auth_routes import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch the replay engine as a background asyncio task.
    # run_replay() loads the trained models once (model_runtime.load_runtime())
    # and then loops forever, scoring one replayed flow every
    # REPLAY_INTERVAL_SECONDS and pushing it to every connected client
    # via websocket_manager.manager.broadcast(). This is the ONLY thing
    # that produces real RiskEvents -- there is deliberately no other
    # trigger for it, so if this task isn't running, /ws stays silent.
    replay_task = asyncio.create_task(run_replay())
    print("[startup] replay engine started")

    yield

    # Shutdown: cancel cleanly. Without this, uvicorn tearing down the
    # event loop on Ctrl+C can kill the task mid-broadcast and print a
    # noisy traceback for something that isn't actually a bug.
    replay_task.cancel()
    try:
        await replay_task
    except asyncio.CancelledError:
        pass
    print("[shutdown] replay engine stopped")


app = FastAPI(title="IDS Backend", lifespan=lifespan)

# The React dev server runs on a different port (usually 5173 or 3000)
# than FastAPI (8000). Browsers block requests between different origins
# by default -- this is CORS -- and without this middleware, the
# dashboard would just see a "Network Error" in the browser console.
# allow_origins=["*"] is fine for local development; narrow it to your
# actual deployed frontend URL before this is ever public. NOTE: this
# middleware does NOT cover the WebSocket handshake (browsers don't
# apply CORS to WS the way they do to fetch/XHR) -- /ws's own auth check
# in websocket_manager.py is what actually gates that connection.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All routers included together, in one place, AFTER app exists --
# this is what the previous version got wrong: app.include_router()
# was called above, before `app = FastAPI(...)` had even run.
app.include_router(websocket_router)
app.include_router(auth_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}