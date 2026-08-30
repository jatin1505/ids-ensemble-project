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
    ws://127.0.0.1:8000/ws         -- the live event stream (connect a
                                       WebSocket client, e.g. Member B's
                                       dashboard, to see RiskEvents)

FIX (was previously missing): this file's docstring always claimed
"lifespan management" but no lifespan actually existed, and nothing
anywhere called replay_engine.run_replay(). Without that, /ws was
still running websocket_manager.py's old Phase-0 dummy-event loop
forever -- real models, real replay data, none of it reachable. The
lifespan() below is what actually starts the real pipeline.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.replay_engine import run_replay
from backend.websocket_manager import router as websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch the replay engine as a background asyncio task.
    # run_replay() loads the trained models once (model_runtime.load_runtime())
    # and then loops forever, scoring one replayed flow every
    # REPLAY_INTERVAL_SECONDS and pushing it to every connected client
    # via websocket_manager.manager.broadcast(). This is the ONLY thing
    # that produces real RiskEvents -- there is deliberately no other
    # trigger for it, so if this task isn't running, /ws stays silent
    # (or, before this fix, kept emitting dummy data instead).
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
# by default -- this is CORS -- and without this middleware, Member B
# would just see a "Network Error" in the browser console with no
# useful detail about why. allow_origins=["*"] is fine for local
# development; narrow it to your actual deployed frontend URL later if
# you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}