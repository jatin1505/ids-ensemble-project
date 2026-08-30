"""
WebSocket connection management.

Two things live here:
1. ConnectionManager -- tracks which dashboard clients are currently
   connected, so we can push a new result to all of them at once.
2. The /ws route -- what happens when a client connects.

backend/replay_engine.py owns "where results come from": it's started
as a background task by backend/main.py's lifespan handler (not from
this file) and calls manager.broadcast(event) directly, once per
replayed flow. This file only owns "how to talk to connected clients."
The /ws route below just accepts a connection and blocks, keeping it
open until the client disconnects -- it does not generate any events
itself.

FIX (was previously wrong): this file used to contain a Phase-0
placeholder (_make_dummy_event()) that the /ws loop called every 2
seconds, forever -- including a stale "lstm" key in model_breakdown
left over from before the LSTM->GMM swap. That loop ran unconditionally
regardless of whether replay_engine.py / model_runtime.py were wired
up, so it was silently masking the fact that real data was never
reaching the dashboard. Both the dummy generator and the loop that
called it are removed; broadcasting is now driven entirely by
replay_engine.py's own async loop calling manager.broadcast().
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from shared.schemas import RiskEvent

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        # Guard the removal: broadcast() below can already have dropped
        # a dead connection before this gets called a second time from
        # the except block. Without this check, the second call raises
        # ValueError.
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: RiskEvent):
        message = event.model_dump_json()
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # A client can vanish between us listing connections and
                # actually sending to them. Don't let one dead client
                # break the broadcast for everyone else -- just mark it
                # for cleanup and move on.
                dead_connections.append(connection)
        for connection in dead_connections:
            self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # The dashboard isn't expected to send us anything -- this just
        # blocks here so the connection stays registered, and so we
        # find out the MOMENT the client disconnects (tab closed, dev
        # server restarted, etc.) via WebSocketDisconnect, instead of
        # only discovering it later when broadcast() tries to send to a
        # dead socket and fails.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)