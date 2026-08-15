"""
WebSocket connection management.

Two things live here:
1. ConnectionManager -- tracks which dashboard clients are currently
   connected, so we can push a new result to all of them at once.
2. The /ws route -- what happens when a client connects.

Once backend/replay_engine.py exists (next step, Phase 1), it will
import `manager` from this file and call `manager.broadcast(event)`
every time it produces a real RiskEvent -- it won't need to know
anything about WebSockets itself. That separation is deliberate: this
file owns "how to talk to connected clients," replay_engine.py will own
"where results come from."
"""

import asyncio
import random
import time

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


def _make_dummy_event() -> RiskEvent:
    """
    Phase 0 placeholder only. Generates a fake-but-schema-valid RiskEvent
    so Member B has a real stream to build the dashboard against before
    any model or replay engine exists.

    Delete this function (and the loop below that calls it) once
    replay_engine.py is producing real events -- it should never run
    alongside real data.
    """
    return RiskEvent(
        flow_id=f"dummy-{random.randint(1000, 9999)}",
        src_ip="192.168.1.10",
        dst_ip="192.168.1.1",
        protocol="TCP",
        timestamp=time.time(),
        risk_level=random.choice(["Low", "Medium", "High"]),
        final_score=round(random.uniform(0, 1), 3),
        model_breakdown={
            "isolation_forest": round(random.uniform(0, 1), 3),
            "autoencoder": round(random.uniform(0, 1), 3),
            "lstm": round(random.uniform(0, 1), 3),
        },
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Phase 0: broadcast a dummy event every 2 seconds so there's
            # something for the frontend to receive and render.
            # Phase 1: this loop gets replaced by replay_engine.py
            # actually reading flows and calling manager.broadcast().
            await manager.broadcast(_make_dummy_event())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)