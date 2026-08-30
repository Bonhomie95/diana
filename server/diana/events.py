import asyncio
import json
import logging
import time
from collections import deque

log = logging.getLogger("diana.events")


class Hub:
    """WebSocket fan-out hub. Everything the UI shows arrives through here."""

    def __init__(self):
        self.clients: set = set()
        self.status: str = "booting"  # booting | idle | listening | thinking | working | offline
        self.loop: asyncio.AbstractEventLoop | None = None
        self.activity: deque = deque(maxlen=80)  # recent worker/tool events for the UI

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, ws):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws):
        self.clients.discard(ws)

    async def broadcast(self, type_: str, data):
        if type_ == "log" and isinstance(data, dict):
            data = {**data, "ts": time.strftime("%H:%M:%S")}
            self.activity.append(data)
        payload = json.dumps({"type": type_, "data": data})
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def emit(self, type_: str, data):
        """Thread-safe fire-and-forget broadcast."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(type_, data), self.loop)

    async def set_status(self, status: str, detail: str = ""):
        self.status = status
        await self.broadcast("status", {"status": status, "detail": detail})


hub = Hub()
