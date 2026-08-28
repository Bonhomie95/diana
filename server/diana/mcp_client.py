"""MCP connectors: plug any Model Context Protocol server into Diana's agents.

Config lives at /data/mcp.json:
{
  "servers": {
    "time":   {"command": "uvx", "args": ["mcp-server-time"]},
    "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
               "env": {"GITHUB_TOKEN": "..."}},
    "remote": {"url": "https://example.com/mcp"}
  }
}

Each server gets a persistent background session with automatic reconnect.
Tools are exposed to worker agents as "server:tool".
"""
import asyncio
import json
import logging

from . import config
from .events import hub

log = logging.getLogger("diana.mcp")

CONFIG_PATH = config.DATA_DIR / "mcp.json"
CALL_TIMEOUT = 120


def _result_text(result) -> str:
    """Flatten a CallToolResult into text for the worker's context."""
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        elif getattr(item, "data", None) is not None:
            parts.append(f"[binary content: {getattr(item, 'mimeType', 'unknown')}]")
    structured = getattr(result, "structuredContent", None)
    if structured and not parts:
        parts.append(json.dumps(structured)[:4000])
    out = "\n".join(parts).strip() or "(empty result)"
    if getattr(result, "isError", False):
        out = f"TOOL ERROR: {out}"
    return out


class ServerConn:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.status = "connecting"
        self.error = ""
        self.tools: list[dict] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    def describe(self) -> dict:
        transport = "http" if self.cfg.get("url") else "stdio"
        return {"name": self.name, "status": self.status, "error": self.error,
                "transport": transport,
                "target": self.cfg.get("url") or " ".join(
                    [self.cfg.get("command", "")] + list(self.cfg.get("args", []))),
                "tools": self.tools}

    def start(self):
        if not self.task or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        self.status = "disconnected"

    async def call(self, tool: str, args: dict) -> str:
        if self.status != "connected":
            return f"TOOL ERROR: connector '{self.name}' is {self.status} ({self.error})"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((tool, args, fut))
        try:
            return await asyncio.wait_for(fut, timeout=CALL_TIMEOUT + 10)
        except asyncio.TimeoutError:
            return f"TOOL ERROR: call to {self.name}:{tool} timed out"

    async def _run(self):
        backoff = 3
        while True:
            try:
                await self._serve_once()
                backoff = 3
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.status = "error"
                self.error = str(e)[:300]
                log.warning("MCP '%s' connection failed: %s", self.name, e)
                self._flush_pending(f"connector '{self.name}' lost: {e}")
                hub.emit("mcp", manager.describe_all())
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            self.status = "connecting"

    async def _serve_once(self):
        from mcp import ClientSession

        if self.cfg.get("url"):
            try:
                from mcp.client.streamable_http import streamablehttp_client as http_client
            except ImportError:
                from mcp.client.streamable_http import streamable_http_client as http_client
            ctx = http_client(self.cfg["url"])
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(
                command=self.cfg["command"],
                args=list(self.cfg.get("args", [])),
                env=self.cfg.get("env") or None,
            )
            ctx = stdio_client(params)

        async with ctx as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=60)
                listed = await asyncio.wait_for(session.list_tools(), timeout=30)
                self.tools = [{
                    "name": t.name,
                    "description": (t.description or "")[:200],
                    "schema": getattr(t, "inputSchema", None) or {},
                } for t in listed.tools]
                self.status = "connected"
                self.error = ""
                log.info("MCP '%s' connected: %d tools", self.name, len(self.tools))
                hub.emit("mcp", manager.describe_all())
                while True:
                    tool, args, fut = await self.queue.get()
                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(tool, args or {}), timeout=CALL_TIMEOUT)
                        if not fut.done():
                            fut.set_result(_result_text(result))
                    except Exception as e:
                        if not fut.done():
                            fut.set_result(f"TOOL ERROR: {e}")
                        raise

    def _flush_pending(self, msg: str):
        while not self.queue.empty():
            try:
                _, _, fut = self.queue.get_nowait()
                if not fut.done():
                    fut.set_result(f"TOOL ERROR: {msg}")
            except asyncio.QueueEmpty:
                break


class Manager:
    def __init__(self):
        self.servers: dict[str, ServerConn] = {}

    def _load_config(self) -> dict:
        try:
            return json.loads(CONFIG_PATH.read_text())
        except FileNotFoundError:
            return {"servers": {}}
        except Exception as e:
            log.warning("bad mcp.json: %s", e)
            return {"servers": {}}

    def _save_config(self):
        config.ensure_dirs()
        CONFIG_PATH.write_text(json.dumps(
            {"servers": {n: c.cfg for n, c in self.servers.items()}}, indent=2))

    def start_all(self):
        for name, cfg in self._load_config().get("servers", {}).items():
            if name not in self.servers:
                self.servers[name] = ServerConn(name, cfg)
                self.servers[name].start()

    async def add(self, name: str, cfg: dict) -> dict:
        old = self.servers.pop(name, None)
        if old:
            await old.stop()
        conn = ServerConn(name, cfg)
        self.servers[name] = conn
        conn.start()
        self._save_config()
        # give it a moment so the UI sees a real status
        for _ in range(40):
            if conn.status in ("connected", "error"):
                break
            await asyncio.sleep(0.5)
        hub.emit("mcp", self.describe_all())
        return conn.describe()

    async def remove(self, name: str) -> bool:
        conn = self.servers.pop(name, None)
        if not conn:
            return False
        await conn.stop()
        self._save_config()
        hub.emit("mcp", self.describe_all())
        return True

    def describe_all(self) -> list[dict]:
        return [c.describe() for c in self.servers.values()]

    def tool_map(self) -> dict[str, dict]:
        """{'server:tool': {conn, description, schema}} for all connected servers."""
        out = {}
        for conn in self.servers.values():
            if conn.status != "connected":
                continue
            for t in conn.tools:
                out[f"{conn.name}:{t['name']}"] = {
                    "conn": conn, "tool": t["name"],
                    "description": t["description"], "schema": t.get("schema") or {}}
        return out

    async def call(self, qualified: str, args: dict) -> str:
        info = self.tool_map().get(qualified)
        if not info:
            return f"TOOL ERROR: unknown tool '{qualified}'"
        return await info["conn"].call(info["tool"], args)


manager = Manager()
