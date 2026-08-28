import asyncio
import logging
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agents, config, db, llm, skills_import, supervisor, voice_stt, voice_tts
from .events import hub
from .mcp_client import manager as mcp
from .tasks import engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("diana")

STATIC = Path(__file__).parent / "static"


_booted = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # two uvicorn servers (http + https) share this app; boot only once
    global _booted
    if not _booted:
        _booted = True
        config.ensure_dirs()
        db.init()
        agents.seed_agents()
        hub.bind_loop(asyncio.get_running_loop())
        voice_stt.warm()
        asyncio.create_task(engine.run_forever())
        asyncio.create_task(_boot_probe())
        mcp.start_all()
        log.info("Diana is online: http %s / https %s", config.PORT, config.TLS_PORT)
    yield


async def _boot_probe():
    url = await llm.base_url(force=True)
    if url:
        ok_diana = await llm.has_model(config.DIANA_MODEL)
        ok_worker = await llm.has_model(config.WORKER_MODEL)
        detail = f"brain: {config.DIANA_MODEL} @ {url}"
        if not ok_diana or not ok_worker:
            missing = [m for m, ok in ((config.DIANA_MODEL, ok_diana),
                                       (config.WORKER_MODEL, ok_worker)) if not ok]
            detail += f" — missing models: {', '.join(missing)} (ollama pull them)"
        await hub.set_status("idle", detail)
    else:
        await hub.set_status("offline", "Ollama unreachable — start Ollama, then reload")


app = FastAPI(title="Diana", lifespan=lifespan)


class MessageIn(BaseModel):
    text: str


class TTSIn(BaseModel):
    text: str


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/ca")
async def ca_cert():
    """Diana's CA certificate — install & trust on a phone for a clean padlock."""
    from . import certs
    if not certs.CA_CRT.exists():
        return JSONResponse({"error": "CA not generated yet"}, status_code=404)
    return FileResponse(certs.CA_CRT, media_type="application/x-x509-ca-cert",
                        filename="diana-ca.crt")


@app.get("/api/health")
async def health():
    return {"ok": True, "status": hub.status}


@app.get("/api/state")
async def state():
    url = await llm.base_url()
    return {
        "status": hub.status,
        "ollama": url,
        "models": {"diana": config.DIANA_MODEL, "worker": config.WORKER_MODEL},
        "messages": db.recent_messages(60),
        "tree": db.task_tree(),
        "agents": db.all_agents(),
        "skills": db.all_skills(),
        "mcp": mcp.describe_all(),
    }


@app.post("/api/message")
async def message(body: MessageIn):
    text = body.text.strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    row = await supervisor.handle_user_message(text, source="text")
    engine.kick()
    return {"reply": row["content"]}


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...)):
    data = await audio.read()
    if len(data) < 200:
        return JSONResponse({"error": "no audio"}, status_code=400)
    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1]
    await hub.set_status("thinking", "transcribing…")
    try:
        text = await voice_stt.transcribe(data, suffix=suffix)
    except Exception as e:
        log.exception("stt failed")
        await hub.set_status("idle")
        return JSONResponse({"error": f"transcription failed: {e}"}, status_code=500)
    if not text:
        await hub.set_status("idle")
        return {"transcript": "", "reply": ""}
    row = await supervisor.handle_user_message(text, source="voice")
    engine.kick()
    return {"transcript": text, "reply": row["content"]}


WAKE_RE = re.compile(r"\b(diana|dianna|deanna|dyana|dayana)\b[,.!?:;]*", re.IGNORECASE)
WAKE_ACKS = ["Yes?", "I'm listening.", "Go ahead.", "At your service.", "I'm here."]


@app.post("/api/wake")
async def wake(audio: UploadFile = File(...), hot: str = Form("0")):
    """Always-ready mode: transcribe an utterance; act only if it addresses Diana
    (or arrives inside the hot window right after a wake)."""
    data = await audio.read()
    if len(data) < 1000:
        return {"wake": False}
    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1]
    try:
        text = await voice_stt.transcribe(data, suffix=suffix)
    except Exception as e:
        log.warning("wake transcription failed: %s", e)
        return {"wake": False}
    if not text or len(text) < 2:
        return {"wake": False}

    m = WAKE_RE.search(text)
    is_hot = hot == "1"
    if not m and not is_hot:
        return {"wake": False, "transcript": text}

    if m:
        after = text[m.end():].strip(" ,.!?:;")
        before = text[:m.start()].strip(" ,.!?:;")
        command = after or before
    else:
        command = text.strip()

    if not command:
        await hub.set_status("listening", "say your command…")
        return {"wake": True, "ack": True, "reply": random.choice(WAKE_ACKS)}

    row = await supervisor.handle_user_message(command, source="voice")
    engine.kick()
    return {"wake": True, "transcript": command, "reply": row["content"]}


class ImportIn(BaseModel):
    url: str
    teach_to: str | None = None


@app.post("/api/skills/import")
async def import_skill(body: ImportIn):
    url = body.url.strip()
    if not url:
        return JSONResponse({"error": "no url"}, status_code=400)
    try:
        info = await skills_import.import_from_url(url, teach_to=body.teach_to)
    except Exception as e:
        log.warning("skill import failed for %s: %s", url, e)
        return JSONResponse({"error": str(e)}, status_code=400)
    return info


@app.post("/api/tts")
async def tts(body: TTSIn):
    out = await voice_tts.synthesize(body.text)
    if not out:
        return Response(status_code=204)
    audio_bytes, mime = out
    return Response(content=audio_bytes, media_type=mime)


class AgentIn(BaseModel):
    name: str
    role: str
    system_prompt: str
    tools: list[str] = []
    model: str | None = None


class AgentSkillsIn(BaseModel):
    skills: list[str]


@app.post("/api/agents")
async def create_agent(body: AgentIn):
    name = body.name.strip()[:24]
    if not name:
        return JSONResponse({"error": "no name"}, status_code=400)
    existing = db.get_agent(name)
    db.upsert_agent(name, body.role, body.system_prompt, tools=body.tools,
                    model=body.model or (existing or {}).get("model"))
    await hub.broadcast("agents", db.all_agents())
    return db.get_agent(name)


@app.post("/api/agents/{name}/skills")
async def set_agent_skills(name: str, body: AgentSkillsIn):
    agent = db.get_agent(name)
    if not agent:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    known = {s["name"] for s in db.all_skills()}
    valid = [s for s in dict.fromkeys(body.skills) if s in known]
    db.set_agent_skills(name, valid)
    await hub.broadcast("agents", db.all_agents())
    return {"agent": name, "skills": valid,
            "unknown": [s for s in body.skills if s not in known]}


class MCPIn(BaseModel):
    name: str
    url: str | None = None
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}


@app.get("/api/mcp")
async def mcp_list():
    return mcp.describe_all()


@app.post("/api/mcp")
async def mcp_add(body: MCPIn):
    name = body.name.strip()[:32]
    if not name:
        return JSONResponse({"error": "no name"}, status_code=400)
    if not body.url and not body.command:
        return JSONResponse({"error": "need a url or a command"}, status_code=400)
    cfg = {"url": body.url} if body.url else \
          {"command": body.command, "args": body.args, "env": body.env}
    return await mcp.add(name, {k: v for k, v in cfg.items() if v})


@app.delete("/api/mcp/{name}")
async def mcp_remove(name: str):
    ok = await mcp.remove(name)
    return {"removed": ok}


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str):
    t = db.get_task(task_id)
    if not t:
        return JSONResponse({"error": "not found"}, status_code=404)
    t["children"] = db.children(task_id)
    return t


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    desc = await supervisor.perform_action({"type": "cancel_task", "task_id": task_id})
    return {"result": desc}


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    t = db.get_task(task_id)
    if not t:
        return JSONResponse({"error": "not found"}, status_code=404)
    db.update_task(task_id, status="pending", attempts=0)
    if t["mission_id"]:
        m = db.get_task(t["mission_id"])
        if m and m["status"] in ("done", "failed"):
            db.update_task(m["id"], status="active")
    await hub.broadcast("tree", db.task_tree())
    engine.kick()
    return {"ok": True}


@app.get("/api/skills/{name}")
async def skill_detail(name: str):
    s = db.get_skill(name)
    if not s:
        return JSONResponse({"error": "not found"}, status_code=404)
    return s


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await hub.connect(websocket)
    try:
        await websocket.send_json({"type": "status",
                                   "data": {"status": hub.status, "detail": ""}})
        while True:
            await websocket.receive_text()  # client pings; content ignored
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception:
        hub.disconnect(websocket)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
