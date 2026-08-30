# Diana — project guide

Personal supervising AI ("Alfred/JARVIS"): voice in/out, delegates missions to worker
agents, evaluates their output, learns skills from GitHub, uses MCP tools, runs
schedules. Fully local (Ollama on the host Mac), dockerized, LAN-reachable.

## Run / develop

```bash
docker compose up -d --build     # build + run; restart: unless-stopped
docker logs -f diana             # server logs
docker exec diana python -m unittest diana.tests.test_core -v   # unit tests
```

- UI: http://localhost:8080 · phone: https://<DIANA_LAN_IP>:8443 (HTTPS needed for mic)
- Config: `.env` (never commit) — models, voice, `DIANA_LAN_IP`, optional `DIANA_TOKEN`, `TZ`
- All state lives in the `diana-data` volume (`/data`): SQLite, skills, certs,
  whisper/piper models, MCP config, agent workspace. `docker compose down -v` wipes her.

## Architecture (server/diana/)

| Module | Role |
|---|---|
| `main.py` | FastAPI app, all endpoints, token-auth middleware, lifespan boot |
| `supervisor.py` | Diana's persona/prompt, JSON action protocol, streaming reply extractor, task evaluation, mission debriefs |
| `tasks.py` | background engine: pending→running→review→done/retry, sequential-mission gating, 25-min stuck-task watchdog |
| `agents.py` | worker roster + run loop: skill-relevance injection, JSON tool loop (builtin + MCP), per-agent model override, "Tools used" trail |
| `mcp_client.py` | MCP connectors: persistent sessions, auto-reconnect, `/data/mcp.json` |
| `scheduler.py` | standing schedules (`every N min/h`, `daily`, `weekly`, `once`), container-local TZ |
| `skills_import.py` | GitHub skill learning: single-doc distillation or bulk SKILL.md repo scan |
| `llm.py` | Ollama client: URL auto-probe, chat + chat_stream, think-mode handling, tolerant JSON extraction |
| `voice_stt.py` / `voice_tts.py` | faster-whisper ears; Piper voice → edge-tts → browser fallback |
| `db.py` | SQLite (WAL): messages, tasks, agents, skills, memories, settings, schedules; additive `MIGRATIONS` list |
| `certs.py` / `run.py` | self-managed CA + LAN cert (SANs from `DIANA_LAN_IP`); dual HTTP/HTTPS uvicorn entrypoint |
| `static/` | single-page UI (vanilla JS): chat + streaming, wake-word VAD loop, mission tree, team rail, PWA |

## Conventions & gotchas

- LLM output is never trusted: everything goes through `llm.extract_json` (fences,
  trailing commas, prose around JSON all tolerated).
- Diana's reply is streamed by scanning the JSON `"reply"` field incrementally
  (`supervisor._ReplyStream`) — the persona prompt requires `reply` first.
- Model settings resolve at call time via `config.diana_model()` / `worker_model()`
  (DB settings override env). Never read `config.DIANA_MODEL` directly for calls.
- Agents' skills/tools are injected by lexical relevance to the task — full content
  for top ~3 skills, top ~30 MCP tools listed, everything still dispatchable.
- `@playwright/mcp` must run via the **global** `playwright-mcp` binary with
  `--browser chromium`; the Dockerfile bakes its exact browser build
  (`playwright-mcp install-browser chrome-for-testing`). `npx -y` drifts versions.
- Ollama URL auto-probes: env → host.docker.internal → ollama → localhost.
- Schema changes: append `ALTER TABLE …` to `db.MIGRATIONS` (idempotent try/except).
- Per instruction from the owner: do not commit or push this repo unless asked.
