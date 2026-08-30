# Diana

> Your personal supervising AI — Alfred to your Batman, JARVIS to your Iron Man.

Diana is a fully local, dockerized AI chief-of-staff. You talk to her (voice or text);
she breaks objectives into missions, delegates tasks to a team of worker agents,
evaluates their output, retries what isn't good enough, and reports back — all
visualized live in a mission tree.

Everything runs on your machine: **Ollama** for brains, **faster-whisper** for ears,
**Piper** for her voice.

## Architecture

```
Browser (any device on your LAN)
  │  voice / text ⇄ WebSocket live updates
  ▼
diana (Docker, FastAPI, auto-restarts)
  ├─ Supervisor  — Diana's persona: plans, delegates, evaluates, debriefs
  ├─ Task engine — background executor, retries, mission completion
  ├─ Agent pool  — Scout (research + web), Sage (analysis), Forge (code), Quill (writing)
  ├─ Skills      — markdown skill docs Diana writes and teaches to agents
  ├─ STT         — faster-whisper (local)
  └─ TTS         — Piper (local) → edge-tts → browser voice fallback
  ▼
Ollama (your host install — Apple-Silicon fast)
```

## Run

```bash
docker compose up -d --build
```

Then open **http://localhost:8080** — or from your phone on the same Wi-Fi,
**https://\<your-mac-ip\>:8443** (HTTPS is required for the microphone on phones;
set `DIANA_LAN_IP` in `.env` so the certificate matches your IP).

On first visit the phone shows a certificate warning (Diana signs her own certs) —
tap through it once, or make it disappear forever: download
`http://<your-mac-ip>:8080/ca` on the phone and trust it
(iOS: Settings → General → VPN & Device Management → install profile, then
Settings → General → About → Certificate Trust Settings → enable "Diana Local CA".
Android: Settings → Security → Install a certificate → CA certificate).

Requirements:
- Docker Desktop (enable *Start Docker Desktop when you sign in* so Diana survives reboots)
- Ollama running on the host with the models in `.env` pulled
  (defaults: `qwen3.6:latest` for Diana, `mistral:latest` for workers)

> The container reaches your Mac's Ollama via `host.docker.internal` — this works
> out of the box with Docker Desktop. If Diana ever reports her brain offline while
> Ollama is running, set `launchctl setenv OLLAMA_HOST 0.0.0.0` and restart Ollama.

## Configure

Copy `.env.example` to `.env` and edit. Notables:

| Variable | Default | Meaning |
|---|---|---|
| `DIANA_MODEL` | `qwen3.6:latest` | Diana's own brain |
| `WORKER_MODEL` | `mistral:latest` | model her worker agents use |
| `STT_MODEL` | `base` | whisper size (`small` = better, slower) |
| `TTS_ENGINE` | `piper` | `piper` local · `edge` cloud · `off` browser voice |
| `WORKER_CONCURRENCY` | `2` | parallel tasks |

## What Diana can do

- **Converse** — mic button or typing; she answers aloud.
- **Missions** — "research X and write me a summary" → she creates a mission with
  subtasks, assigns agents, shows live progress in the tree.
- **Evaluate** — every task result is judged by Diana; failures get feedback and a retry.
- **Grow the team** — "create an agent who specializes in …" adds a new worker.
- **Teach skills** — "write a skill for negotiating and teach it to Quill" creates a
  reusable markdown skill (also mirrored to the `diana-data` volume under `/data/skills`).
- **Web research** — Scout can search (DuckDuckGo) and read pages.
- **Learn from GitHub** — paste a link in chat ("Diana, learn this: https://github.com/…")
  or use the **+** button in the Skill Library. She fetches the markdown (file, skill
  folder, or repo), distills it into an operational skill, and can teach it to an agent.
- **Always ready** — flip the ear toggle in the Conversation panel and just say
  **"Diana, …"** out loud. The browser tab does local voice-activity detection and the
  server checks the wake word with whisper — no cloud, no keyboard. Saying just "Diana"
  gets a "Yes?" and opens a 12-second hot window where the next thing you say is the
  command. The tab must stay open (it's the microphone); it works minimized/backgrounded.

- **MCP connectors** — plug any Model Context Protocol server into the team (Team tab →
  Connectors → **+**, or edit `/data/mcp.json`). Remote URLs and local commands both work
  (`npx` and `uvx` are available in the container). Preconfigured: `time`,
  `files` (a real read/write workspace at `/data/workspace`), `memory` (a knowledge
  graph), `fetch` (clean page extraction), `browser` (a real headless Chromium via
  Playwright MCP — navigate, click, read live websites), `git` (repo operations in the
  workspace), `sqlite` (a scratch analytics database), `markitdown` (convert
  PDFs/Office docs/URLs to markdown), and `thinking` (structured reasoning scratchpad).
  Tools are relevance-ranked into each task's prompt, and results carry a "Tools used"
  trail. `POST /api/mcp/{name}/call` invokes any tool directly. For servers needing
  secrets, the add-connector dialog has a KEY=VALUE box (env vars for commands, HTTP
  headers for URLs) — e.g. GitHub:
  `npx -y @modelcontextprotocol/server-github` + `GITHUB_PERSONAL_ACCESS_TOKEN=<PAT>`.
- **Streaming replies** — Diana's answers stream into the chat token by token, and with
  voice on she starts speaking sentence-by-sentence before the full reply is done.
  Talking to her (mic) interrupts her mid-sentence.
- **Scheduled missions** — "every morning at 9, brief me on X" creates a standing
  schedule (specs: `every N minutes/hours`, `daily HH:MM`, `weekly mon HH:MM`,
  `once YYYY-MM-DDTHH:MM`; container runs in `TZ`, default America/Chicago). Manage
  them in Team → Schedules.
- **Runtime model switching** — click the model chip in the top bar to swap Diana's
  brain or the worker model live, no restart.
- **Sequential missions** — when steps depend on each other, Diana marks the mission
  sequential: each step waits for the previous one and builds on its result.
- **Activity feed** — a live ledger under the mission tree: which agent started what,
  every tool call, and each review verdict (passed / sent back / failed), timestamped.
- **Optional access token** — set `DIANA_TOKEN` in `.env` and every device must open
  `/?token=<value>` once (cookie remembers it). Off by default; `/ca` and health stay open.
- **Tests** — `docker exec diana python -m unittest diana.tests.test_core -v`
  (25 pure-logic tests: JSON extraction, reply streaming, schedule specs, skill parsing).
- **Housekeeping** — new-conversation button, archive-finished-missions button, delete
  skills/agents/memories/schedules from the UI, desktop notifications for mission
  debriefs when the tab is in the background, installable as a home-screen app (PWA),
  and **Download backup** (bottom of the Team rail) exports a zip of the database,
  skills, workspace and connector config.
- **Long-term memory** — tell Diana "remember: …" and it persists across restarts;
  she recalls it in every conversation and can be told to forget.
- **Per-agent models** — each agent can run its own Ollama model (Cipher uses
  `qwen2.5-coder:32b`, Nova uses `qwen3.6:latest`; the rest use `WORKER_MODEL`).

## Data

Everything persists in the `diana-data` Docker volume: SQLite DB (conversations,
missions, agents, skills), whisper models, Piper voices. `docker compose down` keeps it;
`docker compose down -v` wipes her memory.

## First start

First boot downloads the whisper model (~150 MB) and the Piper voice (~60 MB) into the
volume — the first voice interaction may take a minute. After that, everything is warm.
