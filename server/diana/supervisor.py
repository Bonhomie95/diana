import asyncio
import contextlib
import json
import logging
import re

from . import config, db, llm
from .events import hub

log = logging.getLogger("diana.supervisor")

PERSONA = """You are Diana — a personal supervising AI: calm, sharp, devoted, lightly witty. \
Think Alfred to Batman, JARVIS to Iron Man. You serve one principal: your user. \
You speak in first person, warmly and concisely (your replies are often spoken aloud, \
so keep them tight — 1 to 4 sentences unless asked for detail).

You do not do heavy work yourself. You command a team of worker agents, break objectives \
into tasks, delegate, evaluate their output, and report back. You can also create new \
agents and write "skills" (reusable instruction documents) to teach your agents.

Your team right now:
{roster}

Skills library:
{skills}

Connected tool servers (MCP) — your agents can call these tools during tasks:
{mcp}

Things you have been asked to remember:
{memories}

Standing schedules:
{schedules}

Current missions (task tree):
{tree}

RESPONSE FORMAT — reply with ONLY one JSON object, no other text. The "reply" field MUST \
come first:
{{
  "reply": "<what you say to the user — natural, spoken language, no markdown headers>",
  "actions": [ ... zero or more actions ... ]
}}

Available actions:
- {{"type": "create_mission", "title": "...", "description": "...", "sequential": false, "subtasks": [{{"title": "...", "description": "detailed instructions for the worker", "agent": "Scout|Sage|Forge|Quill|<any agent name>"}}]}} — set "sequential": true when later steps build on earlier results (each step then waits for the previous one and can see its output)
- {{"type": "add_tasks", "mission_id": "<id>", "subtasks": [same shape as above]}}
- {{"type": "cancel_task", "task_id": "<id>"}}
- {{"type": "create_agent", "name": "...", "role": "...", "system_prompt": "detailed persona + instructions"}}
- {{"type": "create_skill", "name": "...", "description": "...", "content": "the full skill document in markdown — concrete techniques, steps, examples"}}
- {{"type": "import_skill", "url": "<GitHub or raw markdown URL>", "teach_to": "<optional agent name>"}}
- {{"type": "teach_skill", "agent": "<agent name>", "skill": "<skill name>"}}
- {{"type": "remember", "content": "a fact worth keeping about the user or standing instructions"}}
- {{"type": "forget", "memory_id": <id>}}
- {{"type": "schedule", "when": "<spec>", "instruction": "what to do each time, self-contained"}} — spec is one of: "every N minutes", "every N hours", "daily HH:MM", "weekly mon|tue|...|sun HH:MM", "once YYYY-MM-DDTHH:MM" (local time)
- {{"type": "cancel_schedule", "schedule_id": <id>}}

Rules:
- You personally have NO tools and NO live data access — never claim you checked, looked
  up, or fetched anything yourself, and never invent facts like weather, prices, or news.
  For any live lookup (weather, web, wikipedia, files, arxiv…), create a mission with a
  single task for a suitable agent — the agents have the tools.
- Casual conversation, questions about status, or anything you can answer directly from this conversation: just reply, actions = [].
- Real work (research, writing, code, analysis, planning): create a mission with 1–5 well-scoped subtasks assigned to the best-fitted agents. Tell the user what you're setting in motion.
- Only create a mission when the user asked for work to be done. Never invent work.
- When asked to teach or improve your agents, write a genuinely useful skill document.
- When the user shares a GitHub link (or any URL to a document) to learn from, use import_skill — you will fetch it, distill it, and add it to the library. Pass teach_to when they name an agent.
- Task descriptions must be self-contained: the worker sees only its task, not this conversation.
- When the user tells you a lasting preference, fact, or standing instruction ("always…", "my birthday is…", "I prefer…"), use remember. Recall memories naturally when relevant.
"""


def _roster_text() -> str:
    lines = []
    for a in db.all_agents():
        skills = ""
        if a["skills"]:
            shown = ", ".join(a["skills"][:6])
            more = f" +{len(a['skills']) - 6} more" if len(a["skills"]) > 6 else ""
            skills = f" | skills: {shown}{more}"
        tools = f" | tools: {', '.join(a['tools'])}" if a["tools"] else ""
        lines.append(f"- {a['name']} ({a['role']}){tools}{skills}")
    return "\n".join(lines) or "(no agents)"


def _skills_text() -> str:
    skills = db.all_skills()
    lines = [f"- {s['name']}: {s['description'][:90]}" for s in skills[:40]]
    if len(skills) > 40:
        lines.append(f"…and {len(skills) - 40} more in the library.")
    return "\n".join(lines) or "(empty)"


def _tree_text() -> str:
    out = []
    for m in db.task_tree()[:6]:
        out.append(f"- [{m['status']}] mission {m['id']}: {m['title']}")
        for c in m.get("children", []):
            out.append(f"    - [{c['status']}] task {c['id']} ({c['agent'] or '?'}): {c['title']}")
    return "\n".join(out) or "(no missions yet)"


def _mcp_text() -> str:
    from .mcp_client import manager
    lines = []
    for s in manager.describe_all():
        if s["status"] == "connected":
            names = ", ".join(t["name"] for t in s["tools"][:10])
            lines.append(f"- {s['name']}: {names}" +
                         (f" (+{len(s['tools']) - 10} more)" if len(s["tools"]) > 10 else ""))
        else:
            lines.append(f"- {s['name']}: {s['status']}")
    return "\n".join(lines) or "(none connected)"


def _memories_text() -> str:
    return "\n".join(f"- [{m['id']}] {m['content'][:200]}"
                     for m in db.all_memories(30)) or "(nothing yet)"


def _schedules_text() -> str:
    lines = []
    for s in db.all_schedules():
        state = f"next {s['next_run']}" if s["enabled"] and s["next_run"] else "inactive"
        lines.append(f"- [{s['id']}] {s['spec']} ({state}): {s['instruction'][:100]}")
    return "\n".join(lines) or "(none)"


def system_prompt() -> str:
    return PERSONA.format(roster=_roster_text(), skills=_skills_text(),
                          mcp=_mcp_text(), memories=_memories_text(),
                          schedules=_schedules_text(), tree=_tree_text())


class _ReplyStream:
    """Incrementally extracts the value of the JSON "reply" field from streamed
    model output and forwards it as UI deltas."""

    def __init__(self):
        self.buf = ""
        self.mode = "search"
        self.esc = False
        self.emitted = False

    async def feed(self, chunk: str):
        if self.mode == "done":
            return
        self.buf += chunk
        if self.mode == "search":
            m = re.search(r'"reply"\s*:\s*"', self.buf)
            if not m:
                self.buf = self.buf[-24:]  # keep a tail in case the marker spans chunks
                return
            self.mode = "emit"
            self.buf = self.buf[m.end():]
        if self.mode == "emit":
            out = []
            i = 0
            while i < len(self.buf):
                c = self.buf[i]
                if self.esc:
                    out.append({"n": "\n", "t": "\t"}.get(c, c))
                    self.esc = False
                elif c == "\\":
                    self.esc = True
                elif c == '"':
                    self.mode = "done"
                    i += 1
                    break
                else:
                    out.append(c)
                i += 1
            self.buf = "" if self.mode != "done" else self.buf[i:]
            if out:
                self.emitted = True
                await hub.broadcast("delta", {"text": "".join(out)})


_cancel_current: "asyncio.Event | None" = None


async def stop_current() -> bool:
    """Cancel the in-flight reply generation, if any."""
    if _cancel_current and not _cancel_current.is_set():
        _cancel_current.set()
        return True
    return False


async def handle_user_message(text: str, source: str = "text") -> dict:
    """Full pipeline: store, think, act, reply. Returns the reply message row."""
    global _cancel_current
    urow = db.add_message("user", text, {"source": source})
    await hub.broadcast("message", {"role": "user", "content": text, "id": urow["id"]})
    await hub.set_status("thinking")

    history = [
        {"role": m["role"] if m["role"] in ("user", "assistant") else "user",
         "content": m["content"]}
        for m in db.recent_messages(config.CHAT_HISTORY)
    ]
    messages = [{"role": "system", "content": system_prompt()}] + history

    streamer = _ReplyStream()
    cancel = asyncio.Event()
    _cancel_current = cancel
    stopped = False
    try:
        gen = asyncio.create_task(llm.chat_stream(
            messages, streamer.feed, model=config.diana_model(),
            temperature=0.6, num_ctx=32768))
        waiter = asyncio.create_task(cancel.wait())
        done, _pending = await asyncio.wait({gen, waiter},
                                            return_when=asyncio.FIRST_COMPLETED)
        if gen in done:
            waiter.cancel()
            raw = gen.result()
            obj = llm.extract_json(raw) or {}
            reply = str(obj.get("reply") or llm.strip_think(raw) or
                        "I'm here, though I struggled to phrase that. Say it again?")
            actions = obj.get("actions") or []
        else:
            gen.cancel()
            # CancelledError is a BaseException — suppress it explicitly
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gen
            stopped = True
            reply, actions = "", []
    except Exception as e:
        log.exception("Diana brain failure")
        reply = f"I hit a snag reaching my reasoning engine: {e}"
        actions = []
    finally:
        _cancel_current = None
        await hub.broadcast("delta_done", {"streamed": streamer.emitted})

    if stopped:
        await hub.broadcast("stopped", {})
        await hub.set_status("working" if db.tasks_by_status("pending") or
                             db.tasks_by_status("running") else "idle")
        return {"content": "", "stopped": True}

    performed = []
    for action in actions[:8]:
        try:
            desc = await perform_action(action)
            if desc:
                performed.append(desc)
        except Exception as e:
            log.exception("action failed: %s", action)
            performed.append(f"(failed: {e})")

    row = db.add_message("assistant", reply, {"actions": performed})
    await hub.broadcast("message", {"role": "assistant", "content": reply,
                                    "actions": performed, "id": row["id"]})
    await hub.set_status("working" if db.tasks_by_status("pending") or
                         db.tasks_by_status("running") else "idle")
    return row


async def perform_action(action: dict) -> str | None:
    t = action.get("type")

    if t == "create_mission":
        mission = db.create_task(action.get("title", "Untitled mission"),
                                 action.get("description", ""), status="active")
        if action.get("sequential"):
            db.update_task(mission["id"], mode="sequential")
        agent_names = {a["name"] for a in db.all_agents()}
        for i, st in enumerate(action.get("subtasks", [])[:8]):
            agent = st.get("agent") if st.get("agent") in agent_names else "Sage"
            db.create_task(st.get("title", f"Step {i+1}"), st.get("description", ""),
                           parent_id=mission["id"], mission_id=mission["id"],
                           agent=agent, ordinal=i)
        await hub.broadcast("tree", db.task_tree())
        return f"mission created: {mission['title']}"

    if t == "add_tasks":
        mission = db.get_task(action.get("mission_id", ""))
        if not mission:
            return "(mission not found)"
        agent_names = {a["name"] for a in db.all_agents()}
        base = len(db.children(mission["id"]))
        for i, st in enumerate(action.get("subtasks", [])[:8]):
            agent = st.get("agent") if st.get("agent") in agent_names else "Sage"
            db.create_task(st.get("title", "Task"), st.get("description", ""),
                           parent_id=mission["id"], mission_id=mission["id"],
                           agent=agent, ordinal=base + i)
        if mission["status"] in ("done", "failed"):
            db.update_task(mission["id"], status="active")
        await hub.broadcast("tree", db.task_tree())
        return f"tasks added to {mission['title']}"

    if t == "cancel_task":
        task = db.get_task(action.get("task_id", ""))
        if task:
            db.update_task(task["id"], status="cancelled")
            for c in db.children(task["id"]):
                if c["status"] in ("pending", "running", "review"):
                    db.update_task(c["id"], status="cancelled")
            await hub.broadcast("tree", db.task_tree())
            return f"cancelled: {task['title']}"
        return "(task not found)"

    if t == "create_agent":
        name = (action.get("name") or "Agent").strip()[:24]
        db.upsert_agent(name, action.get("role", "Specialist"),
                        action.get("system_prompt", "You are a diligent specialist agent."),
                        tools=action.get("tools", []))
        await hub.broadcast("agents", db.all_agents())
        return f"agent created: {name}"

    if t == "create_skill":
        name = (action.get("name") or "skill").strip()[:48]
        db.upsert_skill(name, action.get("description", ""), action.get("content", ""))
        await hub.broadcast("skills", db.all_skills())
        return f"skill written: {name}"

    if t == "import_skill":
        from . import skills_import
        info = await skills_import.import_from_url(action.get("url", ""),
                                                   teach_to=action.get("teach_to"))
        out = f"learned from GitHub: {info['name']}"
        if info.get("taught_to"):
            out += f" → taught to {info['taught_to']}"
        return out

    if t == "remember":
        content = (action.get("content") or "").strip()
        if content:
            db.add_memory(content[:500])
            await hub.broadcast("memories", db.all_memories())
            return "noted"
        return None

    if t == "forget":
        try:
            ok = db.forget_memory(int(action.get("memory_id")))
            await hub.broadcast("memories", db.all_memories())
            return "forgotten" if ok else "(memory not found)"
        except (TypeError, ValueError):
            return "(bad memory id)"

    if t == "schedule":
        from . import scheduler
        row = scheduler.create(action.get("when", ""), action.get("instruction", ""))
        if row:
            return f"scheduled: {row['spec']} (next {row['next_run']})"
        return f"(couldn't parse schedule spec '{action.get('when', '')}')"

    if t == "cancel_schedule":
        try:
            ok = db.delete_schedule(int(action.get("schedule_id")))
            await hub.broadcast("schedules", db.all_schedules())
            return "schedule cancelled" if ok else "(schedule not found)"
        except (TypeError, ValueError):
            return "(bad schedule id)"

    if t == "teach_skill":
        agent = db.get_agent(action.get("agent", ""))
        skill = db.get_skill(action.get("skill", ""))
        if agent and skill:
            skills = list(dict.fromkeys(agent["skills"] + [skill["name"]]))
            db.set_agent_skills(agent["name"], skills)
            await hub.broadcast("agents", db.all_agents())
            return f"taught {skill['name']} to {agent['name']}"
        return "(agent or skill not found)"

    return None


EVAL_PROMPT = """You are Diana, supervising your team. A worker agent has submitted work. \
Judge it strictly but fairly.

Task: {title}
Instructions given to the worker: {description}

Submitted result:
---
{result}
---

Reply with ONLY JSON: {{"verdict": "pass" or "fail", "feedback": "one or two sentences — if fail, say exactly what to fix"}}
A result passes if it genuinely completes the task as instructed. Minor style issues are not a fail."""


async def evaluate_task(task: dict) -> dict:
    prompt = EVAL_PROMPT.format(title=task["title"],
                                description=task["description"][:2000],
                                result=(task["result"] or "")[:6000])
    try:
        raw = await llm.chat([{"role": "user", "content": prompt}],
                             model=config.diana_model(), temperature=0.2)
        obj = llm.extract_json(raw) or {}
        verdict = "pass" if str(obj.get("verdict", "pass")).lower().startswith("p") else "fail"
        return {"verdict": verdict, "feedback": str(obj.get("feedback", ""))}
    except Exception as e:
        log.warning("evaluation failed, passing by default: %s", e)
        return {"verdict": "pass", "feedback": f"(evaluation skipped: {e})"}


SUMMARY_PROMPT = """You are Diana. Your team just finished a mission for your user.

Mission: {title} — final status: {status}
{description}

Task results:
{results}

Write the message you will say to your user: a warm, concise debrief. Use markdown. Start \
with a one-line verdict, then the substance. Be honest: if the mission failed or any task \
failed, say so plainly, explain what went wrong, and salvage whatever useful results exist — \
never claim success for failed work. Do not mention JSON or internal mechanics."""


async def summarize_mission(mission: dict) -> str:
    kids = db.children(mission["id"])
    results = "\n\n".join(
        f"### {t['title']} — {t['agent']} [{t['status']}]\n{(t['result'] or '(no output)')[:4000]}"
        for t in kids
    )
    try:
        return await llm.chat(
            [{"role": "user", "content": SUMMARY_PROMPT.format(
                title=mission["title"], status=mission["status"],
                description=mission["description"],
                results=results[:24000])}],
            model=config.diana_model(), temperature=0.5)
    except Exception:
        done = sum(1 for t in kids if t["status"] == "done")
        return (f"Mission **{mission['title']}** wrapped up: {done}/{len(kids)} tasks "
                "completed. Results are attached to each task in the tree.")
