import html
import json
import logging
import re

import httpx

from . import config, db, llm
from .events import hub

log = logging.getLogger("diana.agents")

DEFAULT_AGENTS = [
    {
        "name": "Scout",
        "role": "Researcher",
        "tools": ["web_search", "fetch_url"],
        "system_prompt": (
            "You are Scout, a research agent on Diana's team. You dig up facts, compare "
            "options, and report findings with sources. Be thorough but concise. When you "
            "use web results, cite the URLs you relied on. Present findings as clear, "
            "well-structured markdown."
        ),
    },
    {
        "name": "Sage",
        "role": "Analyst & Strategist",
        "tools": [],
        "system_prompt": (
            "You are Sage, an analyst agent on Diana's team. You reason carefully about "
            "trade-offs, plans, numbers, and decisions. Structure your output: situation, "
            "analysis, recommendation. Be decisive — always land on a recommendation."
        ),
    },
    {
        "name": "Forge",
        "role": "Engineer",
        "tools": [],
        "system_prompt": (
            "You are Forge, an engineering agent on Diana's team. You write clean, working, "
            "well-commented code and explain how to run it. Prefer complete runnable files "
            "over fragments. State assumptions explicitly."
        ),
    },
    {
        "name": "Quill",
        "role": "Writer",
        "tools": [],
        "system_prompt": (
            "You are Quill, a writing agent on Diana's team. You draft emails, documents, "
            "posts, and summaries with excellent structure and tone. Match the requested "
            "voice; default to clear and warm. Deliver a finished draft, not an outline."
        ),
    },
    {
        "name": "Cipher",
        "role": "Coder",
        "tools": [],
        "system_prompt": (
            "You are Cipher, the coding agent on Diana's team. You implement features, "
            "write algorithms, debug, and refactor. Deliver complete, runnable, idiomatic "
            "code with brief usage notes — never pseudocode or fragments. State language "
            "and versions, handle edge cases, and keep code readable over clever."
        ),
    },
    {
        "name": "Pixel",
        "role": "Designer",
        "tools": [],
        "system_prompt": (
            "You are Pixel, the design agent on Diana's team. You handle UI/UX, visual "
            "identity, layouts, color, typography, and design systems. Deliver concrete "
            "artifacts: design specs, CSS/design tokens, HTML mockups, or precise "
            "component descriptions — always with accessibility (contrast, touch "
            "targets) and a consistent visual language in mind."
        ),
    },
    {
        "name": "Atlas",
        "role": "Planner",
        "tools": [],
        "system_prompt": (
            "You are Atlas, the planning agent on Diana's team. You turn goals into "
            "executable plans: milestones, task breakdowns, dependencies, estimates, "
            "risks and mitigations. Deliver structured, prioritized plans someone could "
            "start executing today — concrete next actions, not vague phases."
        ),
    },
    {
        "name": "Nova",
        "role": "AI/ML Specialist",
        "tools": ["web_search", "fetch_url"],
        "system_prompt": (
            "You are Nova, the AI and machine-learning specialist on Diana's team. You "
            "cover model architectures, training and fine-tuning, LLM tooling, datasets, "
            "and applied ML engineering. Give technically precise answers with concrete "
            "configs, commands, and trade-offs; cite sources when you use the web."
        ),
    },
]


def seed_agents():
    existing = {a["name"] for a in db.all_agents()}
    for a in DEFAULT_AGENTS:
        if a["name"] not in existing:
            db.upsert_agent(a["name"], a["role"], a["system_prompt"],
                            tools=a["tools"], builtin=True)


# ---------------- tools available to workers ----------------

async def web_search(query: str, max_results: int = 6) -> str:
    """DuckDuckGo HTML search — no API key needed."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh) Diana/1.0"},
            )
        results = []
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            r.text, flags=re.DOTALL,
        ):
            url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            uddg = re.search(r"uddg=([^&]+)", url)
            if uddg:
                from urllib.parse import unquote
                url = unquote(uddg.group(1))
            results.append(f"- {html.unescape(title)}\n  {url}")
            if len(results) >= max_results:
                break
        return "\n".join(results) or "No results found."
    except Exception as e:
        return f"Search failed: {e}"


async def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a page and return readable-ish text."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) Diana/1.0"})
        text = r.text
        text = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ",
                      text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        return text[:max_chars] or "Page had no readable text."
    except Exception as e:
        return f"Fetch failed: {e}"


TOOLS = {"web_search": web_search, "fetch_url": fetch_url}

BUILTIN_TOOL_DOCS = {
    "web_search": ('{"query": "..."}', "search the web (DuckDuckGo)"),
    "fetch_url": ('{"url": "https://..."}', "fetch a web page as readable text"),
}


def _schema_hint(schema: dict) -> str:
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    if not props:
        return "{}"
    parts = []
    for k, v in list(props.items())[:6]:
        mark = "" if k in required else "?"
        parts.append(f'"{k}{mark}": <{v.get("type", "any")}>')
    return "{" + ", ".join(parts) + "}"


def _tool_instructions(agent: dict) -> tuple[str, set[str]]:
    """Build the tool prompt from the agent's builtin tools + all MCP connectors."""
    from .mcp_client import manager
    lines = []
    available: set[str] = set()
    for t in agent.get("tools", []):
        if t in TOOLS:
            args, desc = BUILTIN_TOOL_DOCS[t]
            lines.append(f'- {t} — {desc}. args: {args}')
            available.add(t)
    mcp_tools = manager.tool_map()
    for name, info in list(mcp_tools.items())[:24]:
        lines.append(f'- {name} — {info["description"] or "MCP tool"}. '
                     f'args: {_schema_hint(info["schema"])}')
        available.add(name)
    if len(mcp_tools) > 24:
        lines.append(f"…and {len(mcp_tools) - 24} more MCP tools (same call format).")
    if not lines:
        return "", available
    instructions = (
        "\nYou can use tools. Available tools:\n" + "\n".join(lines) +
        '\n\nTo use a tool, reply with ONLY a JSON object:\n'
        '{"tool": "<tool name exactly as listed>", "args": {...}}\n\n'
        'When you have everything you need, reply with ONLY:\n'
        '{"final": "<your complete answer in markdown>"}\n\n'
        "Use at most a few tool calls, then give your final answer.\n")
    return instructions, available


async def _dispatch_tool(name: str, args: dict) -> str:
    from .mcp_client import manager
    if name in TOOLS:
        try:
            return await TOOLS[name](**args)
        except TypeError as e:
            return f"Bad arguments: {e}"
    return await manager.call(name, args or {})


def _skill_block(agent: dict, task_text: str, max_full: int = 3) -> str:
    """Agents may carry many skills: list them all briefly, but inject full
    content only for the few most relevant to this task."""
    skills = [s for s in (db.get_skill(n) for n in agent.get("skills", [])) if s]
    if not skills:
        return ""
    words = set(re.findall(r"[a-z]{3,}", task_text.lower()))

    def score(s):
        hay = set(re.findall(r"[a-z]{3,}",
                             (s["name"] + " " + (s["description"] or "")).lower()))
        return len(words & hay)

    ranked = sorted(skills, key=score, reverse=True)
    chosen = [s for s in ranked[:max_full] if score(s) > 0]

    index = "\n".join(f"- {s['name']}: {(s['description'] or '')[:90]}"
                      for s in skills[:30])
    if len(skills) > 30:
        index += f"\n…and {len(skills) - 30} more."
    block = f"\n\nSkills you have been trained in:\n{index}"
    if chosen:
        block += ("\n\nThe skills most relevant to this task, in full — "
                  "apply them:\n\n")
        block += "\n\n".join(f"### Skill: {s['name']}\n{s['content'][:5000]}"
                             for s in chosen)
    return block


async def run_task(task: dict, agent: dict) -> str:
    """Execute one task with a worker agent. Returns the agent's final answer."""
    mission = db.get_task(task["mission_id"]) if task["mission_id"] else None
    sibling_context = ""
    if mission:
        done = [t for t in db.children(mission["id"])
                if t["status"] == "done" and t["id"] != task["id"] and t["result"]]
        if done:
            sibling_context = "\n\nCompleted sibling tasks you can build on:\n" + "\n".join(
                f"- {t['title']}:\n{(t['result'] or '')[:1500]}" for t in done[-4:]
            )

    tool_instructions, available_tools = _tool_instructions(agent)
    system = agent["system_prompt"] + _skill_block(
        agent, f"{task['title']} {task['description']}")
    if tool_instructions:
        system += "\n" + tool_instructions

    user = ""
    if mission and mission["id"] != task["id"]:
        user += f"Overall mission: {mission['title']}\n{mission['description']}\n\n"
    user += f"Your task: {task['title']}\n{task['description']}"
    if task.get("feedback"):
        user += ("\n\nYour previous attempt was rejected by Diana with this feedback — "
                 f"fix it:\n{task['feedback']}")
    user += sibling_context

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    model = agent.get("model") or config.WORKER_MODEL
    for _ in range(6):
        out = await llm.chat(messages, model=model, temperature=0.7)
        obj = llm.extract_json(out) if available_tools else None
        if obj and "final" in obj:
            return str(obj["final"])
        if obj and obj.get("tool") in available_tools:
            tool_name = obj["tool"]
            args = obj.get("args") or {}
            hub.emit("log", {"agent": agent["name"], "task_id": task["id"],
                             "text": f"{tool_name}({json.dumps(args)[:120]})"})
            result = await _dispatch_tool(tool_name, args)
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user",
                             "content": f"Tool result for {tool_name}:\n{result[:8000]}\n\n"
                                        "Continue. Remember to finish with a final JSON."})
            continue
        return out  # plain answer
    return out
