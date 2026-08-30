"""Import skills from GitHub: fetch a markdown doc, have Diana distill and learn it."""
import logging
import re

import httpx

from . import config, db, llm
from .events import hub

log = logging.getLogger("diana.skills_import")

GITHUB_BLOB = re.compile(r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)")
GITHUB_TREE = re.compile(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+)(?:/(.*?))?)?/?$")
PREFERRED_FILES = ("skill.md", "readme.md", "index.md")


async def _get_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
        r = await c.get(url, headers={"User-Agent": "Diana/1.0",
                                      "Accept": "application/vnd.github.raw+json"})
        r.raise_for_status()
        return r.text


async def _get_json(url: str):
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
        r = await c.get(url, headers={"User-Agent": "Diana/1.0",
                                      "Accept": "application/vnd.github+json"})
        r.raise_for_status()
        return r.json()


def _pick_markdown(items: list[dict]) -> dict | None:
    files = [i for i in items if i.get("type") == "file"
             and i["name"].lower().endswith((".md", ".markdown"))]
    for pref in PREFERRED_FILES:
        for f in files:
            if f["name"].lower() == pref:
                return f
    return files[0] if files else None


async def resolve_markdown(url: str) -> tuple[str, str, str]:
    """Returns (name_hint, markdown_text, resolved_url)."""
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    m = GITHUB_BLOB.search(url)
    if m:
        o, r, branch, path = m.groups()
        raw = f"https://raw.githubusercontent.com/{o}/{r}/{branch}/{path}"
        return path.rsplit("/", 1)[-1].rsplit(".", 1)[0], await _get_text(raw), url

    if "raw.githubusercontent.com" in url or url.lower().endswith(
            (".md", ".markdown", ".txt")):
        name = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return name, await _get_text(url), url

    m = GITHUB_TREE.search(url)
    if m and "github.com" in url:
        o, r, branch, path = m.groups()
        api = f"https://api.github.com/repos/{o}/{r}/contents/{path or ''}"
        if branch:
            api += f"?ref={branch}"
        items = await _get_json(api)
        if isinstance(items, dict):  # single file
            items = [items]
        chosen = _pick_markdown(items)
        if not chosen:
            # look one level into subdirectories for a SKILL.md
            for d in [i for i in items if i.get("type") == "dir"][:6]:
                sub = await _get_json(d["url"])
                chosen = _pick_markdown(sub)
                if chosen:
                    break
        if not chosen:
            raise ValueError("No markdown file found at that GitHub location.")
        name = (path or r).rstrip("/").rsplit("/", 1)[-1]
        if chosen["name"].lower() not in PREFERRED_FILES:
            name = chosen["name"].rsplit(".", 1)[0]
        return name, await _get_text(chosen["download_url"]), chosen.get("html_url", url)

    # any other URL: try fetching it as plain text
    return url.rsplit("/", 1)[-1] or "imported-skill", await _get_text(url), url


def _parse_frontmatter(doc: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---", doc, flags=re.DOTALL)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            kv = line.split(":", 1)
            if len(kv) == 2 and kv[1].strip():
                meta[kv[0].strip().lower()] = kv[1].strip().strip("'\"")
    return meta


async def _find_skill_files(o: str, r: str, branch: str, prefix: str) -> list[str]:
    """All SKILL.md paths in a repo (optionally under a subpath)."""
    tree = await _get_json(
        f"https://api.github.com/repos/{o}/{r}/git/trees/{branch}?recursive=1")
    out = []
    for item in tree.get("tree", []):
        p = item.get("path", "")
        if item.get("type") == "blob" and p.lower().endswith("skill.md"):
            if not prefix or p == prefix or p.startswith(prefix.rstrip("/") + "/"):
                out.append(p)
    return out


async def _import_skill_md(o: str, r: str, branch: str, path: str) -> str:
    """Import one SKILL.md verbatim (they are already operational skill docs)."""
    raw_url = f"https://raw.githubusercontent.com/{o}/{r}/{branch}/{path}"
    doc = await _get_text(raw_url)
    meta = _parse_frontmatter(doc)
    parent = path.rsplit("/", 2)[-2] if "/" in path else r
    name = (meta.get("name") or parent).strip()[:48]
    fb_name, fb_desc = _fallback_meta(name, doc)
    name = name or fb_name
    desc = (meta.get("description") or fb_desc).split("\n")[0].strip()[:160]
    html_url = f"https://github.com/{o}/{r}/blob/{branch}/{path}"
    db.upsert_skill(name, desc, doc[:15000] + f"\n\n---\n*Learned from: {html_url}*")
    return name


async def import_repo_skills(url: str) -> dict | None:
    """If the URL is a GitHub repo/folder containing SKILL.md files, import them all.
    Returns None when the location has no SKILL.md (caller falls back to single-doc)."""
    m = GITHUB_TREE.search(url.strip().rstrip("/"))
    if not m or "github.com" not in url or GITHUB_BLOB.search(url):
        return None
    o, r, branch, prefix = m.groups()
    if not branch:
        branch = (await _get_json(f"https://api.github.com/repos/{o}/{r}")).get(
            "default_branch", "main")
    try:
        paths = await _find_skill_files(o, r, branch, prefix or "")
    except Exception as e:
        log.warning("repo tree scan failed for %s: %s", url, e)
        return None
    if not paths:
        return None
    imported, failed = [], 0
    for p in paths[:60]:
        try:
            imported.append(await _import_skill_md(o, r, branch, p))
        except Exception as e:
            failed += 1
            log.warning("failed to import %s: %s", p, e)
    await hub.broadcast("skills", db.all_skills())
    if len(paths) > 60:
        log.info("capped import at 60 of %d skills in %s", len(paths), url)
    return {"name": f"{len(imported)} skills from {r}" if len(imported) != 1 else imported[0],
            "description": f"imported from github.com/{o}/{r}",
            "source": url, "imported": imported, "failed": failed,
            "taught_to": None}


DISTILL_PROMPT = """You are Diana, a supervising AI. You are learning a new skill from a \
document so you can teach it to your worker agents. Distill the document below into an \
operational skill: what it's for, when to use it, the concrete method/steps, key \
techniques, and pitfalls. Keep every actionable detail; drop marketing fluff, \
installation boilerplate, and licensing text.

Document (from {url}):
---
{doc}
---

Reply with ONLY JSON:
{{"name": "short skill name (2-4 words)", "description": "one line: what this skill enables", "content": "the distilled skill document in markdown"}}"""


def _fallback_meta(name_hint: str, doc: str) -> tuple[str, str]:
    m = re.search(r"^#\s+(.+)$", doc, flags=re.MULTILINE)
    name = (m.group(1).strip() if m else name_hint).strip()[:48] or "imported-skill"
    fm = re.search(r"^description:\s*(.+)$", doc, flags=re.MULTILINE | re.IGNORECASE)
    para = re.search(r"^(?!#|---|>)([^\n]{20,200})$", doc, flags=re.MULTILINE)
    desc = (fm.group(1) if fm else (para.group(1) if para else "Imported from GitHub")).strip()[:160]
    return name, desc


async def import_from_url(url: str, teach_to: str | None = None) -> dict:
    # a repo or folder full of SKILL.md files → import the whole collection
    bulk = await import_repo_skills(url)
    if bulk:
        if teach_to and len(bulk["imported"]) == 1:
            agent = db.get_agent(teach_to)
            if agent:
                skills = list(dict.fromkeys(agent["skills"] + bulk["imported"]))
                db.set_agent_skills(agent["name"], skills)
                await hub.broadcast("agents", db.all_agents())
                bulk["taught_to"] = agent["name"]
        return bulk

    name_hint, doc, resolved = await resolve_markdown(url)
    if len(doc.strip()) < 40:
        raise ValueError("That document is too short to learn anything from.")

    name, description, content = None, None, None
    try:
        raw = await llm.chat(
            [{"role": "user", "content": DISTILL_PROMPT.format(
                url=resolved, doc=doc[:16000])}],
            model=config.diana_model(), temperature=0.4)
        obj = llm.extract_json(raw) or {}
        if obj.get("content") and len(str(obj["content"])) > 100:
            name = str(obj.get("name") or "").strip()[:48]
            description = str(obj.get("description") or "").strip()[:160]
            content = str(obj["content"])
    except Exception as e:
        log.warning("distillation failed, storing raw doc: %s", e)

    if not content:
        content = doc[:15000]
    if not name or not description:
        fb_name, fb_desc = _fallback_meta(name_hint, doc)
        name = name or fb_name
        description = description or fb_desc

    content += f"\n\n---\n*Learned from: {resolved}*"
    db.upsert_skill(name, description, content)
    await hub.broadcast("skills", db.all_skills())

    taught = None
    if teach_to:
        agent = db.get_agent(teach_to)
        if agent:
            skills = list(dict.fromkeys(agent["skills"] + [name]))
            db.set_agent_skills(agent["name"], skills)
            await hub.broadcast("agents", db.all_agents())
            taught = agent["name"]

    return {"name": name, "description": description, "source": resolved,
            "taught_to": taught}
