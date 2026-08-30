import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from . import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.ensure_dirs()
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  meta TEXT DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  mission_id TEXT,
  title TEXT NOT NULL,
  description TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending',
  agent TEXT,
  result TEXT,
  evaluation TEXT,
  feedback TEXT,
  attempts INTEGER DEFAULT 0,
  ordinal INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
  name TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  skills TEXT DEFAULT '[]',
  tools TEXT DEFAULT '[]',
  builtin INTEGER DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skills (
  name TEXT PRIMARY KEY,
  description TEXT DEFAULT '',
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schedules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  spec TEXT NOT NULL,
  instruction TEXT NOT NULL,
  next_run TEXT,
  last_run TEXT,
  enabled INTEGER DEFAULT 1,
  created_at TEXT NOT NULL
);
"""

MIGRATIONS = [
    "ALTER TABLE agents ADD COLUMN model TEXT",
    "ALTER TABLE tasks ADD COLUMN mode TEXT",
]


def init():
    with _lock:
        conn().executescript(SCHEMA)
        for stmt in MIGRATIONS:
            try:
                conn().execute(stmt)
            except sqlite3.OperationalError:
                pass  # already applied
        conn().commit()


def _rows(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ---------- messages ----------

def add_message(role: str, content: str, meta: dict | None = None) -> dict:
    with _lock:
        cur = conn().execute(
            "INSERT INTO messages(role, content, meta, created_at) VALUES (?,?,?,?)",
            (role, content, json.dumps(meta or {}), now()),
        )
        conn().commit()
        row = conn().execute("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def recent_messages(limit: int = 20) -> list[dict]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return list(reversed(_rows(rows)))


def clear_messages():
    with _lock:
        conn().execute("DELETE FROM messages")
        conn().commit()


def delete_messages_from(mid: int):
    """Rewind: drop this message and everything after it."""
    with _lock:
        conn().execute("DELETE FROM messages WHERE id>=?", (mid,))
        conn().commit()


def prune_messages(keep: int = 500):
    with _lock:
        conn().execute(
            "DELETE FROM messages WHERE id NOT IN"
            " (SELECT id FROM messages ORDER BY id DESC LIMIT ?)", (keep,))
        conn().commit()


# ---------- settings ----------

def get_setting(key: str, default: str | None = None) -> str | None:
    with _lock:
        row = conn().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    with _lock:
        conn().execute("INSERT INTO settings(key, value) VALUES (?,?)"
                       " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (key, value))
        conn().commit()


# ---------- schedules ----------

def add_schedule(spec: str, instruction: str, next_run: str) -> dict:
    with _lock:
        cur = conn().execute(
            "INSERT INTO schedules(spec, instruction, next_run, created_at) VALUES (?,?,?,?)",
            (spec, instruction, next_run, now()))
        conn().commit()
        row = conn().execute("SELECT * FROM schedules WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_schedule(sid: int, **fields):
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with _lock:
        conn().execute(f"UPDATE schedules SET {sets} WHERE id=?", (*fields.values(), sid))
        conn().commit()


def delete_schedule(sid: int) -> bool:
    with _lock:
        cur = conn().execute("DELETE FROM schedules WHERE id=?", (sid,))
        conn().commit()
    return cur.rowcount > 0


def all_schedules() -> list[dict]:
    with _lock:
        rows = conn().execute("SELECT * FROM schedules ORDER BY id ASC").fetchall()
    return _rows(rows)


# ---------- tasks ----------

def create_task(title: str, description: str = "", parent_id: str | None = None,
                mission_id: str | None = None, agent: str | None = None,
                status: str = "pending", ordinal: int = 0) -> dict:
    tid = new_id()
    t = now()
    with _lock:
        conn().execute(
            "INSERT INTO tasks(id, parent_id, mission_id, title, description, status, agent, ordinal, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tid, parent_id, mission_id or (None if parent_id else tid), title,
             description, status, agent, ordinal, t, t),
        )
        conn().commit()
    return get_task(tid)


def get_task(tid: str) -> dict | None:
    with _lock:
        row = conn().execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    return dict(row) if row else None


def update_task(tid: str, **fields) -> dict | None:
    if not fields:
        return get_task(tid)
    fields["updated_at"] = now()
    sets = ", ".join(f"{k}=?" for k in fields)
    with _lock:
        conn().execute(f"UPDATE tasks SET {sets} WHERE id=?", (*fields.values(), tid))
        conn().commit()
    return get_task(tid)


def all_tasks() -> list[dict]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM tasks ORDER BY created_at ASC, ordinal ASC"
        ).fetchall()
    return _rows(rows)


def tasks_by_status(status: str) -> list[dict]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM tasks WHERE status=? ORDER BY created_at ASC, ordinal ASC",
            (status,),
        ).fetchall()
    return _rows(rows)


def children(tid: str) -> list[dict]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM tasks WHERE parent_id=? ORDER BY ordinal ASC, created_at ASC",
            (tid,),
        ).fetchall()
    return _rows(rows)


def task_tree() -> list[dict]:
    """All missions (root tasks) with nested children."""
    tasks = all_tasks()
    by_parent: dict = {}
    for t in tasks:
        by_parent.setdefault(t["parent_id"], []).append(t)

    def build(node):
        node = dict(node)
        kids = by_parent.get(node["id"], [])
        node["children"] = [build(k) for k in kids]
        return node

    roots = [r for r in by_parent.get(None, []) if r["status"] != "archived"]
    return [build(r) for r in reversed(roots)]


def archive_finished_missions() -> int:
    with _lock:
        cur = conn().execute(
            "UPDATE tasks SET status='archived', updated_at=? WHERE parent_id IS NULL"
            " AND status IN ('done','failed','cancelled')", (now(),))
        conn().commit()
    return cur.rowcount


# ---------- agents ----------

def upsert_agent(name: str, role: str, system_prompt: str,
                 skills: list | None = None, tools: list | None = None,
                 builtin: bool = False, model: str | None = None):
    with _lock:
        conn().execute(
            "INSERT INTO agents(name, role, system_prompt, skills, tools, builtin, model, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET role=excluded.role,"
            " system_prompt=excluded.system_prompt, tools=excluded.tools,"
            " model=excluded.model",
            (name, role, system_prompt, json.dumps(skills or []),
             json.dumps(tools or []), int(builtin), model, now()),
        )
        conn().commit()


def get_agent(name: str) -> dict | None:
    with _lock:
        row = conn().execute("SELECT * FROM agents WHERE name=?", (name,)).fetchone()
    if not row:
        return None
    a = dict(row)
    a["skills"] = json.loads(a["skills"] or "[]")
    a["tools"] = json.loads(a["tools"] or "[]")
    return a


def all_agents() -> list[dict]:
    with _lock:
        rows = conn().execute("SELECT * FROM agents ORDER BY builtin DESC, name ASC").fetchall()
    out = []
    for r in _rows(rows):
        r["skills"] = json.loads(r["skills"] or "[]")
        r["tools"] = json.loads(r["tools"] or "[]")
        out.append(r)
    return out


def set_agent_skills(name: str, skills: list[str]):
    with _lock:
        conn().execute("UPDATE agents SET skills=? WHERE name=?", (json.dumps(skills), name))
        conn().commit()


def delete_agent(name: str) -> bool:
    with _lock:
        cur = conn().execute("DELETE FROM agents WHERE name=? AND builtin=0", (name,))
        conn().commit()
    return cur.rowcount > 0


def delete_skill(name: str) -> bool:
    with _lock:
        cur = conn().execute("DELETE FROM skills WHERE name=?", (name,))
        conn().commit()
    if cur.rowcount:
        for a in all_agents():
            if name in a["skills"]:
                set_agent_skills(a["name"], [s for s in a["skills"] if s != name])
        try:
            safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "-")
            (config.SKILLS_DIR / f"{safe}.md").unlink(missing_ok=True)
        except Exception:
            pass
    return cur.rowcount > 0


# ---------- skills ----------

def upsert_skill(name: str, description: str, content: str):
    t = now()
    with _lock:
        conn().execute(
            "INSERT INTO skills(name, description, content, created_at, updated_at)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET description=excluded.description,"
            " content=excluded.content, updated_at=excluded.updated_at",
            (name, description, content, t, t),
        )
        conn().commit()
    # mirror to disk so skills are portable / inspectable
    try:
        config.ensure_dirs()
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "-")
        (config.SKILLS_DIR / f"{safe}.md").write_text(
            f"# {name}\n\n> {description}\n\n{content}\n", encoding="utf-8")
    except Exception:
        pass


# ---------- memories ----------

def add_memory(content: str) -> dict:
    with _lock:
        cur = conn().execute("INSERT INTO memories(content, created_at) VALUES (?,?)",
                             (content, now()))
        conn().commit()
        row = conn().execute("SELECT * FROM memories WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def forget_memory(mem_id: int) -> bool:
    with _lock:
        cur = conn().execute("DELETE FROM memories WHERE id=?", (mem_id,))
        conn().commit()
    return cur.rowcount > 0


def all_memories(limit: int = 40) -> list[dict]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return list(reversed(_rows(rows)))


def get_skill(name: str) -> dict | None:
    with _lock:
        row = conn().execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
    return dict(row) if row else None


def all_skills() -> list[dict]:
    with _lock:
        rows = conn().execute("SELECT * FROM skills ORDER BY name ASC").fetchall()
    return _rows(rows)
