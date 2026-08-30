"""Scheduled missions: Diana runs standing instructions on a clock.

Spec formats (container-local time, set TZ in docker-compose):
  "every 30 minutes" | "every 2 hours"
  "daily 09:00"
  "weekly mon 09:00"  (mon..sun)
  "once 2026-09-01T15:00"
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta

from . import db
from .events import hub

log = logging.getLogger("diana.scheduler")

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def compute_next(spec: str, after: datetime | None = None) -> datetime | None:
    """Next fire time strictly after `after` (default: now). None = invalid/expired."""
    now = after or datetime.now()
    s = spec.strip().lower()

    m = re.fullmatch(r"every\s+(\d+)\s*(minute|minutes|min|hour|hours|hr|hrs)", s)
    if m:
        n = max(1, int(m.group(1)))
        delta = timedelta(minutes=n) if m.group(2).startswith(("min",)) else timedelta(hours=n)
        return now + delta

    m = re.fullmatch(r"daily\s+(\d{1,2}):(\d{2})", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        candidate = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        return candidate if candidate > now else candidate + timedelta(days=1)

    m = re.fullmatch(r"weekly\s+(mon|tue|wed|thu|fri|sat|sun)\s+(\d{1,2}):(\d{2})", s)
    if m:
        target_dow = DAYS.index(m.group(1))
        h, mi = int(m.group(2)), int(m.group(3))
        candidate = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        days_ahead = (target_dow - candidate.weekday()) % 7
        candidate += timedelta(days=days_ahead)
        if candidate <= now:
            candidate += timedelta(days=7)
        return candidate

    m = re.fullmatch(r"once\s+(\d{4}-\d{2}-\d{2}[t ]\d{1,2}:\d{2})(?::\d{2})?", s)
    if m:
        try:
            dt = datetime.fromisoformat(m.group(1).replace(" ", "T"))
            return dt if dt > now else None
        except ValueError:
            return None
    return None


def create(spec: str, instruction: str) -> dict | None:
    nxt = compute_next(spec)
    if not nxt:
        return None
    row = db.add_schedule(spec, instruction, nxt.isoformat(timespec="minutes"))
    hub.emit("schedules", db.all_schedules())
    return row


async def run_forever():
    from . import supervisor
    from .tasks import engine
    while True:
        try:
            now = datetime.now()
            for s in db.all_schedules():
                if not s["enabled"] or not s["next_run"]:
                    continue
                try:
                    due = datetime.fromisoformat(s["next_run"]) <= now
                except ValueError:
                    due = False
                if not due:
                    continue
                nxt = compute_next(s["spec"])
                is_once = s["spec"].strip().lower().startswith("once")
                db.update_schedule(
                    s["id"], last_run=now.isoformat(timespec="minutes"),
                    next_run=None if is_once or not nxt else nxt.isoformat(timespec="minutes"),
                    enabled=0 if is_once or not nxt else 1)
                await hub.broadcast("schedules", db.all_schedules())
                log.info("firing schedule %s: %s", s["id"], s["instruction"][:80])
                try:
                    await supervisor.handle_user_message(
                        f"[Automatic scheduled run #{s['id']} — do the following now. "
                        "It already repeats on its own schedule: do NOT create another "
                        f"schedule for it.] {s['instruction']}",
                        source="schedule")
                    engine.kick()
                except Exception:
                    log.exception("scheduled run failed")
        except Exception:
            log.exception("scheduler tick failed")
        await asyncio.sleep(20)
