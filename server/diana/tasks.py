import asyncio
import logging
from datetime import datetime, timedelta, timezone

from . import config, db, supervisor
from . import agents as agents_mod
from .events import hub

log = logging.getLogger("diana.tasks")


class Engine:
    """Background executor: picks pending tasks, runs workers, has Diana evaluate."""

    def __init__(self):
        self.wake = asyncio.Event()
        self.running_ids: set[str] = set()
        self.sem = asyncio.Semaphore(config.WORKER_CONCURRENCY)
        self._stopped = False

    def kick(self):
        self.wake.set()

    async def run_forever(self):
        # recover tasks stuck mid-run from a previous boot
        for t in db.tasks_by_status("running") + db.tasks_by_status("review"):
            db.update_task(t["id"], status="pending")
        while not self._stopped:
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()
            await self._reap_stuck()
            for task in db.tasks_by_status("pending"):
                if task["id"] in self.running_ids or not task["parent_id"]:
                    continue
                mission = db.get_task(task["mission_id"])
                if not mission or mission["status"] in ("cancelled", "failed"):
                    continue
                if mission.get("mode") == "sequential" and self._blocked(task, mission):
                    continue
                self.running_ids.add(task["id"])
                asyncio.create_task(self._execute(task["id"]))

    def _blocked(self, task: dict, mission: dict) -> bool:
        """In a sequential mission, a step waits until all earlier steps finish."""
        for sib in db.children(mission["id"]):
            if sib["ordinal"] < task["ordinal"] and \
                    sib["status"] not in ("done", "cancelled", "failed"):
                return True
        return False

    async def _reap_stuck(self):
        """Fail tasks that have been 'running' for over 25 minutes (hung tool/LLM)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=25)
        for t in db.tasks_by_status("running"):
            try:
                started = datetime.fromisoformat(t["updated_at"])
            except ValueError:
                continue
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if started < cutoff:
                log.warning("reaping stuck task %s (%s)", t["id"], t["title"])
                db.update_task(t["id"], status="failed",
                               evaluation="timed out after 25 minutes")
                self.running_ids.discard(t["id"])
                await hub.broadcast("tree", db.task_tree())
                await self._check_mission(t["mission_id"])

    async def _execute(self, task_id: str):
        async with self.sem:
            try:
                await self._execute_inner(task_id)
            except Exception:
                log.exception("task %s crashed", task_id)
                db.update_task(task_id, status="failed",
                               evaluation="internal error while executing")
                await hub.broadcast("tree", db.task_tree())
            finally:
                self.running_ids.discard(task_id)
                task = db.get_task(task_id)
                if task:
                    await self._check_mission(task["mission_id"])
                self.kick()

    async def _execute_inner(self, task_id: str):
        task = db.get_task(task_id)
        if not task or task["status"] != "pending":
            return
        agent = db.get_agent(task["agent"] or "") or db.get_agent("Sage")
        if not agent:
            db.update_task(task_id, status="failed", evaluation="no agent available")
            return

        db.update_task(task_id, status="running",
                       attempts=task["attempts"] + 1, agent=agent["name"])
        await hub.broadcast("tree", db.task_tree())
        await hub.broadcast("log", {"agent": agent["name"],
                                    "text": f"started: {task['title'][:70]}"})
        await hub.set_status("working", f"{agent['name']} → {task['title']}")

        result = await agents_mod.run_task(db.get_task(task_id), agent)
        current = db.get_task(task_id)
        if not current or current["status"] != "running":
            return  # reaped by the watchdog or cancelled while we worked
        db.update_task(task_id, status="review", result=result)
        await hub.broadcast("tree", db.task_tree())

        verdict = await supervisor.evaluate_task(db.get_task(task_id))
        task = db.get_task(task_id)
        if verdict["verdict"] == "pass":
            db.update_task(task_id, status="done", evaluation=verdict["feedback"])
            outcome = "passed review"
        elif task["attempts"] < config.MAX_TASK_ATTEMPTS:
            db.update_task(task_id, status="pending",
                           feedback=verdict["feedback"],
                           evaluation=f"retry requested: {verdict['feedback']}")
            outcome = "sent back for retry"
        else:
            db.update_task(task_id, status="failed", evaluation=verdict["feedback"])
            outcome = "failed review"
        await hub.broadcast("log", {"agent": "Diana",
                                    "text": f"{task['title'][:60]} — {outcome}"})
        await hub.broadcast("tree", db.task_tree())

    async def _check_mission(self, mission_id: str | None):
        if not mission_id:
            return
        mission = db.get_task(mission_id)
        if not mission or mission["status"] not in ("active",):
            return
        kids = db.children(mission_id)
        if not kids:
            return
        terminal = {"done", "failed", "cancelled"}
        if all(k["status"] in terminal for k in kids):
            ok = all(k["status"] == "done" for k in kids if k["status"] != "cancelled")
            db.update_task(mission_id, status="done" if ok else "failed")
            await hub.broadcast("tree", db.task_tree())
            summary = await supervisor.summarize_mission(db.get_task(mission_id))
            db.update_task(mission_id, result=summary)
            db.add_message("assistant", summary, {"mission_id": mission_id})
            await hub.broadcast("message", {"role": "assistant", "content": summary,
                                            "speak": True})
            busy = db.tasks_by_status("pending") or db.tasks_by_status("running")
            await hub.set_status("working" if busy else "idle")


engine = Engine()
