import asyncio
import logging

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
            for task in db.tasks_by_status("pending"):
                if task["id"] in self.running_ids or not task["parent_id"]:
                    continue
                mission = db.get_task(task["mission_id"])
                if not mission or mission["status"] in ("cancelled", "failed"):
                    continue
                self.running_ids.add(task["id"])
                asyncio.create_task(self._execute(task["id"]))

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
        await hub.set_status("working", f"{agent['name']} → {task['title']}")

        result = await agents_mod.run_task(db.get_task(task_id), agent)
        db.update_task(task_id, status="review", result=result)
        await hub.broadcast("tree", db.task_tree())

        verdict = await supervisor.evaluate_task(db.get_task(task_id))
        task = db.get_task(task_id)
        if verdict["verdict"] == "pass":
            db.update_task(task_id, status="done", evaluation=verdict["feedback"])
        elif task["attempts"] < config.MAX_TASK_ATTEMPTS:
            db.update_task(task_id, status="pending",
                           feedback=verdict["feedback"],
                           evaluation=f"retry requested: {verdict['feedback']}")
        else:
            db.update_task(task_id, status="failed", evaluation=verdict["feedback"])
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
