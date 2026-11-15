"""Single-worker job queue: one reconstruction at a time keeps VRAM predictable."""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from ..core import gpu
from ..database.db import Database
from ..models.registry import manager
from . import pipeline

log = logging.getLogger("spatialai.jobs")

STAGE_KEYS = [k for k, _, _ in pipeline.STAGES]
STAGE_NAMES = {k: n for k, n, _ in pipeline.STAGES}
STAGE_WEIGHTS = {k: w for k, _, w in pipeline.STAGES}


@dataclass
class JobState:
    project_id: str
    quality: str
    status: str = "queued"
    stage: str = "upload"
    stage_fraction: float = 0.0
    progress: float = 0.0
    message: str = "Waiting for the GPU"
    error: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None
    version: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        stages = []
        cur = STAGE_KEYS.index(self.stage) if self.stage in STAGE_KEYS else 0
        for i, key in enumerate(STAGE_KEYS):
            if self.status == "completed" or i < cur:
                state = "done"
            elif i == cur:
                state = {"failed": "failed", "cancelled": "cancelled"}.get(self.status, "active")
            else:
                state = "pending"
            stages.append({"key": key, "name": STAGE_NAMES[key], "state": state})
        return {
            "project_id": self.project_id, "status": self.status, "stage": self.stage,
            "stage_name": STAGE_NAMES.get(self.stage), "progress": round(self.progress, 4),
            "message": self.message, "error": self.error, "stages": stages,
            "logs": self.logs[-200:], "started_at": self.started_at,
            "finished_at": self.finished_at, "version": self.version,
        }


class JobManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.jobs: dict[str, JobState] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="spatialai-worker", daemon=True)
        self._worker.start()

    def submit(self, project_id: str, quality: str) -> JobState:
        with self._lock:
            existing = self.jobs.get(project_id)
            if existing and existing.status in ("queued", "running"):
                return existing
            job = JobState(project_id, quality)
            job.logs.append(_line("info", f"Queued in {quality} mode"))
            self.jobs[project_id] = job
        self.db.update(project_id, status="queued", stage="upload", progress=0.0,
                       message=job.message, error=None, quality=quality)
        self._queue.put(project_id)
        return job

    def cancel(self, project_id: str) -> bool:
        job = self.jobs.get(project_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job.cancel.set()
        job.message = "Cancelling…"
        job.version += 1
        return True

    def get(self, project_id: str) -> JobState | None:
        return self.jobs.get(project_id)

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            pid = self._queue.get()
            job = self.jobs.get(pid)
            if job is None:
                continue
            if job.cancel.is_set():
                self._finish(job, "cancelled", "Cancelled before it started")
                continue
            self._run(job)

    def _run(self, job: JobState) -> None:
        job.status, job.started_at = "running", time.time()
        self.db.update(job.project_id, status="running")

        def emit(stage: str, frac: float, msg: str) -> None:
            job.stage, job.stage_fraction, job.message = stage, frac, msg
            done = sum(STAGE_WEIGHTS[k] for k in STAGE_KEYS[: STAGE_KEYS.index(stage)])
            job.progress = min(0.999, done + STAGE_WEIGHTS[stage] * frac)
            job.version += 1
            if frac in (0.0, 1.0):
                self.db.update(job.project_id, stage=stage, progress=job.progress, message=msg)

        def log_line(level: str, msg: str) -> None:
            job.logs.append(_line(level, msg))
            job.version += 1

        ctx = pipeline.Context(job.project_id, job.quality, emit, log_line, job.cancel.is_set)
        emit("upload", 1.0, "Image ready")
        try:
            result = pipeline.run(ctx)
            job.progress = 1.0
            self.db.update(job.project_id, scene_category=result["category"], report=result["report"])
            self._finish(job, "completed", "Scene ready")
        except pipeline.Cancelled:
            self._finish(job, "cancelled", "Cancelled")
        except Exception as exc:  # noqa: BLE001
            log.error("Reconstruction failed: %s", traceback.format_exc())
            msg = str(exc)
            if gpu.is_oom(exc):
                msg = ("The GPU ran out of memory. Switch to Fast mode, use a smaller photo, "
                       "or close other apps using the GPU.")
            job.error = msg
            log_line("error", msg)
            self._finish(job, "failed", "Reconstruction failed")
        finally:
            manager.unload_all()      # recover a clean GPU state for the next job

    def _finish(self, job: JobState, status: str, msg: str) -> None:
        job.status, job.message, job.finished_at = status, msg, time.time()
        job.version += 1
        job.logs.append(_line("stage" if status == "completed" else "warn", msg))
        self.db.update(job.project_id, status=status, message=msg, error=job.error,
                       progress=job.progress, stage=job.stage)


def _line(level: str, msg: str) -> dict[str, Any]:
    return {"t": round(time.time(), 2), "level": level, "message": msg}
