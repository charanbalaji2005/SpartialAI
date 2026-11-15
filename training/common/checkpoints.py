"""Checkpoints, run registry and promotion to production.

Layout
    checkpoints/
      registry.json                         every run + current production per task
      runs/<task>/<run_id>/
        run.json  config.yaml  metrics.jsonl  train.log
        last.pt   best.pt   recovery.pt       full training state (torch)
        export/                             deployable model (HF save_pretrained / YOLO .pt)
      production/<task>/                    what the SpatialAI backend loads
        manifest.json  <model files>
      production_previous/<task>/           automatic rollback copy

A run never overwrites production by itself. `promote()` validates the candidate
(loads it and runs a forward pass via a task-specific smoke test), compares its
monitored metric with production, and only then swaps it in atomically.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import CHECKPOINT_DIR

_LOCK = threading.Lock()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- registry
class Registry:
    def __init__(self, root: Path = CHECKPOINT_DIR) -> None:
        self.root = root
        self.path = root / "registry.json"

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"version": 1, "tasks": {}}

    def update(self, fn: Callable[[dict], None]) -> dict:
        with _LOCK:
            data = self.load()
            fn(data)
            _atomic_write_json(self.path, data)
            return data

    def task(self, task: str) -> dict[str, Any]:
        return self.load()["tasks"].get(task, {"production": None, "runs": {}})

    def record_run(self, task: str, run_id: str, info: dict[str, Any]) -> None:
        def fn(data: dict) -> None:
            t = data["tasks"].setdefault(task, {"production": None, "runs": {}})
            t["runs"].setdefault(run_id, {}).update(info)
        self.update(fn)


# --------------------------------------------------------------------------- runs
@dataclass
class MonitorSpec:
    metric: str                  # e.g. "val/abs_rel"
    mode: str = "min"            # "min" or "max"
    min_delta: float = 0.0

    def better(self, new: float, old: float | None) -> bool:
        if old is None:
            return True
        return new < old - self.min_delta if self.mode == "min" else new > old + self.min_delta


class CheckpointManager:
    """Saves last/best/recovery checkpoints for one training run."""

    def __init__(self, task: str, monitor: MonitorSpec, run_id: str | None = None,
                 root: Path = CHECKPOINT_DIR, resume: str | None = None) -> None:
        self.task = task
        self.monitor = monitor
        self.root = root
        self.registry = Registry(root)
        task_dir = root / "runs" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self.run_dir = self._find_resume(task_dir, resume)
            self.run_id = self.run_dir.name
        else:
            self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
            self.run_dir = task_dir / self.run_id
            suffix = 1
            while self.run_dir.exists():
                self.run_dir = task_dir / f"{self.run_id}-{suffix}"
                suffix += 1
            self.run_id = self.run_dir.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.run_dir / "export"
        self.best_value: float | None = self._read_run().get("best_value")

    @staticmethod
    def _find_resume(task_dir: Path, resume: str) -> Path:
        if resume == "last":
            runs = sorted((p for p in task_dir.iterdir() if (p / "last.pt").exists()),
                          key=lambda p: (p / "last.pt").stat().st_mtime)
            if not runs:
                raise FileNotFoundError(f"No run with last.pt under {task_dir}")
            return runs[-1]
        path = Path(resume)
        run = path if path.is_dir() else task_dir / resume
        if not (run / "last.pt").exists():
            raise FileNotFoundError(f"Cannot resume: {run / 'last.pt'} not found")
        return run

    def _read_run(self) -> dict:
        p = self.run_dir / "run.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def write_run(self, **info: Any) -> None:
        data = self._read_run()
        data.update(info, task=self.task, run_id=self.run_id, updated_at=time.time())
        _atomic_write_json(self.run_dir / "run.json", data)
        self.registry.record_run(self.task, self.run_id, {
            k: v for k, v in data.items() if k in (
                "status", "best_value", "best_epoch", "monitor", "metrics", "dataset_version",
                "model_name", "updated_at", "export_dir")})

    # ------------------------------------------------------------------ saving
    def save(self, state: dict[str, Any], metrics: dict[str, float], epoch: int,
             export_fn: Callable[[Path], None] | None = None) -> bool:
        """Save last.pt; if the monitored metric improved, also best.pt + export. Returns is_best."""
        state = {**state, "epoch": epoch, "metrics": metrics, "best_value": self.best_value,
                 "saved_at": time.time()}
        _atomic_torch_save(state, self.run_dir / "last.pt")
        value = metrics.get(self.monitor.metric)
        is_best = value is not None and self.monitor.better(value, self.best_value)
        if is_best:
            self.best_value = float(value)
            state["best_value"] = self.best_value
            _atomic_torch_save(state, self.run_dir / "best.pt")
            if export_fn is not None:
                tmp = self.run_dir / "export.tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                export_fn(tmp)
                shutil.rmtree(self.export_dir, ignore_errors=True)
                os.replace(tmp, self.export_dir)
            self.write_run(best_value=self.best_value, best_epoch=epoch, metrics=metrics,
                           monitor={"metric": self.monitor.metric, "mode": self.monitor.mode},
                           export_dir=str(self.export_dir.relative_to(self.root)))
        return is_best

    def save_recovery(self, state: dict[str, Any]) -> None:
        """Mid-epoch safety checkpoint (overwritten each time)."""
        _atomic_torch_save({**state, "recovery": True, "saved_at": time.time()},
                           self.run_dir / "recovery.pt")

    def load_for_resume(self, prefer_recovery: bool = True) -> dict[str, Any] | None:
        import torch
        candidates = [self.run_dir / "recovery.pt", self.run_dir / "last.pt"] if prefer_recovery \
            else [self.run_dir / "last.pt"]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            return None
        newest = max(existing, key=lambda p: p.stat().st_mtime)
        return torch.load(newest, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- promotion
def production_dir(task: str, root: Path = CHECKPOINT_DIR) -> Path:
    return root / "production" / task


def promote(task: str, run_id: str, smoke_test: Callable[[Path], None], force: bool = False,
            root: Path = CHECKPOINT_DIR) -> dict[str, Any]:
    """Validate a run's export and make it the production model if it is better."""
    registry = Registry(root)
    run_dir = root / "runs" / task / run_id
    run = json.loads((run_dir / "run.json").read_text())
    export = run_dir / "export"
    if not export.exists() or not any(export.iterdir()):
        raise RuntimeError(f"Run {run_id} has no exported model (it never reached a best checkpoint).")

    smoke_test(export)       # raises if the model can't load or run

    monitor = run["monitor"]
    spec = MonitorSpec(monitor["metric"], monitor["mode"])
    current = registry.task(task).get("production")
    current_value = current.get("value") if current and current.get("metric") == spec.metric else None
    candidate_value = run["best_value"]
    if not force and current is not None and not spec.better(candidate_value, current_value):
        return {"promoted": False, "reason": f"{spec.metric} {candidate_value:.4f} does not beat production "
                f"{current_value:.4f}" if current_value is not None else "production uses a different metric; use --force"}

    prod = production_dir(task, root)
    prev = root / "production_previous" / task
    staging = prod.with_name(task + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(export, staging)
    manifest = {"task": task, "run_id": run_id, "metric": spec.metric, "value": candidate_value,
                "mode": spec.mode, "metrics": run.get("metrics", {}), "model_name": run.get("model_name"),
                "dataset_version": run.get("dataset_version"), "promoted_at": time.time()}
    _atomic_write_json(staging / "manifest.json", manifest)
    if prod.exists():
        shutil.rmtree(prev, ignore_errors=True)
        prev.parent.mkdir(parents=True, exist_ok=True)
        os.replace(prod, prev)
    prod.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, prod)

    def fn(data: dict) -> None:
        t = data["tasks"].setdefault(task, {"production": None, "runs": {}})
        t["production"] = manifest
    registry.update(fn)
    return {"promoted": True, **manifest}


def rollback(task: str, root: Path = CHECKPOINT_DIR) -> dict[str, Any]:
    prod, prev = production_dir(task, root), root / "production_previous" / task
    if not prev.exists():
        raise RuntimeError(f"No previous production model for {task}")
    tmp = prod.with_name(task + ".rollback")
    shutil.rmtree(tmp, ignore_errors=True)
    if prod.exists():
        os.replace(prod, tmp)
    os.replace(prev, prod)
    shutil.rmtree(tmp, ignore_errors=True)
    manifest = json.loads((prod / "manifest.json").read_text())
    Registry(root).update(lambda d: d["tasks"].setdefault(task, {"runs": {}}).__setitem__("production", manifest))
    return manifest
