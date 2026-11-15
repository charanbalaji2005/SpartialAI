"""A small, explicit training loop shared by the depth and segmentation trainers.

Features: CUDA mixed precision (FP16 autocast + GradScaler, or BF16), gradient
accumulation, gradient clipping, warmup + cosine LR schedule, per-epoch validation,
early stopping, best/last/recovery checkpoints, exact resume (model, optimizer,
scheduler, scaler, epoch, step, RNG), TensorBoard + JSONL logging, and automatic
recovery from a CUDA out-of-memory step (skips the batch and clears the cache).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.checkpoints import CheckpointManager
from ..common.logger import RunLogger


@dataclass
class TrainSettings:
    epochs: int = 10
    grad_accum: int = 4
    amp: str = "fp16"                 # fp16 | bf16 | off
    clip_grad: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.05
    early_stop_patience: int = 4
    recovery_every: int = 500         # optimizer steps between recovery checkpoints
    log_every: int = 20
    max_steps_per_epoch: int | None = None
    val_every: int = 1

    @classmethod
    def from_cfg(cls, cfg: dict) -> "TrainSettings":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(total - warmup, 1))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader, loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
                 validate: Callable[[torch.nn.Module], dict[str, float]], settings: TrainSettings,
                 ckpt: CheckpointManager, logger: RunLogger, device: torch.device,
                 export_fn: Callable[[torch.nn.Module, Path], None] | None = None,
                 on_epoch_start: Callable[[int, torch.nn.Module, torch.optim.Optimizer], None] | None = None) -> None:
        self.model, self.opt, self.loader = model, optimizer, train_loader
        self.loss_fn, self.validate, self.s = loss_fn, validate, settings
        self.ckpt, self.log, self.device = ckpt, logger, device
        self.export_fn, self.on_epoch_start = export_fn, on_epoch_start
        steps_per_epoch = math.ceil(min(len(train_loader), settings.max_steps_per_epoch or 1e12) / settings.grad_accum)
        self.total_steps = max(1, steps_per_epoch * settings.epochs)
        self.sched = warmup_cosine(optimizer, settings.warmup_steps, self.total_steps, settings.min_lr_ratio)
        use_cuda = device.type == "cuda"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(settings.amp) if use_cuda else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and settings.amp == "fp16")
        self.epoch, self.step, self.bad_epochs = 0, 0, 0
        self.oom_skips = 0

    # ------------------------------------------------------------------ state
    def state(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "epoch": self.epoch, "step": self.step, "bad_epochs": self.bad_epochs,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}

    def load_state(self, st: dict[str, Any]) -> None:
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        # A recovery checkpoint is mid-epoch: restart that epoch; a last.pt is end-of-epoch.
        self.epoch = st["epoch"] if st.get("recovery") else st["epoch"] + 1
        self.step, self.bad_epochs = st["step"], st.get("bad_epochs", 0)
        rng = st.get("rng") or {}
        if rng.get("python"):
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        self.log.text(f"Resumed at epoch {self.epoch}, step {self.step}")

    # ------------------------------------------------------------------ loop
    def fit(self) -> dict[str, Any]:
        best_metrics: dict[str, float] = {}
        self.ckpt.write_run(status="running")
        try:
            while self.epoch < self.s.epochs:
                if self.on_epoch_start:
                    self.on_epoch_start(self.epoch, self.model, self.opt)
                t0 = time.time()
                train_loss = self._train_epoch()
                metrics = {"train/loss": train_loss, "lr": self.opt.param_groups[0]["lr"],
                           "epoch_seconds": round(time.time() - t0, 1)}
                if (self.epoch + 1) % self.s.val_every == 0 or self.epoch + 1 == self.s.epochs:
                    self.model.eval()
                    with torch.no_grad():
                        metrics.update({f"val/{k}": v for k, v in self.validate(self.model).items()
                                        if isinstance(v, (int, float))})
                    self.model.train()
                if torch.cuda.is_available():
                    metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                self.log.log(self.step, metrics)
                export = (lambda d: self.export_fn(self.model, d)) if self.export_fn else None
                is_best = self.ckpt.save(self.state(), metrics, self.epoch, export)
                summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("val/"))
                self.log.text(f"epoch {self.epoch + 1}/{self.s.epochs} loss={train_loss:.4f} {summary}"
                              f"{'  (best)' if is_best else ''}")
                if is_best:
                    best_metrics = metrics
                    self.bad_epochs = 0
                elif self.ckpt.monitor.metric in metrics:
                    self.bad_epochs += 1
                    if self.bad_epochs >= self.s.early_stop_patience:
                        self.log.text(f"Early stopping: no improvement in {self.bad_epochs} epochs")
                        break
                self.epoch += 1
            self.ckpt.write_run(status="completed", oom_skips=self.oom_skips)
        except KeyboardInterrupt:
            self.ckpt.save_recovery(self.state())
            self.ckpt.write_run(status="interrupted")
            self.log.text("Interrupted: recovery checkpoint saved. Resume with --resume last")
            raise
        except Exception:
            self.ckpt.write_run(status="failed")
            raise
        return {"best": best_metrics, "best_value": self.ckpt.best_value, "run_dir": str(self.ckpt.run_dir)}

    def _train_epoch(self) -> float:
        self.model.train()
        total, count, pending = 0.0, 0, 0
        self.opt.zero_grad(set_to_none=True)
        limit = self.s.max_steps_per_epoch * self.s.grad_accum if self.s.max_steps_per_epoch else None
        for i, batch in enumerate(self.loader):
            if limit is not None and i >= limit:
                break
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            try:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32,
                                    enabled=self.amp_dtype is not None):
                    loss = self.loss_fn(self.model, batch)
                if not torch.isfinite(loss):
                    self.log.text(f"Non-finite loss at step {self.step}; batch skipped")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.scaler.scale(loss / self.s.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                self.oom_skips += 1
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                self.log.text(f"CUDA out of memory on a batch (skipped, {self.oom_skips} so far). "
                              "Lower batch_size or input size if this repeats.")
                continue
            total += float(loss.detach())
            count += 1
            pending += 1
            if pending == self.s.grad_accum:
                self._optimizer_step()
                pending = 0
        if pending:
            self._optimizer_step()
        return total / max(count, 1)

    def _optimizer_step(self) -> None:
        if self.s.clip_grad:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], self.s.clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        self.step += 1
        if self.step % self.s.log_every == 0:
            self.log.log(self.step, {"lr": self.opt.param_groups[0]["lr"]}, "train")
        if self.s.recovery_every and self.step % self.s.recovery_every == 0:
            self.ckpt.save_recovery(self.state())
