"""Metrics logging to JSONL (always) and TensorBoard (when installed)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: Path, use_tensorboard: bool = True) -> None:
        self.run_dir = run_dir
        self.jsonl = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            except Exception:
                self.tb = None

    def log(self, step: int, values: dict[str, Any], prefix: str = "") -> None:
        record: dict[str, Any] = {"step": step, "time": round(time.time(), 2)}
        for key, value in values.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                record[name] = value
                if self.tb is not None:
                    self.tb.add_scalar(name, value, step)
        self.jsonl.write(json.dumps(record) + "\n")
        self.jsonl.flush()

    def text(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.run_dir / "train.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def close(self) -> None:
        self.jsonl.close()
        if self.tb is not None:
            self.tb.close()
