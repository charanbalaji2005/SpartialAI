"""GPU detection, VRAM monitoring and memory release.

torch is imported lazily so the API, tests and geometry code work without it.
"""
from __future__ import annotations

import gc
import logging
import shutil
import subprocess
from typing import Any

from .config import settings

log = logging.getLogger("spatialai.gpu")


def _torch():
    try:
        import torch  # noqa: WPS433
        return torch
    except Exception:  # pragma: no cover - depends on install
        return None


def select_device() -> str:
    torch = _torch()
    pref = settings.device_preference
    if pref == "cpu" or torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        log.warning("CUDA requested but not available; falling back to CPU")
    return "cpu"


def use_fp16(device: str) -> bool:
    return device == "cuda" and settings.use_fp16


def gpu_info() -> dict[str, Any]:
    torch = _torch()
    info: dict[str, Any] = {
        "torch_installed": torch is not None,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": False,
        "device": "cpu",
        "name": None,
        "total_vram_gb": None,
        "allocated_gb": None,
        "reserved_gb": None,
        "free_gb": None,
        "utilization_pct": None,
        "driver": None,
    }
    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            cuda_available=True,
            device="cuda",
            name=props.name,
            cuda_version=torch.version.cuda,
            total_vram_gb=round(props.total_memory / 1024**3, 2),
            allocated_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            reserved_gb=round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        )
        try:
            free, _total = torch.cuda.mem_get_info(0)
            info["free_gb"] = round(free / 1024**3, 2)
        except Exception:
            pass
    smi = _nvidia_smi()
    if smi:
        info.update({k: v for k, v in smi.items() if v is not None})
    return info


def _nvidia_smi() -> dict[str, Any] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()
        if not out:
            return None
        name, driver, util, used, total = [p.strip() for p in out[0].split(",")]
        return {
            "name": name,
            "driver": driver,
            "utilization_pct": float(util),
            "smi_used_gb": round(float(used) / 1024, 2),
            "smi_total_gb": round(float(total) / 1024, 2),
        }
    except Exception:
        return None


def release_memory() -> None:
    """Free cached CUDA blocks. Only frees memory no longer referenced by live tensors."""
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float | None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    return None


def is_oom(exc: BaseException) -> bool:
    torch = _torch()
    if torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()
