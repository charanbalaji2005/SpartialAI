"""Access to the SpatialAI backend so evaluation measures exactly what the app runs."""
from __future__ import annotations

import sys
from pathlib import Path

from ..common.config import ROOT

if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))


def resolve_model(task: str, choice: str) -> str:
    """'production' -> checkpoints/production/<task> if present; 'pretrained' -> default HF id;
    anything else is used as a path or Hugging Face id."""
    from app.core.config import settings
    defaults = {"depth": settings.depth_model_id, "segmentation": settings.semantic_model_id}
    if choice == "production":
        prod = ROOT / "checkpoints" / "production" / task
        if (prod / "manifest.json").exists():
            return str(prod)
        print(f"No production {task} model yet; evaluating the pretrained one.")
        return defaults[task]
    if choice == "pretrained":
        return defaults[task]
    return choice
