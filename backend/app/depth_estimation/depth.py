"""Monocular depth with Depth Anything V2 (Transformers).

Depth Anything V2 (relative) predicts *disparity-like* values: larger = closer.
`to_metric_like_depth` turns that into positive distances along the camera axis
using an assumed near/far range. The result is relative, not metric.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..core import gpu
from ..core.config import model_source, settings


def load(device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    path, info = model_source("depth", settings.depth_model_id)
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForDepthEstimation.from_pretrained(path)
    model.to(device).eval()
    return {"processor": processor, "model": model, "device": device, "torch": torch, "source": info}


def predict_disparity(bundle: dict[str, Any], image: Image.Image) -> np.ndarray:
    """Return float32 HxW disparity map at the image's resolution."""
    torch = bundle["torch"]
    device = bundle["device"]
    inputs = bundle["processor"](images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if gpu.use_fp16(device):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = bundle["model"](**inputs)
        else:
            out = bundle["model"](**inputs)
        pred = out.predicted_depth.unsqueeze(1).float()
        pred = torch.nn.functional.interpolate(
            pred, size=(image.height, image.width), mode="bicubic", align_corners=False
        ).squeeze()
    disp = pred.detach().cpu().numpy().astype(np.float32)
    del out, pred, inputs
    return disp


def mock_disparity(image: Image.Image) -> np.ndarray:
    """Test-mode stand-in: a smooth gradient (bottom = near). Not a model output."""
    h, w = image.height, image.width
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    return np.repeat(0.2 + 0.8 * y, w, axis=1)


def refine_disparity(disp: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize to [0,1] (robust percentiles), repair invalid values, edge-aware smooth."""
    disp = disp.astype(np.float32)
    invalid = ~np.isfinite(disp)
    if invalid.any():
        disp[invalid] = np.nanmedian(disp[~invalid]) if (~invalid).any() else 0.0
    lo, hi = np.percentile(disp, [1.0, 99.5])
    norm = np.clip((disp - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)

    # Edge-aware smoothing guided by the photo keeps object boundaries crisp.
    try:
        guide = rgb.astype(np.uint8)
        # Bilateral filter pre-pass smooths noise while keeping high-contrast edges
        norm = cv2.bilateralFilter(norm, d=5, sigmaColor=0.03, sigmaSpace=3)
        # Guided filter snaps depth contours precisely to the photo color edges
        norm = cv2.ximgproc.guidedFilter(guide, norm, radius=3, eps=1e-5)  # needs opencv-contrib
    except Exception:
        norm = cv2.bilateralFilter(norm, d=7, sigmaColor=0.05, sigmaSpace=5)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    stats = {
        "invalid_fraction": float(invalid.mean()),
        "dynamic_range": float(hi - lo),
        "smoothness": float(_smoothness(norm)),
    }
    return norm, stats


def to_depth(norm_disp: np.ndarray, near: float, far: float) -> np.ndarray:
    """Map normalized disparity (1 = nearest) to depth in scene units."""
    inv = 1.0 / far + norm_disp * (1.0 / near - 1.0 / far)
    return (1.0 / inv).astype(np.float32)


def depth_preview(norm_disp: np.ndarray) -> np.ndarray:
    """Colored preview image (near = warm)."""
    u8 = (norm_disp * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _smoothness(d: np.ndarray) -> float:
    gy, gx = np.gradient(d)
    return 1.0 - float(np.clip(np.mean(np.hypot(gx, gy)) * 20, 0, 1))
