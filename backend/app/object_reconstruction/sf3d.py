"""Stable Fast 3D (SF3D) object reconstruction.

SF3D has heavy, Windows-experimental native dependencies, so it lives in its own
virtual environment. SpatialAI calls its official `run.py` as a subprocess on a
masked RGBA crop, then loads the produced GLB and fits it into the scene.

Configure with SPATIALAI_SF3D_PYTHON and SPATIALAI_SF3D_REPO (see README).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from ..core.config import settings

log = logging.getLogger("spatialai.sf3d")


def available() -> tuple[bool, str]:
    if not settings.sf3d_python or not settings.sf3d_repo:
        return False, "Not configured (set SPATIALAI_SF3D_PYTHON and SPATIALAI_SF3D_REPO)."
    if not Path(settings.sf3d_python).exists():
        return False, f"Python not found: {settings.sf3d_python}"
    if not (Path(settings.sf3d_repo) / "run.py").exists():
        return False, f"run.py not found in {settings.sf3d_repo}"
    return True, "Configured"


def make_rgba_crop(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.12) -> Image.Image:
    """Square RGBA crop of the object with transparent background (SF3D's preferred input)."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = np.zeros((side, side, 4), np.uint8)
    sy0, sx0 = cy - side // 2, cx - side // 2
    h, w = mask.shape
    ry0, rx0 = max(0, sy0), max(0, sx0)
    ry1, rx1 = min(h, sy0 + side), min(w, sx0 + side)
    canvas[ry0 - sy0:ry1 - sy0, rx0 - sx0:rx1 - sx0, :3] = rgb[ry0:ry1, rx0:rx1]
    canvas[ry0 - sy0:ry1 - sy0, rx0 - sx0:rx1 - sx0, 3] = mask[ry0:ry1, rx0:rx1] * 255
    return Image.fromarray(canvas, "RGBA").resize((512, 512), Image.LANCZOS)


def reconstruct(crop: Image.Image, work_dir: Path, cancel_check=None) -> trimesh.Trimesh:
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / "input.png"
    crop.save(src)
    out_dir = work_dir / "sf3d_out"
    cmd = [settings.sf3d_python, "run.py", str(src), "--output-dir", str(out_dir),
           "--texture-resolution", str(settings.sf3d_texture_resolution)]
    log.info("Running SF3D: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=settings.sf3d_repo, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = proc.communicate(timeout=settings.sf3d_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Stable Fast 3D timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"Stable Fast 3D failed: {output[-600:]}")
    glbs = sorted(out_dir.rglob("*.glb"))
    if not glbs:
        raise RuntimeError("Stable Fast 3D produced no GLB")
    loaded = trimesh.load(glbs[0], force="scene")
    return loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)


def fit_to_observation(mesh: trimesh.Trimesh, observed_points: np.ndarray) -> trimesh.Trimesh:
    """Scale and place a canonical object mesh over the observed surface.

    SF3D outputs are centred and roughly camera-facing; we match the observed
    width/height and put the object's front at the observed front depth.
    Orientation alignment is approximate.
    """
    mesh = mesh.copy()
    obs_min, obs_max = observed_points.min(0), observed_points.max(0)
    obs_size = obs_max - obs_min
    bmin, bmax = mesh.bounds
    size = bmax - bmin
    scale = float(np.mean([obs_size[0] / max(size[0], 1e-6), obs_size[1] / max(size[1], 1e-6)]))
    mesh.apply_translation(-(bmin + bmax) / 2)
    mesh.apply_scale(scale)
    front_z = float(np.percentile(observed_points[:, 2], 90))   # nearest observed (z is negative)
    half_depth = (mesh.bounds[1][2] - mesh.bounds[0][2]) / 2
    center = np.array([(obs_min[0] + obs_max[0]) / 2, (obs_min[1] + obs_max[1]) / 2,
                       front_z - half_depth])
    mesh.apply_translation(center)
    return mesh
