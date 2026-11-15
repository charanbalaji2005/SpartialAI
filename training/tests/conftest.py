import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def make_scene(rng, w=96, h=72):
    """A synthetic RGB-D frame: a sloped floor/wall plus a nearer box; colours encode depth."""
    y = np.linspace(0, 1, h)[:, None]
    depth = np.repeat(4.0 - 2.5 * y, w, 1).astype(np.float32)
    x0, y0 = rng.integers(10, w // 2), rng.integers(10, h // 2)
    depth[y0:y0 + 20, x0:x0 + 25] = 1.2
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = np.clip(255 - depth * 50, 0, 255)
    rgb[..., 1] = (np.arange(w) * 2 % 255)[None, :]
    rgb[..., 2] = 90
    return Image.fromarray(rgb), depth


@pytest.fixture
def depth_folder(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "rgbd"
    for s in range(6):
        d = root / f"scene{s}"
        d.mkdir(parents=True)
        for f in range(4):
            img, depth = make_scene(rng)
            img.save(d / f"f{f}.png")
            np.save(d / f"f{f}_depth.npy", depth)
    return root
