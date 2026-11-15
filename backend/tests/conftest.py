"""Tests run in test mode (deterministic stand-ins for neural networks) on CPU.

Integration tests that need real model weights live in test_real_models.py and
are skipped automatically when torch/transformers or the weights are missing.
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_TMP = tempfile.mkdtemp(prefix="spatialai_test_")
os.environ["SPATIALAI_MOCK_MODELS"] = "1"
os.environ["SPATIALAI_STORAGE"] = _TMP
os.environ["SPATIALAI_DB"] = str(Path(_TMP) / "test.db")
os.environ["SPATIALAI_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic_scene(w: int = 320, h: int = 240) -> np.ndarray:
    """A simple 'room': gradient wall, darker floor, and a red box in front."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: int(h * 0.6)] = np.linspace(180, 220, w, dtype=np.uint8)[None, :, None]
    img[int(h * 0.6):] = (90, 70, 50)
    img[int(h * 0.35):int(h * 0.8), int(w * 0.38):int(w * 0.62)] = (200, 40, 40)
    return img


@pytest.fixture
def scene_rgb():
    return synthetic_scene()


@pytest.fixture
def jpeg_bytes(scene_rgb):
    buf = io.BytesIO()
    Image.fromarray(scene_rgb).save(buf, format="JPEG", quality=92)
    return buf.getvalue()
