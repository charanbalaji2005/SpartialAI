"""Real-model integration tests. Skipped unless torch + transformers + weights are present.

Run on the target machine with:  set SPATIALAI_REAL=1 && pytest tests/test_real_models.py -v
"""
import os

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(os.getenv("SPATIALAI_REAL") != "1",
                                reason="set SPATIALAI_REAL=1 to run real-model tests")


def test_depth_anything_runs(scene_rgb):
    from app.core import gpu
    from app.depth_estimation import depth
    bundle = depth.load(gpu.select_device())
    disp = depth.predict_disparity(bundle, Image.fromarray(scene_rgb))
    assert disp.shape == scene_rgb.shape[:2] and np.isfinite(disp).all()
    # The red box is in front of the wall: it should read as nearer (higher disparity).
    h, w = disp.shape
    assert disp[int(h * .55), int(w * .5)] > disp[int(h * .15), int(w * .1)]


def test_detector_runs(scene_rgb):
    from app.core import gpu
    from app.object_detection import detector
    bundle = detector.load(gpu.select_device())
    dets = detector.detect(bundle, Image.fromarray(scene_rgb), ["box"], ["floor"], 0.2, 0.2)
    assert isinstance(dets, list)
