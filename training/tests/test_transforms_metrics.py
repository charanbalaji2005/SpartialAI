import random

import numpy as np
from PIL import Image

from training.datasets import transforms as T
from training.evaluation import metrics as M


def test_depth_aug_keeps_alignment():
    # Image red channel == depth*10 everywhere, so any misalignment shows up.
    depth = np.tile(np.linspace(1, 20, 160, dtype=np.float32), (120, 1))
    depth[40:80, 60:100] = 3.0
    img = Image.fromarray(np.stack([np.round(depth * 10).astype(np.uint8)] * 3, -1))
    for seed in range(5):
        x, d = T.depth_train(img, depth, (56, 70), random.Random(seed), color=0.0)
        red = x[0] * T.STD[0] + T.MEAN[0]
        assert x.shape == (3, 56, 70) and d.shape == (56, 70)
        # nearest-resized depth vs bilinear image: compare away from edges
        assert np.median(np.abs(red * 255 / 10 - d)) < 0.6


def test_seg_aug_pads_with_ignore():
    img = Image.new("RGB", (40, 30))
    mask = np.ones((30, 40), np.uint8)
    x, m = T.seg_train(img, mask, 64, random.Random(0), scale=(1.0, 1.0))
    assert x.shape == (3, 64, 64) and m.shape == (64, 64)
    assert set(np.unique(m)) <= {1, 255}


def test_depth_metrics_affine_invariant():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 8, (40, 50)).astype(np.float32)
    perfect = M.evaluate_depth_pair(3.0 / gt + 0.2, gt)          # any affine disparity
    assert perfect["abs_rel"] < 1e-5 and perfect["delta1"] == 1.0
    noisy = M.evaluate_depth_pair(1 / gt + rng.normal(0, 0.05, gt.shape), gt)
    assert noisy["abs_rel"] > 0.01 and noisy["delta1"] < 1.0


def test_confusion_matrix():
    cm = M.ConfusionMatrix(3)
    gt = np.array([[0, 0, 1, 1], [2, 2, 255, 255]])
    pr = np.array([[0, 1, 1, 1], [2, 0, 0, 0]])
    cm.update(pr, gt)
    r = cm.result(["a", "b", "c"])
    assert np.isclose(r["per_class_iou"]["a"], 1 / 3, atol=1e-3) and np.isclose(r["per_class_iou"]["b"], 2 / 3, atol=1e-3)
    assert np.isclose(r["pixel_acc"], 4 / 6)


def test_shape_comparison_finds_rotation():
    import trimesh
    gt = trimesh.creation.box((1, 2, 0.5)).sample(3000, seed=1) if hasattr(trimesh.creation.box(), "sample") else None
    gt = trimesh.sample.sample_surface(trimesh.creation.box((1, 2, 0.5)), 3000, seed=1)[0]
    rot = gt @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T * 3 + 5        # rotated, scaled, moved
    same = M.compare_shapes(rot, gt)
    other = M.compare_shapes(trimesh.sample.sample_surface(trimesh.creation.icosphere(), 3000, seed=1)[0], gt)
    assert same["fscore"] > 0.95 and same["chamfer_l2"] < other["chamfer_l2"]
