"""End-to-end trainer tests with tiny models on synthetic data (CPU, ~seconds)."""
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from training.common.checkpoints import CheckpointManager, MonitorSpec, Registry, promote, rollback  # noqa: E402
from training.datasets import prepare_depth, prepare_segmentation  # noqa: E402
from training.trainers import train_depth, train_segmentation  # noqa: E402


def test_ssi_loss_is_affine_invariant():
    gt = torch.rand(2, 16, 16) * 5 + 1
    valid = torch.ones_like(gt, dtype=torch.bool)
    disp = 1 / gt
    good = train_depth.ssi_loss(disp, gt, valid)
    shifted = train_depth.ssi_loss(disp * 7 + 3, gt, valid)
    bad = train_depth.ssi_loss(torch.rand(2, 16, 16), gt, valid)
    assert good < 1e-4 and shifted < 1e-4 and bad > good + 0.01


def _depth_cfg(manifest, root, epochs):
    return {"seed": 0, "device": "cpu", "checkpoint_root": str(root),
            "model": {"name": "tiny-test"},
            "data": {"manifests": [str(manifest)], "input_size": [28, 42], "workers": 0},
            "train": {"epochs": epochs, "batch_size": 2, "grad_accum": 2, "warmup_steps": 1, "amp": "off",
                      "freeze_backbone_epochs": 1, "lr_head": 1e-3, "lr_backbone": 1e-3, "recovery_every": 2},
            "log": {"tensorboard": False}}


def test_depth_training_resume_and_promotion(depth_folder, tmp_path):
    prepare_depth.prepare("folder", depth_folder, out_root=tmp_path / "out", manifest_dir=tmp_path / "m")
    manifest, ck = tmp_path / "m" / "depth_folder.csv", tmp_path / "ck"
    r1 = train_depth.run(_depth_cfg(manifest, ck, 2), run_name="a")
    run = json.loads((ck / "runs" / "depth" / "a" / "run.json").read_text())
    assert run["status"] == "completed" and run["best_value"] is not None
    assert (ck / "runs" / "depth" / "a" / "export" / "tiny.pt").exists()
    assert {"val/abs_rel", "val/delta1"} <= run["metrics"].keys()

    # resume continues from epoch 2 up to epoch 3
    train_depth.run(_depth_cfg(manifest, ck, 3), resume="a")
    lines = (ck / "runs" / "depth" / "a" / "train.log").read_text()
    assert "Resumed at epoch 2" in lines and "epoch 3/3" in lines

    res = promote("depth", "a", train_depth.smoke_test, root=ck)
    assert res["promoted"] and (ck / "production" / "depth" / "manifest.json").exists()
    assert Registry(ck).task("depth")["production"]["run_id"] == "a"
    assert r1["run_id"] == "a"


def test_promotion_requires_improvement_and_rollback(tmp_path):
    ck = tmp_path / "ck"
    for run_id, value in (("good", 0.10), ("worse", 0.20)):
        m = CheckpointManager("depth", MonitorSpec("val/abs_rel", "min"), run_id, root=ck)
        m.save({"model": {}}, {"val/abs_rel": value}, 0,
               export_fn=lambda d: (d / "spatialai.json").write_text("{}"))
    smoke = lambda d: None  # noqa: E731
    assert promote("depth", "good", smoke, root=ck)["promoted"]
    assert not promote("depth", "worse", smoke, root=ck)["promoted"]
    assert promote("depth", "worse", smoke, force=True, root=ck)["promoted"]
    assert rollback("depth", root=ck)["run_id"] == "good"

    def broken(_):
        raise RuntimeError("cannot load")
    with pytest.raises(RuntimeError):
        promote("depth", "good", broken, force=True, root=ck)
    assert Registry(ck).task("depth")["production"]["run_id"] == "good"   # untouched


def test_segmentation_training(tmp_path):
    root = tmp_path / "seg"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        m = np.zeros((40, 40), np.uint8)
        m[:, 20:] = 1
        m[30:, :] = 2
        img = np.zeros((40, 40, 3), np.uint8)
        img[m == 1] = (200, 30, 30)
        img[m == 2] = (30, 200, 30)
        img = np.clip(img + rng.integers(0, 20, img.shape), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(root / f"s{i}.jpg", quality=98)
        Image.fromarray(m).save(root / f"s{i}.png")
    prepare_segmentation.prepare("folder", root, max_side=None, num_classes=3, val_ratio=0.2,
                                 out_root=tmp_path / "out", manifest_dir=tmp_path / "m")
    (tmp_path / "m" / "segmentation_folder_classes.json").write_text(json.dumps(["wall", "sky", "floor"]))
    cfg = {"seed": 0, "device": "cpu", "checkpoint_root": str(tmp_path / "ck"), "model": {"name": "tiny-test"},
           "data": {"manifest": str(tmp_path / "m" / "segmentation_folder.csv"), "crop": 32, "workers": 0},
           "augment": {"scale": [0.9, 1.1]},
           "train": {"epochs": 6, "batch_size": 4, "grad_accum": 1, "lr": 0.05, "warmup_steps": 1, "amp": "off"},
           "log": {"tensorboard": False}}
    result = train_segmentation.run(cfg, run_name="s")
    assert result["best_value"] > 0.5          # colour-coded classes are easy: it must actually learn
    train_segmentation.smoke_test(tmp_path / "ck" / "runs" / "segmentation" / "s" / "export")
