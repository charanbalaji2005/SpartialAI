import json

import numpy as np
from fastapi.testclient import TestClient

from app.core import config
from app.main import app
from app.segmentation import semantic


def test_model_source_prefers_promoted_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "checkpoint_dir", tmp_path)
    path, info = config.model_source("depth", "hf/default")
    assert path == "hf/default" and info["source"] == "pretrained"
    prod = tmp_path / "production" / "depth"
    prod.mkdir(parents=True)
    (prod / "manifest.json").write_text(json.dumps({"run_id": "r1", "metric": "val/abs_rel", "value": 0.08}))
    path, info = config.model_source("depth", "hf/default")
    assert path == str(prod) and info["source"] == "fine-tuned" and info["run_id"] == "r1"
    monkeypatch.setattr(config.settings, "use_finetuned", False)
    assert config.model_source("depth", "hf/default")[0] == "hf/default"


def test_regions_merge_ade_classes():
    names = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "grass", "earth"]
    labels = np.zeros((60, 80), np.int32)        # wall
    labels[:10] = 2                              # sky
    labels[45:, :40] = 9                         # earth  -> ground
    labels[45:, 40:] = 6                         # road
    labels[20:30, 30:50] = 7                     # bed (an object, not a region)
    regs = {d.label: d for d in semantic.regions_from_labels(labels, names)}
    assert set(regs) == {"wall", "sky", "ground", "road"}
    assert regs["sky"].mask[:10].all() and regs["ground"].mask_method == "segformer"


def test_training_registry_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "checkpoint_dir", tmp_path)
    (tmp_path / "registry.json").write_text(json.dumps({"version": 1, "tasks": {"depth": {
        "production": {"run_id": "r1", "metric": "val/abs_rel", "value": 0.08},
        "runs": {"r1": {"status": "completed", "best_value": 0.08, "metrics": {"val/abs_rel": 0.08}}}}}}))
    with TestClient(app) as c:
        data = c.get("/api/training/registry").json()
        assert data["tasks"]["depth"]["production"]["run_id"] == "r1"
        assert data["tasks"]["segmentation"]["runs"] == []
        models = {m["name"]: m for m in c.get("/api/models/status").json()["models"]}
        assert models["semantic"]["source"]["source"] == "pretrained"
