import csv
import json

import numpy as np
import trimesh
import yaml
from PIL import Image

from training.datasets import prepare_depth, prepare_detection, prepare_reconstruction, prepare_segmentation
from training.datasets.common import abs_path, check_leakage, load_depth_png, read_manifest
from training.datasets.validate_dataset import validate


def test_depth_folder_prepare_and_split(depth_folder, tmp_path):
    s = prepare_depth.prepare("folder", depth_folder, max_side=64, out_root=tmp_path / "out",
                              manifest_dir=tmp_path / "m")
    rows = read_manifest(tmp_path / "m" / "depth_folder.csv")
    assert s["samples"] == 24 and set(s["splits"]) == {"train", "val", "test"}
    assert check_leakage(rows, "scene") == {}
    d = load_depth_png(abs_path(rows[0]["depth"]))
    img = Image.open(abs_path(rows[0]["image"]))
    assert d.shape == (img.height, img.width) and max(img.size) == 64
    # mm encoding round-trips to within 1 mm
    assert 1.19 <= d[d > 0].min() <= 1.21
    assert validate(tmp_path / "m" / "depth_folder.csv")["ok"]


def test_validator_catches_corruption(depth_folder, tmp_path):
    prepare_depth.prepare("folder", depth_folder, out_root=tmp_path / "out", manifest_dir=tmp_path / "m")
    rows = read_manifest(tmp_path / "m" / "depth_folder.csv")
    abs_path(rows[3]["image"]).write_bytes(b"not a png")
    res = validate(tmp_path / "m" / "depth_folder.csv")
    assert not res["ok"] and res["problems"]["corrupt_image"]["count"] == 1


def test_nyu_kaggle_depth_encodings(tmp_path):
    base = tmp_path / "nyu_data"
    for split, folder, dtype, value in (("train", "living_room_0001_out", np.uint8, 51), ("test", "x", np.uint16, 2500)):
        d = base / "data" / f"nyu2_{split}" / folder
        d.mkdir(parents=True)
        Image.fromarray(np.full((48, 64, 3), 120, np.uint8)).save(d / "1.jpg")
        Image.fromarray(np.full((48, 64), value, dtype)).save(d / "1.png")
        (base / "data" / f"nyu2_{split}.csv").write_text(
            f"data/nyu2_{split}/{folder}/1.jpg,data/nyu2_{split}/{folder}/1.png\n")
    prepare_depth.prepare("nyu_kaggle", base, out_root=tmp_path / "out", manifest_dir=tmp_path / "m")
    rows = {r["split"]: r for r in read_manifest(tmp_path / "m" / "depth_nyu_kaggle.csv")}
    assert abs(load_depth_png(abs_path(rows["test"]["depth"])).mean() - 2.5) < 1e-3    # 16-bit mm
    train_like = [r for r in read_manifest(tmp_path / "m" / "depth_nyu_kaggle.csv") if r["split"] != "test"][0]
    assert abs(load_depth_png(abs_path(train_like["depth"])).mean() - 2.0) < 1e-2      # 51/255*10 m


def test_hypersim_distance_to_planar():
    dist = np.full((768, 1024), 5.0, np.float32)
    planar = prepare_depth.hypersim_planar(dist)
    assert abs(planar[384, 512] - 5.0) < 1e-3 and planar[0, 0] < planar[384, 512]


def _coco(tmp_path):
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    imgs = []
    for split in ("train2017", "val2017"):
        (root / split).mkdir()
        Image.new("RGB", (200, 100)).save(root / split / "000001.jpg")
        imgs.append(split)
    ann = {"categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 62, "name": "chair"}],
           "images": [{"id": 1, "file_name": "000001.jpg", "width": 200, "height": 100}],
           "annotations": [
               {"image_id": 1, "category_id": 3, "bbox": [10, 10, 50, 40], "iscrowd": 0},
               {"image_id": 1, "category_id": 62, "bbox": [150, 50, 80, 80], "iscrowd": 0},   # clipped
               {"image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "iscrowd": 1},    # crowd
               {"image_id": 1, "category_id": 1, "bbox": [5, 5, 1, 1], "iscrowd": 0},        # tiny
           ]}
    for split in ("train2017", "val2017"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann))
    return root


def test_coco_to_yolo(tmp_path):
    s = prepare_detection.prepare(_coco(tmp_path), out_root=tmp_path / "out", manifest_dir=tmp_path / "m")
    out = tmp_path / "out" / "detection" / "coco_yolo"
    lines = (out / "labels" / "train" / "000001.txt").read_text().split()
    assert len(lines) == 10                      # two boxes, 5 values each
    assert lines[:5] == ["1", "0.175000", "0.300000", "0.250000", "0.400000"]   # car -> contiguous 1
    assert lines[5] == "2" and float(lines[8]) == 0.25                          # chair clipped to 50 px
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "person", 1: "car", 2: "chair"}
    assert (out / "images" / "val" / "000001.jpg").exists()
    assert s["splits"]["train"]["boxes_per_class"]["dropped_tiny_box"] == 1


def test_coco_subset_remaps(tmp_path):
    prepare_detection.prepare(_coco(tmp_path), keep=["chair"], out_root=tmp_path / "out", name="sub", manifest_dir=tmp_path / "m")
    out = tmp_path / "out" / "detection" / "sub"
    assert (out / "labels" / "train" / "000001.txt").read_text().split()[0] == "0"
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == {0: "chair"}


def test_ade20k_masks(tmp_path):
    root = tmp_path / "ADEChallengeData2016"
    for off in ("training", "validation"):
        (root / "images" / off).mkdir(parents=True)
        (root / "annotations" / off).mkdir(parents=True)
        for i in range(6):
            Image.new("RGB", (64, 48), (i * 20, 0, 0)).save(root / "images" / off / f"ADE_{off}_{i}.jpg")
            m = np.zeros((48, 64), np.uint8)
            m[:, 16:] = 1
            m[:, 40:] = 150
            Image.fromarray(m).save(root / "annotations" / off / f"ADE_{off}_{i}.png")
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(root / "annotations" / "training" / "ADE_training_5.png")
    s = prepare_segmentation.prepare("ade20k", root, out_root=tmp_path / "out", manifest_dir=tmp_path / "m",
                                     val_ratio=0.2)
    assert s["problems"]["mask_size_mismatch"]["count"] == 1
    rows = read_manifest(tmp_path / "m" / "segmentation_ade20k.csv")
    m = np.asarray(Image.open(abs_path(rows[0]["mask"])))
    assert m[0, 0] == 255 and m[0, 20] == 0 and m[0, 50] == 149
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_pix3d(tmp_path):
    root = tmp_path / "pix3d"
    entries = []
    for i in range(10):
        cat = "chair" if i % 2 else "table"
        model = f"model/{cat}/m{i // 2}/model.obj"
        (root / model).parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box((1, 2, 1)).export(root / model)
        for kind, im in (("img", Image.new("RGB", (64, 64))), ("mask", Image.fromarray(np.pad(np.full((20, 20), 255, np.uint8), 22)))):
            p = root / kind / cat / f"{i:04d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            im.save(p)
        entries.append({"img": f"img/{cat}/{i:04d}.png", "mask": f"mask/{cat}/{i:04d}.png", "model": model,
                        "category": cat, "bbox": [22, 22, 42, 42], "truncated": i == 9, "occluded": False})
    (root / "pix3d.json").write_text(json.dumps(entries))
    s = prepare_reconstruction.prepare_pix3d(root, manifest_dir=tmp_path / "m")
    assert s["samples"] == 9 and s["leaking_models"] == {}
    assert s["problems"]["truncated_or_occluded"]["count"] == 1
