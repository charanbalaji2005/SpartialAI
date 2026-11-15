"""Convert COCO 2017 (instances_*.json) into the YOLO layout used by Ultralytics.

  training/data/processed/detection/coco_yolo/
    images/{train,val}/xxx.jpg     hard links to the originals (or copies)
    labels/{train,val}/xxx.txt     one line per box: class cx cy w h (normalised 0..1)
    data.yaml                      paths + the full class-name list

COCO category ids (1..90, with gaps) are remapped to contiguous 0..N-1. Crowd boxes
and degenerate boxes are dropped; boxes are clipped to the image. `--classes` keeps
a subset (e.g. scene objects SpatialAI cares about) and remaps them again.

Example:
  python -m training.datasets.prepare_detection --coco training/data/raw/coco --preset spatialai
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from .common import MANIFEST_DIR, PROCESSED_DIR, ValidationReport, check_image, link_or_copy

# COCO classes that matter for scene reconstruction (furniture, vehicles, people, props).
SPATIALAI_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "bench", "bird", "cat", "dog", "horse", "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "cup", "bowl", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "keyboard", "microwave", "oven", "sink", "refrigerator", "book", "clock", "vase",
]


def convert_split(ann_path: Path, images_dir: Path, out: Path, split: str, keep: list[str] | None,
                  max_images: int | None, min_box_px: float, link: bool) -> tuple[dict, list[str], ValidationReport]:
    coco = json.loads(ann_path.read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    names_all = [c["name"] for c in cats]
    names = [n for n in names_all if keep is None or n in keep]
    if keep:
        unknown = sorted(set(keep) - set(names_all))
        if unknown:
            raise ValueError(f"Unknown COCO classes: {unknown}")
    cat_to_idx = {c["id"]: names.index(c["name"]) for c in cats if c["name"] in names}

    boxes_by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) or a["category_id"] not in cat_to_idx:
            continue
        boxes_by_img[a["image_id"]].append(a)

    report = ValidationReport()
    stats: Counter = Counter()
    images = sorted(coco["images"], key=lambda im: im["id"])
    if keep:      # only images that contain at least one kept class
        images = [im for im in images if boxes_by_img.get(im["id"])]
    if max_images:
        images = images[:max_images]

    for im in tqdm(images, desc=f"COCO {split}"):
        report.total += 1
        src = images_dir / im["file_name"]
        ok, why, size = check_image(src)
        if not ok:
            report.reject(why, im["file_name"])
            continue
        w, h = size
        if (w, h) != (im["width"], im["height"]):
            report.reject("size_mismatch_with_annotation", im["file_name"])
            continue
        lines = []
        for a in boxes_by_img.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            if x1 - x0 < min_box_px or y1 - y0 < min_box_px:
                stats["dropped_tiny_box"] += 1
                continue
            cls = cat_to_idx[a["category_id"]]
            lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            stats[names[cls]] += 1
        dst_img = out / "images" / split / im["file_name"]
        if link:
            link_or_copy(src, dst_img)
        label = out / "labels" / split / (Path(im["file_name"]).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""))   # empty file = background image
        report.kept += 1
    return dict(stats), names, report


def prepare(coco_root: Path, keep: list[str] | None = None, max_train: int | None = None,
            max_val: int | None = None, min_box_px: float = 2.0, link: bool = True,
            out_root: Path = PROCESSED_DIR, name: str = "coco_yolo", manifest_dir: Path = MANIFEST_DIR) -> dict:
    out = out_root / "detection" / name
    ann = coco_root / "annotations"
    splits = {"train": ("instances_train2017.json", "train2017", max_train),
              "val": ("instances_val2017.json", "val2017", max_val)}
    summary: dict = {"output": str(out), "splits": {}}
    names: list[str] = []
    for split, (ann_file, img_dir, limit) in splits.items():
        if not (ann / ann_file).exists():
            raise FileNotFoundError(f"Missing {ann / ann_file}")
        stats, names, report = convert_split(ann / ann_file, coco_root / img_dir, out, split, keep,
                                             limit, min_box_px, link)
        summary["splits"][split] = {**report.summary(), "boxes_per_class": stats}
    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                 "names": {i: n for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"detection_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    summary["data_yaml"] = str(out / "data.yaml")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, type=Path, help="folder with annotations/, train2017/, val2017/")
    ap.add_argument("--preset", choices=["all", "spatialai"], default="all")
    ap.add_argument("--classes", nargs="*", default=None, help="explicit class names to keep")
    ap.add_argument("--max-train", type=int, default=None, help="subset size for quick experiments")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    keep = a.classes or (SPATIALAI_CLASSES if a.preset == "spatialai" else None)
    name = a.name or ("coco_yolo" if keep is None else "coco_yolo_spatialai")
    s = prepare(a.coco, keep, a.max_train, a.max_val, link=True, name=name)
    print(json.dumps({"data_yaml": s["data_yaml"], **{k: {kk: vv for kk, vv in v.items() if kk != "boxes_per_class"}
                                                       for k, v in s["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
