"""Prepare semantic segmentation data (ADE20K SceneParsing) as class-index masks.

Input: ADEChallengeData2016/ with images/{training,validation}/*.jpg and
annotations/{training,validation}/*.png where 0 = "other" and 1..150 are classes.

Output masks use 0..149 for the 150 classes and 255 for "ignore" (the original 0),
the convention SegFormer and most trainers expect. Images are optionally downscaled
(LANCZOS) and masks resized with NEAREST so labels are never blended.

Splits: the official training set is split into train / val (val is a held-out 5%
used for checkpoint selection); the official validation set becomes `test` and is
only used for the final report. A `folder` source accepts <name>.jpg + <name>.png pairs.

Example:
  python -m training.datasets.prepare_segmentation --source ade20k --input training/data/raw/ade20k/ADEChallengeData2016
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import (IGNORE_INDEX, MANIFEST_DIR, PROCESSED_DIR, ValidationReport, check_image,
                     group_split, rel, split_counts, write_manifest)

FIELDS = ["image", "mask", "split", "source", "group", "width", "height", "labeled_ratio", "classes"]


def ade_class_names(root: Path) -> list[str]:
    """Read objectInfo150.(txt|csv) when present; the first name of each row is used."""
    for name in ("objectInfo150.txt", "objectInfo150.csv"):
        p = root / name
        if p.exists():
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
            names = []
            for line in lines:
                parts = [x.strip() for x in line.replace(",", "\t", 4).split("\t") if x.strip()]
                if parts:
                    names.append(parts[-1].split(";")[0].strip())
            if len(names) == 150:
                return names
    return [f"class_{i}" for i in range(150)]


def convert_ade_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int32) - 1                    # 1..150 -> 0..149, 0 -> -1
    m[m < 0] = IGNORE_INDEX
    m[m >= 150] = IGNORE_INDEX
    return m.astype(np.uint8)


def _process(img_p: Path, mask_p: Path, out: Path, key: str, convert, max_side: int | None,
             report: ValidationReport, rows: list, source: str, split: str, num_classes: int) -> None:
    report.total += 1
    ok, why, size = check_image(img_p)
    if not ok:
        report.reject(why, str(img_p)); return
    if not mask_p.exists():
        report.reject("missing_mask", str(img_p)); return
    try:
        mask = np.asarray(Image.open(mask_p))
    except OSError:
        report.reject("corrupt_mask", str(mask_p)); return
    if mask.ndim == 3:
        report.reject("mask_not_single_channel", str(mask_p)); return
    if mask.shape[::-1] != size:
        report.reject("mask_size_mismatch", str(mask_p)); return
    mask = convert(mask)
    img = Image.open(img_p).convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        new = (max(1, round(img.width * s)), max(1, round(img.height * s)))
        img = img.resize(new, Image.LANCZOS)
        mask = np.asarray(Image.fromarray(mask).resize(new, Image.NEAREST))
    labeled = mask != IGNORE_INDEX
    if labeled.mean() < 0.05:
        report.reject("almost_unlabeled", str(mask_p)); return
    if mask[labeled].max(initial=0) >= num_classes:
        report.reject("label_out_of_range", str(mask_p)); return
    out_img, out_mask = out / "images" / f"{key}.jpg", out / "masks" / f"{key}.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img, quality=95)
    Image.fromarray(mask).save(out_mask)
    rows.append({"image": rel(out_img), "mask": rel(out_mask), "split": split, "source": source,
                 "group": key, "width": img.width, "height": img.height,
                 "labeled_ratio": round(float(labeled.mean()), 4),
                 "classes": " ".join(map(str, np.unique(mask[labeled]).tolist()))})
    report.kept += 1


def prepare(source: str, root: Path, max_side: int | None = 512, val_ratio: float = 0.05, seed: int = 0,
            max_images: int | None = None, num_classes: int = 150,
            out_root: Path = PROCESSED_DIR, manifest_dir: Path = MANIFEST_DIR) -> dict:
    out = out_root / "segmentation" / source
    rows: list = []
    report = ValidationReport()
    if source == "ade20k":
        names = ade_class_names(root)
        for official, split in (("training", "train"), ("validation", "test")):
            imgs = sorted((root / "images" / official).glob("*.jpg"))[: max_images or None]
            for img in tqdm(imgs, desc=f"ADE20K {official}"):
                _process(img, root / "annotations" / official / f"{img.stem}.png", out, img.stem,
                         convert_ade_mask, max_side, report, rows, source, split, num_classes)
    elif source == "folder":
        names = [f"class_{i}" for i in range(num_classes)]
        for img in tqdm(sorted(root.rglob("*.jpg"))[: max_images or None], desc="folder"):
            _process(img, img.with_suffix(".png"), out, img.stem, lambda m: m.astype(np.uint8),
                     max_side, report, rows, source, "", num_classes)
    else:
        raise ValueError(source)
    if not rows:
        raise RuntimeError(f"No usable samples found in {root}")

    pending = [r for r in rows if r["split"] in ("", "train")]
    test_ratio = 0.1 if source == "folder" else 0.0
    assign = group_split((r["group"] for r in pending), (1 - val_ratio - test_ratio, val_ratio, test_ratio), seed)
    for r in pending:
        r["split"] = assign[r["group"]]

    manifest = write_manifest(rows, manifest_dir / f"segmentation_{source}.csv", FIELDS)
    (manifest_dir / f"segmentation_{source}_classes.json").write_text(json.dumps(names, indent=1))
    report.save(manifest_dir / f"segmentation_{source}_report.json")
    return {"manifest": str(manifest), "samples": len(rows), "splits": split_counts(rows),
            "num_classes": num_classes, **report.summary()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ade20k", "folder"], default="ade20k")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=None, help="per official split, for quick experiments")
    ap.add_argument("--num-classes", type=int, default=150)
    a = ap.parse_args()
    s = prepare(a.source, a.input, a.max_side, max_images=a.max_images, num_classes=a.num_classes)
    print(json.dumps({k: v for k, v in s.items() if k != "problems"}, indent=2))


if __name__ == "__main__":
    main()
