"""Audit a prepared manifest before training.

Checks every row: files exist and decode, image/target sizes match, depth values are
in range, masks only contain known class ids (or 255), no duplicate images (by content
hash) inside or across splits, and no group (scene / object model) shared by splits.

  python -m training.datasets.validate_dataset training/data/manifests/depth_nyu_kaggle.csv
  python -m training.datasets.validate_dataset training/data/manifests/segmentation_ade20k.csv --num-classes 150
Exit code 1 if any problem is found (usable in CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import (IGNORE_INDEX, ValidationReport, abs_path, check_image, check_leakage, file_hash,
                     load_depth_png, read_manifest, split_counts)


def validate(manifest: Path, num_classes: int | None = None, check_hashes: bool = True,
             max_depth: float = 80.0) -> dict:
    rows = read_manifest(manifest)
    kind = "depth" if "depth" in rows[0] else "segmentation" if "mask" in rows[0] and "mesh" not in rows[0] \
        else "reconstruction"
    report = ValidationReport()
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in tqdm(rows, desc=f"validate {manifest.name}"):
        report.total += 1
        img = abs_path(r["image"]) if "image" in r else None
        if img is not None:
            ok, why, size = check_image(img)
            if not ok:
                report.reject(why, r["image"]); continue
        if kind == "depth":
            dp = abs_path(r["depth"])
            if not dp.exists():
                report.reject("missing_depth", r["depth"]); continue
            d = load_depth_png(dp)
            if d.shape[::-1] != size:
                report.reject("depth_size_mismatch", r["depth"]); continue
            valid = d > 0
            if valid.mean() < 0.05:
                report.reject("depth_mostly_invalid", r["depth"]); continue
            if d[valid].max() > max_depth:
                report.reject("depth_out_of_range", r["depth"]); continue
        elif kind == "segmentation":
            mp = abs_path(r["mask"])
            if not mp.exists():
                report.reject("missing_mask", r["mask"]); continue
            m = np.asarray(Image.open(mp))
            if m.shape[::-1] != size:
                report.reject("mask_size_mismatch", r["mask"]); continue
            if num_classes is not None:
                bad = (m != IGNORE_INDEX) & (m >= num_classes)
                if bad.any():
                    report.reject("unknown_class_id", r["mask"]); continue
        else:
            if not abs_path(r["mesh"]).exists():
                report.reject("missing_mesh", r["mesh"]); continue
        if check_hashes and img is not None:
            hashes[file_hash(img)].append((r["split"], r["image"]))
        report.kept += 1

    dupes = {h: v for h, v in hashes.items() if len(v) > 1}
    cross_split = [v for v in dupes.values() if len({s for s, _ in v}) > 1]
    group_key = next((k for k in ("scene", "model_id", "uid", "group") if k in rows[0]), None)
    leaks = check_leakage(rows, group_key) if group_key else {}
    result = {"manifest": str(manifest), "kind": kind, "splits": split_counts(rows), **report.summary(),
              "duplicate_images": len(dupes), "duplicates_across_splits": len(cross_split),
              "duplicate_examples": cross_split[:5], f"groups_in_multiple_splits ({group_key})": len(leaks)}
    result["ok"] = report.kept == report.total and not cross_split and not leaks
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--no-hash", action="store_true", help="skip duplicate detection (faster)")
    a = ap.parse_args()
    res = validate(a.manifest, a.num_classes, not a.no_hash)
    print(json.dumps(res, indent=2))
    out = a.manifest.with_name(a.manifest.stem + "_validation.json")
    out.write_text(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
