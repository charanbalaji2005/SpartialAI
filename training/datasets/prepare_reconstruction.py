"""Prepare single-image 3D reconstruction data for evaluation (Pix3D) and assets (Objaverse).

Pix3D: reads pix3d.json, checks every image/mask/model, drops truncated or occluded
samples by default (their ground truth doesn't match what's visible), and splits by
3D model so the same chair never appears in both train and test.

Objaverse: validates a folder of downloaded GLBs (see download.py) and records mesh
statistics (faces, watertightness, textures) with an LVIS category when known.

Examples:
  python -m training.datasets.prepare_reconstruction pix3d --input training/data/raw/pix3d
  python -m training.datasets.prepare_reconstruction objaverse --input training/data/raw/objaverse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from .common import (MANIFEST_DIR, ValidationReport, check_image, check_leakage, group_split, rel,
                     split_counts, write_manifest)

PIX3D_FIELDS = ["image", "mask", "mesh", "category", "model_id", "split", "bbox", "width", "height",
                "mask_ratio", "faces", "watertight"]
OBJ_FIELDS = ["mesh", "uid", "category", "split", "faces", "vertices", "watertight", "textured", "extent"]


def mesh_stats(path: Path) -> dict | None:
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception:
        return None
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    if not np.isfinite(loaded.vertices).all():
        return None
    textured = bool(getattr(loaded.visual, "uv", None) is not None)
    return {"faces": int(len(loaded.faces)), "vertices": int(len(loaded.vertices)),
            "watertight": bool(loaded.is_watertight), "textured": textured,
            "extent": round(float(np.ptp(loaded.vertices, axis=0).max()), 4)}


def prepare_pix3d(root: Path, include_occluded: bool = False, seed: int = 0,
                  manifest_dir: Path = MANIFEST_DIR) -> dict:
    entries = json.loads((root / "pix3d.json").read_text())
    report = ValidationReport()
    rows = []
    mesh_cache: dict[str, dict | None] = {}
    for e in tqdm(entries, desc="Pix3D"):
        report.total += 1
        if not include_occluded and (e.get("truncated") or e.get("occluded")):
            report.reject("truncated_or_occluded", e["img"]); continue
        img, mask, model = root / e["img"], root / e["mask"], root / e["model"]
        ok, why, size = check_image(img)
        if not ok:
            report.reject(why, e["img"]); continue
        if not mask.exists():
            report.reject("missing_mask", e["img"]); continue
        m = np.asarray(Image.open(mask).convert("L")) > 127
        if m.shape[::-1] != size:
            report.reject("mask_size_mismatch", e["img"]); continue
        if m.mean() < 0.005:
            report.reject("empty_mask", e["img"]); continue
        if e["model"] not in mesh_cache:
            mesh_cache[e["model"]] = mesh_stats(model) if model.exists() else None
        stats = mesh_cache[e["model"]]
        if stats is None:
            report.reject("invalid_or_missing_mesh", e["model"]); continue
        rows.append({"image": rel(img), "mask": rel(mask), "mesh": rel(model), "category": e["category"],
                     "model_id": e["model"], "split": "", "bbox": " ".join(map(str, e.get("bbox", []))),
                     "width": size[0], "height": size[1], "mask_ratio": round(float(m.mean()), 4),
                     "faces": stats["faces"], "watertight": stats["watertight"]})
        report.kept += 1
    if not rows:
        raise RuntimeError(f"No usable Pix3D samples under {root}")
    assign = group_split((r["model_id"] for r in rows), (0.7, 0.1, 0.2), seed)
    for r in rows:
        r["split"] = assign[r["model_id"]]
    manifest = write_manifest(rows, manifest_dir / "reconstruction_pix3d.csv", PIX3D_FIELDS)
    report.save(manifest_dir / "reconstruction_pix3d_report.json")
    cats: dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return {"manifest": str(manifest), "samples": len(rows), "models": len(mesh_cache),
            "splits": split_counts(rows), "categories": cats,
            "leaking_models": check_leakage(rows, "model_id"), **report.summary()}


def prepare_objaverse(root: Path, seed: int = 0, manifest_dir: Path = MANIFEST_DIR) -> dict:
    meta_path = root / "objects.json"          # written by download.py: uid -> {path, category}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    report = ValidationReport()
    rows = []
    for glb in tqdm(sorted(root.rglob("*.glb")), desc="Objaverse"):
        report.total += 1
        stats = mesh_stats(glb)
        if stats is None:
            report.reject("invalid_mesh", glb.name); continue
        if stats["faces"] > 2_000_000:
            report.reject("too_many_faces", glb.name); continue
        uid = glb.stem
        rows.append({"mesh": rel(glb), "uid": uid, "category": meta.get(uid, {}).get("category", "unknown"),
                     "split": "", **stats})
        report.kept += 1
    if not rows:
        raise RuntimeError(f"No GLB files under {root}. Run download.py objaverse first.")
    assign = group_split((r["uid"] for r in rows), (0.8, 0.1, 0.1), seed)
    for r in rows:
        r["split"] = assign[r["uid"]]
    manifest = write_manifest(rows, manifest_dir / "reconstruction_objaverse.csv", OBJ_FIELDS)
    report.save(manifest_dir / "reconstruction_objaverse_report.json")
    return {"manifest": str(manifest), "samples": len(rows), "splits": split_counts(rows), **report.summary()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["pix3d", "objaverse"])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--include-occluded", action="store_true")
    a = ap.parse_args()
    s = prepare_pix3d(a.input, a.include_occluded) if a.dataset == "pix3d" else prepare_objaverse(a.input)
    print(json.dumps({k: v for k, v in s.items() if k != "problems"}, indent=2))


if __name__ == "__main__":
    main()
