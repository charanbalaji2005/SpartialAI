"""Prepare RGB + depth pairs for depth fine-tuning and evaluation.

Supported sources (pick with --source):
  nyu_mat      Official NYU Depth V2 labeled set: nyu_depth_v2_labeled.mat (+ optional splits.mat).
               1449 aligned RGB-D frames; official 795/654 train/test split when splits.mat is given.
  nyu_kaggle   The widely mirrored DenseDepth export ("nyu_data.zip" on Kaggle): data/nyu2_train.csv,
               data/nyu2_test.csv. Train depth PNGs are 8-bit (0..255 -> 0..10 m), test are 16-bit mm.
  hypersim     Apple Hypersim scenes: images/scene_cam_XX_final_preview/frame.NNNN.tonemap.jpg with
               scene_cam_XX_geometry_hdf5/frame.NNNN.depth_meters.hdf5. Hypersim stores ray distance;
               it is converted to planar depth here.
  folder       Any folder of <name>.<jpg|png> + <name>_depth.<png|npy> (png = mm, npy = metres).

Output: training/data/processed/depth/<source>/{rgb,depth}/ and manifests/depth_<source>.csv
Splits are by scene, so frames of one room never appear in both train and test.

Example:
  python -m training.datasets.prepare_depth --source nyu_kaggle --input training/data/raw/nyu_depth/nyu_data
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import (MANIFEST_DIR, PROCESSED_DIR, ValidationReport, check_image, check_leakage,
                     group_split, rel, save_depth_png, split_counts, write_manifest)

FIELDS = ["image", "depth", "scene", "split", "source", "width", "height", "valid_ratio", "min_m", "max_m"]


def _resize_pair(rgb: Image.Image, depth: np.ndarray, max_side: int | None) -> tuple[Image.Image, np.ndarray]:
    if depth.shape[:2] != (rgb.height, rgb.width):
        # Depth must be pixel-aligned with RGB; nearest keeps real measurements (no blending across edges).
        depth = np.asarray(Image.fromarray(depth).resize(rgb.size, Image.NEAREST))
    if max_side and max(rgb.size) > max_side:
        s = max_side / max(rgb.size)
        size = (max(1, round(rgb.width * s)), max(1, round(rgb.height * s)))
        rgb = rgb.resize(size, Image.LANCZOS)
        depth = np.asarray(Image.fromarray(depth).resize(size, Image.NEAREST))
    return rgb, depth


def _emit(rows: list, report: ValidationReport, out: Path, source: str, key: str, scene: str,
          rgb: Image.Image, depth_m: np.ndarray, max_side: int | None, min_valid: float) -> None:
    report.total += 1
    rgb = rgb.convert("RGB")
    rgb, depth_m = _resize_pair(rgb, depth_m.astype(np.float32), max_side)
    valid = np.isfinite(depth_m) & (depth_m > 1e-3)
    ratio = float(valid.mean())
    if ratio < min_valid:
        report.reject("too_little_valid_depth", key)
        return
    img_path = out / "rgb" / f"{key}.png"
    dep_path = out / "depth" / f"{key}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(img_path)
    save_depth_png(np.where(valid, depth_m, 0), dep_path)
    rows.append({"image": rel(img_path), "depth": rel(dep_path), "scene": scene, "split": "",
                 "source": source, "width": rgb.width, "height": rgb.height,
                 "valid_ratio": round(ratio, 4), "min_m": round(float(depth_m[valid].min()), 3),
                 "max_m": round(float(depth_m[valid].max()), 3)})
    report.kept += 1


# --------------------------------------------------------------------------- sources
def from_nyu_mat(mat: Path, out: Path, splits_mat: Path | None, **kw) -> tuple[list, ValidationReport, dict | None]:
    import h5py
    rows: list = []
    report = ValidationReport()
    official = None
    if splits_mat and splits_mat.exists():
        from scipy.io import loadmat
        sm = loadmat(splits_mat)
        official = {int(i) - 1: "train" for i in sm["trainNdxs"].ravel()}
        official.update({int(i) - 1: "test" for i in sm["testNdxs"].ravel()})
    with h5py.File(mat, "r") as f:
        images, depths = f["images"], f["depths"]
        scenes = [''.join(chr(c) for c in f[ref][:].ravel()) for ref in f["scenes"][0]]
        for i in tqdm(range(images.shape[0]), desc="NYU .mat"):
            rgb = Image.fromarray(np.asarray(images[i]).transpose(2, 1, 0).astype(np.uint8))
            depth = np.asarray(depths[i]).T.astype(np.float32)
            _emit(rows, report, out, "nyu", f"nyu_{i:05d}", scenes[i], rgb, depth, **kw)
            if official is not None and rows and rows[-1]["image"].endswith(f"nyu_{i:05d}.png"):
                rows[-1]["split"] = official.get(i, "train")
    return rows, report, ({"official_split": True} if official else None)


def from_nyu_kaggle(root: Path, out: Path, **kw) -> tuple[list, ValidationReport, dict | None]:
    import csv
    rows: list = []
    report = ValidationReport()
    base = root if (root / "data").exists() else root.parent
    csvs = {"train": base / "data" / "nyu2_train.csv", "test": base / "data" / "nyu2_test.csv"}
    if not any(p.exists() for p in csvs.values()):
        raise FileNotFoundError(f"Expected data/nyu2_train.csv under {root}")
    for split, path in csvs.items():
        if not path.exists():
            continue
        with path.open() as fh:
            pairs = [r for r in csv.reader(fh) if len(r) >= 2]
        for img_rel, dep_rel in tqdm(pairs, desc=f"NYU kaggle {split}"):
            img_p, dep_p = base / img_rel.strip(), base / dep_rel.strip()
            ok, why, _ = check_image(img_p)
            if not ok or not dep_p.exists():
                report.total += 1
                report.reject(why if not ok else "missing_depth", img_rel)
                continue
            raw = np.asarray(Image.open(dep_p))
            # DenseDepth export: 8-bit train maps encode 0..10 m; 16-bit test maps are millimetres.
            depth = raw.astype(np.float32) / 255.0 * 10.0 if raw.dtype == np.uint8 else raw.astype(np.float32) / 1000.0
            scene = img_p.parent.name.replace("_out", "")
            key = re.sub(r"[^A-Za-z0-9_]", "_", f"{split}_{scene}_{img_p.stem}")
            _emit(rows, report, out, "nyu", key, scene, Image.open(img_p), depth, **kw)
            if rows and rows[-1]["image"].endswith(f"{key}.png"):
                rows[-1]["split"] = "test" if split == "test" else ""
    return rows, report, {"official_split": True}


def hypersim_planar(dist: np.ndarray, focal: float = 886.81) -> np.ndarray:
    """Hypersim depth_meters is Euclidean distance to the camera centre; convert to planar z."""
    h, w = dist.shape
    x = np.linspace(-w / 2 + 0.5, w / 2 - 0.5, w)[None, :]
    y = np.linspace(-h / 2 + 0.5, h / 2 - 0.5, h)[:, None]
    return dist / np.sqrt(1 + (x / focal) ** 2 + (y / focal) ** 2)


def from_hypersim(root: Path, out: Path, max_frames_per_scene: int = 100, **kw) -> tuple[list, ValidationReport, dict | None]:
    import h5py
    rows: list = []
    report = ValidationReport()
    jpgs = sorted(root.glob("ai_*/images/scene_cam_*_final_preview/frame.*.tonemap.jpg"))
    per_scene: dict[str, int] = {}
    for jpg in tqdm(jpgs, desc="Hypersim"):
        scene = jpg.parts[-4]
        if per_scene.get(scene, 0) >= max_frames_per_scene:
            continue
        cam = jpg.parent.name.replace("_final_preview", "")
        frame = jpg.name.split(".")[1]
        h5 = jpg.parent.parent / f"{cam}_geometry_hdf5" / f"frame.{frame}.depth_meters.hdf5"
        if not h5.exists():
            report.total += 1
            report.reject("missing_depth", str(jpg))
            continue
        with h5py.File(h5, "r") as f:
            dist = np.asarray(f["dataset"], dtype=np.float32)
        depth = hypersim_planar(dist, focal=886.81 * dist.shape[1] / 1024)
        per_scene[scene] = per_scene.get(scene, 0) + 1
        _emit(rows, report, out, "hypersim", f"{scene}_{cam}_{frame}", scene, Image.open(jpg), depth, **kw)
    return rows, report, None


def from_folder(root: Path, out: Path, **kw) -> tuple[list, ValidationReport, dict | None]:
    rows: list = []
    report = ValidationReport()
    for img in tqdm(sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                           and not p.stem.endswith("_depth")), desc="folder"):
        dep_png, dep_npy = img.with_name(img.stem + "_depth.png"), img.with_name(img.stem + "_depth.npy")
        if dep_npy.exists():
            depth = np.load(dep_npy).astype(np.float32)
        elif dep_png.exists():
            depth = np.asarray(Image.open(dep_png), dtype=np.float32) / 1000.0
        else:
            continue
        ok, why, _ = check_image(img)
        if not ok:
            report.total += 1
            report.reject(why, str(img))
            continue
        scene = img.parent.name if img.parent != root else img.stem
        key = re.sub(r"[^A-Za-z0-9_]", "_", f"{scene}_{img.stem}")
        _emit(rows, report, out, "folder", key, scene, Image.open(img), depth, **kw)
    return rows, report, None


# --------------------------------------------------------------------------- main
def prepare(source: str, input_path: Path, max_side: int | None = 640, min_valid: float = 0.3,
            val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 0,
            splits_mat: Path | None = None, max_frames_per_scene: int = 100,
            out_root: Path = PROCESSED_DIR, manifest_dir: Path = MANIFEST_DIR) -> dict:
    out = out_root / "depth" / source
    kw = {"max_side": max_side, "min_valid": min_valid}
    if source == "nyu_mat":
        rows, report, info = from_nyu_mat(input_path, out, splits_mat, **kw)
    elif source == "nyu_kaggle":
        rows, report, info = from_nyu_kaggle(input_path, out, **kw)
    elif source == "hypersim":
        rows, report, info = from_hypersim(input_path, out, max_frames_per_scene=max_frames_per_scene, **kw)
    elif source == "folder":
        rows, report, info = from_folder(input_path, out, **kw)
    else:
        raise ValueError(f"Unknown source {source}")
    if not rows:
        raise RuntimeError(f"No usable samples found in {input_path}. Check the path and layout.")

    # Rows without an official split get a scene-level split. Scenes already in the
    # official test set stay entirely in test so no scene leaks across.
    test_scenes = {r["scene"] for r in rows if r["split"] == "test"}
    unassigned = [r for r in rows if not r["split"] and r["scene"] not in test_scenes]
    if info and info.get("official_split"):
        # Official train data: hold out whole scenes for validation; keep official test as test.
        scene_split = group_split((r["scene"] for r in unassigned), (1 - val_ratio, val_ratio, 0.0), seed)
    else:
        scene_split = group_split((r["scene"] for r in unassigned), (1 - val_ratio - test_ratio, val_ratio, test_ratio), seed)
    for r in rows:
        if not r["split"]:
            r["split"] = "test" if r["scene"] in test_scenes else scene_split.get(r["scene"], "train")

    leaks = check_leakage(rows, "scene")
    manifest = write_manifest(rows, manifest_dir / f"depth_{source}.csv", FIELDS)
    report.save(manifest_dir / f"depth_{source}_report.json")
    summary = {"manifest": str(manifest), "samples": len(rows), "splits": split_counts(rows),
               "scenes": len({r['scene'] for r in rows}), "leaking_scenes": leaks, **report.summary()}
    (manifest_dir / f"depth_{source}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["nyu_mat", "nyu_kaggle", "hypersim", "folder"])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--splits-mat", type=Path, default=None, help="NYU splits.mat for the official split")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--min-valid", type=float, default=0.3, help="minimum fraction of valid depth pixels")
    ap.add_argument("--max-frames-per-scene", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    summary = prepare(a.source, a.input, a.max_side, a.min_valid, seed=a.seed, splits_mat=a.splits_mat,
                      max_frames_per_scene=a.max_frames_per_scene)
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=2))


if __name__ == "__main__":
    main()
