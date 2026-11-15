"""Evaluate single-image object reconstruction on Pix3D (test split).

  python -m training.evaluation.evaluate_reconstruction --method relief --limit 100
  python -m training.evaluation.evaluate_reconstruction --method sf3d --limit 50

For each image: take the ground-truth mask, reconstruct the object with the chosen
SpatialAI method, then compare to the Pix3D CAD model: both are normalised to a unit
bounding-box diagonal, the best axis-aligned rotation is searched and refined with
ICP, and Chamfer-L2 and F-score@0.05 are computed on 5,000 surface samples each.

  relief  the app's default: Depth Anything V2 surface + mirrored inferred back
  sf3d    Stable Fast 3D (needs SPATIALAI_SF3D_PYTHON / SPATIALAI_SF3D_REPO)

Rotation search makes the protocol orientation-agnostic, which favours methods with an
unknown canonical pose; report it alongside your numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from ..datasets.common import abs_path, read_manifest
from .metrics import compare_shapes
from . import _backend  # noqa: F401  (adds backend to sys.path)


def reconstruct_relief(bundle, image: Image.Image, mask: np.ndarray) -> trimesh.Trimesh:
    from app.depth_estimation import depth as D
    from app.geometry import mesh as geo
    disp = D.predict_disparity(bundle, image)
    norm, _ = D.refine_disparity(disp, np.asarray(image))
    z = D.to_depth(norm, 1.0, 3.0)
    cam = geo.make_camera(image.width, image.height, 50.0)
    grid = geo.build_grid(z, cam, step=2)
    sel = grid.face_ok & mask[grid.face_px[:, 1], grid.face_px[:, 0]]
    mat = geo.material(image, "eval")
    front = geo.submesh(grid, sel, mat)
    if front is None:
        raise RuntimeError("empty mask")
    width = float(np.ptp(front.vertices[:, 0]))
    back = geo.close_relief(front, max(width * 0.35, 0.02), mat)
    return trimesh.util.concatenate([front, back])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("training/data/manifests/reconstruction_pix3d.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--method", choices=["relief", "sf3d"], default="relief")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--tau", type=float, default=0.05)
    a = ap.parse_args()

    from app.core import gpu
    from app.depth_estimation import depth as D
    from app.object_reconstruction import sf3d
    bundle = D.load(gpu.select_device()) if a.method == "relief" else None
    if a.method == "sf3d":
        ok, why = sf3d.available()
        if not ok:
            raise SystemExit(f"Stable Fast 3D not available: {why}")
    rows = read_manifest(a.manifest, a.split)[: a.limit or None]
    per_cat: dict[str, list[dict]] = defaultdict(list)
    failures = 0
    t0 = time.time()
    for i, r in enumerate(tqdm(rows, desc=f"Pix3D {a.method}")):
        try:
            img = Image.open(abs_path(r["image"])).convert("RGB")
            mask = np.asarray(Image.open(abs_path(r["mask"])).convert("L")) > 127
            if a.method == "relief":
                pred = reconstruct_relief(bundle, img, mask)
            else:
                crop = sf3d.make_rgba_crop(np.asarray(img), mask)
                pred = sf3d.reconstruct(crop, Path("training/logs/sf3d_eval") / f"{i:05d}")
            gt = trimesh.load(abs_path(r["mesh"]), force="mesh")
            score = compare_shapes(pred.sample(a.samples), gt.sample(a.samples), a.tau)
            per_cat[r["category"]].append(score)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  failed on {r['image']}: {exc}")
    def mean(scores, k): return float(np.mean([s[k] for s in scores])) if scores else None
    all_scores = [s for v in per_cat.values() for s in v]
    result = {"method": a.method, "split": a.split, "evaluated": len(all_scores), "failures": failures,
              "chamfer_l2": mean(all_scores, "chamfer_l2"), "fscore": mean(all_scores, "fscore"),
              "per_category": {c: {"n": len(v), "chamfer_l2": mean(v, "chamfer_l2"), "fscore": mean(v, "fscore")}
                               for c, v in per_cat.items()},
              "seconds_per_object": round((time.time() - t0) / max(len(rows), 1), 2),
              "protocol": f"unit-diagonal normalisation, 24-rotation search + ICP, tau={a.tau}"}
    print(json.dumps(result, indent=2))
    out = Path("training/logs") / f"eval_reconstruction_{a.method}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
