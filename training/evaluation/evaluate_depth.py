"""Evaluate a depth model on a prepared manifest split (default: test).

  python -m training.evaluation.evaluate_depth --manifest training/data/manifests/depth_nyu_kaggle.csv
  python -m training.evaluation.evaluate_depth --model pretrained      # baseline before fine-tuning
  python -m training.evaluation.evaluate_depth --model checkpoints/runs/depth/<run>/export

Runs the backend's own inference code at full image resolution, aligns scale/shift in
disparity space, applies the Eigen crop on 480x640 NYU frames, and reports AbsRel,
SqRel, RMSE, RMSE-log and δ1-3 averaged over images, per source and overall.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..datasets.common import abs_path, load_depth_png, read_manifest
from .metrics import DepthMeter, evaluate_depth_pair
from ._backend import resolve_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="production")
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from app.core import gpu
    from app.core.config import settings
    from app.depth_estimation import depth
    model_id = resolve_model("depth", a.model)
    object.__setattr__(settings, "depth_model_id", model_id)
    bundle = depth.load(gpu.select_device())
    rows = read_manifest(a.manifest, a.split)[: a.limit or None]
    meters: dict[str, DepthMeter] = {}
    t0 = time.time()
    for r in tqdm(rows, desc=f"depth eval ({model_id})"):
        img = Image.open(abs_path(r["image"])).convert("RGB")
        gt = load_depth_png(abs_path(r["depth"]))
        disp = depth.predict_disparity(bundle, img)
        m = evaluate_depth_pair(disp, gt, max_depth=a.max_depth, eigen_crop=r["source"] == "nyu")
        meters.setdefault(r["source"], DepthMeter()).update(m)
        meters.setdefault("all", DepthMeter()).update(m)
    result = {"model": model_id, "manifest": str(a.manifest), "split": a.split,
              "seconds_per_image": round((time.time() - t0) / max(len(rows), 1), 3),
              "peak_vram_gb": gpu.peak_vram_gb(), **{k: v.result() for k, v in meters.items()}}
    print(json.dumps(result, indent=2))
    out = Path("training/logs") / f"eval_depth_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
