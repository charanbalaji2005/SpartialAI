"""Evaluate a SegFormer model on a prepared segmentation manifest split (default: test).

  python -m training.evaluation.evaluate_segmentation --manifest training/data/manifests/segmentation_ade20k.csv
  python -m training.evaluation.evaluate_segmentation --model pretrained

Reports mIoU, pixel accuracy, mean class accuracy, per-class IoU, and the mIoU over the
region classes SpatialAI uses for walls, floors, sky and ground.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..datasets import transforms as T
from ..datasets.common import abs_path, read_manifest
from ..trainers.train_segmentation import REGION_CLASSES
from .metrics import ConfusionMatrix
from ._backend import resolve_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="production")
    ap.add_argument("--short-side", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from transformers import SegformerForSemanticSegmentation
    model_id = resolve_model("segmentation", a.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    cm = ConfusionMatrix(len(names))
    rows = read_manifest(a.manifest, a.split)[: a.limit or None]
    t0 = time.time()
    for r in tqdm(rows, desc="segmentation eval"):
        img = Image.open(abs_path(r["image"])).convert("RGB")
        mask = np.asarray(Image.open(abs_path(r["mask"])))
        x, m = T.seg_eval(img, mask, a.short_side)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(pixel_values=torch.from_numpy(x)[None].to(device)).logits
        logits = F.interpolate(logits.float(), size=m.shape, mode="bilinear", align_corners=False)
        cm.update(logits.argmax(1)[0].cpu().numpy(), m)
    res = cm.result(names)
    regions = [v for k, v in res["per_class_iou"].items() if k in REGION_CLASSES]
    result = {"model": model_id, "split": a.split, "images": len(rows), "miou": res["miou"],
              "pixel_acc": res["pixel_acc"], "mean_acc": res["mean_acc"],
              "region_miou": sum(regions) / len(regions) if regions else None,
              "seconds_per_image": round((time.time() - t0) / max(len(rows), 1), 3),
              "per_class_iou": res["per_class_iou"]}
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_iou"}, indent=2))
    out = Path("training/logs") / f"eval_segmentation_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
