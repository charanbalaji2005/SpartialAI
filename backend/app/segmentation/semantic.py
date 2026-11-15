"""Semantic scene regions with SegFormer (ADE20K classes).

Produces one mask per background region (wall, floor, ceiling, sky, road, ...), which
the composer turns into separately named, separately hideable parts of the scene. Uses
the fine-tuned checkpoint from training when one has been promoted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..core.config import model_source, settings
from ..object_detection.detector import Detection
from .segmenter import clean_mask

# ADE20K label -> SpatialAI region name (several ADE classes merge into one region).
REGIONS = {
    "wall": "wall", "floor": "floor", "ceiling": "ceiling", "sky": "sky", "road": "road",
    "sidewalk": "ground", "earth": "ground", "ground": "ground", "path": "ground", "dirt track": "ground",
    "grass": "grass", "field": "grass", "sand": "ground", "mountain": "mountain", "hill": "mountain",
    "water": "water", "sea": "water", "river": "water", "lake": "water", "building": "building",
    "house": "building", "skyscraper": "building", "tree": "trees", "rug": "rug", "carpet": "rug",
}


def load(device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    path, info = model_source("segmentation", settings.semantic_model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(path)
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return {"model": model, "processor": processor, "device": device, "torch": torch,
            "names": names, "source": info}


def label_map(bundle: dict[str, Any], image: Image.Image) -> np.ndarray:
    torch = bundle["torch"]
    inputs = bundle["processor"](images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = bundle["model"](pixel_values=inputs["pixel_values"].to(bundle["device"])).logits
        logits = torch.nn.functional.interpolate(logits.float(), size=(image.height, image.width),
                                                 mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.int32)


def regions_from_labels(labels: np.ndarray, names: list[str], min_fraction: float = 0.015) -> list[Detection]:
    """Group ADE classes into SpatialAI regions; return one Detection (with mask) per region."""
    merged: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        region = REGIONS.get(name.split(",")[0].strip().lower())
        if region is None:
            continue
        m = labels == idx
        if m.any():
            merged[region] = merged[region] | m if region in merged else m
    out = []
    h, w = labels.shape
    for region, m in merged.items():
        frac = float(m.mean())
        if frac < min_fraction:
            continue
        mask = clean_mask(m, min_frac=min_fraction / 2)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        det = Detection(region, round(frac, 3), [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                        "stuff", mask=mask, mask_score=None, mask_method="segformer")
        out.append(det)
    return out
