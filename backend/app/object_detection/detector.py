"""Open-vocabulary detection with Grounding DINO (Transformers)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..core.config import settings


@dataclass
class Detection:
    label: str
    score: float
    box: list[float]            # x0, y0, x1, y1 in image pixels
    kind: str                   # "thing" | "stuff"
    mask: np.ndarray | None = field(default=None, repr=False)
    mask_score: float | None = None
    mask_method: str | None = None


def load(device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(settings.detector_model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(settings.detector_model_id)
    model.to(device).eval()
    return {"processor": processor, "model": model, "device": device, "torch": torch}


def detect(bundle: dict[str, Any], image: Image.Image, things: list[str], stuff: list[str],
           box_threshold: float = 0.30, text_threshold: float = 0.25) -> list[Detection]:
    torch = bundle["torch"]
    labels = things + stuff
    if not labels:
        return []
    text = ". ".join(labels) + "."
    inputs = bundle["processor"](images=image, text=text, return_tensors="pt").to(bundle["device"])
    with torch.inference_mode():
        outputs = bundle["model"](**inputs)

    post = bundle["processor"].post_process_grounded_object_detection
    params = inspect.signature(post).parameters
    kwargs: dict[str, Any] = {"target_sizes": [(image.height, image.width)]}
    if "threshold" in params:          # newer transformers
        kwargs["threshold"] = box_threshold
    else:                              # older transformers
        kwargs["box_threshold"] = box_threshold
    if "text_threshold" in params:
        kwargs["text_threshold"] = text_threshold
    result = post(outputs, inputs.input_ids, **kwargs)[0]

    raw_labels = result.get("text_labels", result.get("labels"))
    dets: list[Detection] = []
    for score, label, box in zip(result["scores"].tolist(), raw_labels, result["boxes"].tolist()):
        label = str(label).strip() or "object"
        matched = _match(label, labels)
        kind = "stuff" if matched in stuff else "thing"
        dets.append(Detection(matched, float(score), [float(v) for v in box], kind))
    return nms(dets)


def _match(label: str, vocabulary: list[str]) -> str:
    for word in vocabulary:
        if word in label:
            return word
    return label.split()[0] if label else "object"


def nms(dets: list[Detection], iou_thr: float = 0.6) -> list[Detection]:
    """Per-label non-maximum suppression."""
    keep: list[Detection] = []
    for det in sorted(dets, key=lambda d: -d.score):
        if all(d.label != det.label or iou(d.box, det.box) < iou_thr for d in keep):
            keep.append(det)
    return keep


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
