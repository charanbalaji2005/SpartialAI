"""Box-prompted segmentation.

Primary: Segment Anything (SAM via Transformers) prompted with detector boxes.
Fallback: OpenCV GrabCut initialised from the same box, refined with depth.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..core.config import settings
from ..object_detection.detector import Detection


def load(device: str) -> dict[str, Any]:
    import torch
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained(settings.sam_model_id).to(device).eval()
    processor = SamProcessor.from_pretrained(settings.sam_model_id)
    return {"model": model, "processor": processor, "device": device, "torch": torch}


def segment_sam(bundle: dict[str, Any], image: Image.Image, dets: list[Detection],
                batch: int = 8) -> None:
    """Fill det.mask in place using SAM. Boxes are processed in small batches for VRAM."""
    torch = bundle["torch"]
    proc, model, device = bundle["processor"], bundle["model"], bundle["device"]
    # Image embedding is computed once and reused for every box batch.
    base = proc(image, return_tensors="pt")
    with torch.inference_mode():
        emb = model.get_image_embeddings(base["pixel_values"].to(device))
    for start in range(0, len(dets), batch):
        chunk = dets[start:start + batch]
        inputs = proc(image, input_boxes=[[d.box for d in chunk]], return_tensors="pt")
        inputs.pop("pixel_values", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(image_embeddings=emb, input_boxes=inputs["input_boxes"],
                        multimask_output=False)
        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0]
        scores = out.iou_scores.float().cpu().numpy().reshape(len(chunk), -1)
        for i, det in enumerate(chunk):
            det.mask = clean_mask(masks[i, 0].numpy().astype(bool))
            det.mask_score = float(scores[i, 0])
            det.mask_method = "sam"


def segment_grabcut(rgb: np.ndarray, det: Detection, disparity: np.ndarray | None = None,
                    iterations: int = 4) -> None:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in det.box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        det.mask = None
        return
    if det.kind == "stuff":
        # Regions like floor/sky: colour + depth similarity inside the box is enough.
        det.mask = _region_mask(rgb, disparity, (x0, y0, x1, y1))
        det.mask_method = "region"
        det.mask_score = None
        return
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (x0, y0, x1 - x0, y1 - y0)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd,
                    iterations, cv2.GC_INIT_WITH_RECT)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        fg = np.zeros((h, w), bool)
        fg[y0:y1, x0:x1] = True
    if disparity is not None and fg.any():
        # Drop pixels much farther than the object's median (background leaking through).
        med = np.median(disparity[fg])
        fg &= disparity > med - 0.15
    det.mask = clean_mask(fg)
    box_area = max(1, (x1 - x0) * (y1 - y0))
    det.mask_score = round(float(det.mask.sum()) / box_area, 3) if det.mask is not None else None
    det.mask_method = "grabcut"


def _region_mask(rgb, disparity, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    inside = np.zeros((h, w), bool)
    inside[y0:y1, x0:x1] = True
    mean = np.median(lab[inside], axis=0)
    dist = np.linalg.norm(lab - mean, axis=2)
    mask = inside & (dist <= max(np.percentile(dist[inside], 70), 6.0))
    return clean_mask(mask)


PART_LABELS = {"face", "head", "hand", "arm", "body", "leg", "hair"}


def merge_parts_and_grow(dets: list[Detection], disparity: np.ndarray, max_frac: float = 0.7,
                         min_separation: float = 0.06) -> list[Detection]:
    """Make each object mask cover its whole depth layer, and fold body parts into people.

    Detectors often return a part of a subject (a face, a hand) rather than all of
    it. Anything left outside the mask stays in the background mesh at the
    subject's depth and is stretched into fans and tubes. Here:
      1. a body-part detection mostly inside a person's mask is dropped;
      2. each mask grows over the connected pixels at a similar depth (the same
         layer), provided that layer stands out from the rest of the image;
      3. a face/body part that grew into most of a figure is relabelled "person".
    """
    persons = [d for d in dets if d.label == "person" and d.mask is not None]
    kept = []
    for d in dets:
        if d.mask is not None and d.label in PART_LABELS and any(
                (d.mask & p.mask).sum() > 0.5 * d.mask.sum() for p in persons):
            continue
        kept.append(d)

    d01 = np.clip(np.nan_to_num(disparity.astype(np.float32)), 0, 1)
    h, w = d01.shape
    taken = np.zeros((h, w), bool)
    for det in sorted(kept, key=lambda x: -x.score):
        if det.mask is None:
            continue
        vals = d01[det.mask]
        lo = float(np.percentile(vals, 10)) - 0.04
        layer = (d01 >= lo) & ~taken
        n, lab = cv2.connectedComponents(layer.astype(np.uint8), connectivity=8)
        ids = np.unique(lab[det.mask & layer])
        ids = ids[ids > 0]
        grown = np.isin(lab, ids) if len(ids) else det.mask
        separation = float(np.median(d01[grown])) - float(np.median(d01[~grown])) if (~grown).any() else 0
        if grown.mean() <= max_frac and separation >= min_separation and grown.sum() > det.mask.sum():
            filled = clean_mask(grown | det.mask)
            if filled is not None:
                factor = filled.sum() / max(det.mask.sum(), 1)
                det.mask = filled
                det.mask_method = f"{det.mask_method or 'mask'}+depth-layer"
                if det.label in PART_LABELS and factor > 3:
                    det.label = "person"
        taken |= det.mask
    return kept


def clean_mask(mask: np.ndarray, min_frac: float = 0.002) -> np.ndarray | None:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = labels == largest
    return out if out.mean() >= min_frac else None
