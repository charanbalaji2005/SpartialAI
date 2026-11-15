"""Depth-based foreground proposals: used when the detector finds no objects.

Without separable objects the whole photo becomes one background surface and
nothing in it gets a back side. When detection returns nothing (unusual
subjects, artwork, a wrong scene class), we split the image into near and far
layers with Otsu's threshold on relative disparity and propose each sizeable
near component as an object. The masks can then be refined with SAM.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection


def propose(disparity: np.ndarray, max_objects: int = 4, min_frac: float = 0.01,
            max_frac: float = 0.75, min_separation: float = 0.08) -> list[Detection]:
    """Near-layer connected components as object proposals.

    `disparity` is normalised to [0, 1] with larger values nearer the camera.
    A component is kept when it covers `min_frac`..`max_frac` of the image and
    is nearer than the rest of the image by at least `min_separation`.
    """
    d = np.clip(np.nan_to_num(disparity.astype(np.float32)), 0.0, 1.0)
    h, w = d.shape
    u8 = cv2.GaussianBlur((d * 255).astype(np.uint8), (0, 0), max(1.0, 0.004 * max(h, w)))
    thr, fg = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = fg > 0
    if not fg.any() or fg.all():
        return []
    k = max(3, int(0.012 * max(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    far_level = float(np.median(d[~fg]))
    out: list[Detection] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] / (h * w)
        if not (min_frac <= area <= max_frac):
            continue
        mask = labels == i
        separation = float(np.median(d[mask])) - far_level
        if separation < min_separation:
            continue
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        # Fill interior holes so the object gets one silhouette.
        filled = mask.astype(np.uint8)
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        det = Detection("foreground object", round(min(0.5, 0.2 + separation), 3),
                        [float(x), float(y), float(x + bw - 1), float(y + bh - 1)], "thing")
        det.mask = filled.astype(bool)
        det.mask_method = "depth-foreground"
        det.mask_score = round(separation, 3)
        out.append(det)
    out.sort(key=lambda dt: -int(dt.mask.sum()))
    return out[:max_objects]
