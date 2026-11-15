"""Evaluation metrics. Pure NumPy/SciPy so they can be unit-tested without a GPU.

Depth (standard monocular protocol): predictions from an affine-invariant model are
first aligned to ground truth with least-squares scale + shift in *disparity* space,
then compared in depth space on valid pixels within [min_depth, max_depth].
  AbsRel = mean(|p-g|/g)   SqRel = mean((p-g)^2/g)   RMSE   RMSE_log
  δ1, δ2, δ3 = fraction with max(p/g, g/p) < 1.25, 1.25^2, 1.25^3

Segmentation: confusion matrix over non-ignored pixels -> per-class IoU, mIoU over
classes that occur in ground truth or prediction, pixel accuracy, mean class accuracy.

Reconstruction: both meshes are centred and scaled to a unit bounding-box diagonal,
the best of 24 axis-aligned rotations (optionally refined by ICP) is chosen, then
Chamfer-L2 and F-score@τ are computed from uniformly sampled surface points.
"""
from __future__ import annotations

import itertools

import numpy as np

# --------------------------------------------------------------------------- depth
NYU_EIGEN_CROP = (45, 471, 41, 601)      # rows y0:y1, cols x0:x1 on 480x640 frames


def align_disparity(pred_disp: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Least-squares scale/shift of predicted disparity to 1/gt; returns aligned depth."""
    target = 1.0 / gt_depth[valid]
    a = np.stack([pred_disp[valid].astype(np.float64), np.ones(int(valid.sum()))], 1)
    s, t = np.linalg.lstsq(a, target, rcond=None)[0]
    disp = s * pred_disp + t
    disp = np.clip(disp, 1.0 / max(float(gt_depth[valid].max()) * 10, 1e-3), None)
    return (1.0 / disp).astype(np.float32)


def depth_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


def evaluate_depth_pair(pred_disp: np.ndarray, gt: np.ndarray, min_depth: float = 1e-3,
                        max_depth: float = 10.0, eigen_crop: bool = False) -> dict[str, float] | None:
    valid = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    if eigen_crop and gt.shape == (480, 640):
        crop = np.zeros_like(valid)
        y0, y1, x0, x1 = NYU_EIGEN_CROP
        crop[y0:y1, x0:x1] = True
        valid &= crop
    if valid.sum() < 10:
        return None
    pred = align_disparity(pred_disp, gt, valid)
    return depth_errors(pred, gt, valid)


class DepthMeter:
    """Averages per-image metrics (the convention used by most depth papers)."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.n = 0

    def update(self, m: dict[str, float] | None) -> None:
        if m is None:
            return
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.n += 1

    def result(self) -> dict[str, float]:
        return {k: v / self.n for k, v in self.sums.items()} | {"images": self.n} if self.n else {"images": 0}


# --------------------------------------------------------------------------- segmentation
class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.k = num_classes
        self.ignore = ignore_index
        self.mat = np.zeros((num_classes, num_classes), np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        keep = gt != self.ignore
        g, p = gt[keep].astype(np.int64), pred[keep].astype(np.int64)
        inside = (g >= 0) & (g < self.k) & (p >= 0) & (p < self.k)
        self.mat += np.bincount(self.k * g[inside] + p[inside], minlength=self.k ** 2).reshape(self.k, self.k)

    def result(self, class_names: list[str] | None = None) -> dict:
        tp = np.diag(self.mat).astype(np.float64)
        gt_count = self.mat.sum(1).astype(np.float64)
        pred_count = self.mat.sum(0).astype(np.float64)
        union = gt_count + pred_count - tp
        present = union > 0
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=present)
        acc = np.divide(tp, gt_count, out=np.zeros_like(tp), where=gt_count > 0)
        names = class_names or [str(i) for i in range(self.k)]
        return {
            "miou": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_acc": float(tp.sum() / max(self.mat.sum(), 1)),
            "mean_acc": float(acc[gt_count > 0].mean()) if (gt_count > 0).any() else 0.0,
            "per_class_iou": {names[i]: round(float(iou[i]), 4) for i in range(self.k) if present[i]},
        }


# --------------------------------------------------------------------------- reconstruction
def normalize_points(points: np.ndarray) -> np.ndarray:
    lo, hi = points.min(0), points.max(0)
    centre = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    return (points - centre) / diag


def _rotations() -> list[np.ndarray]:
    """The 24 proper rotations that map axes onto axes."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def chamfer_fscore(pred: np.ndarray, gt: np.ndarray, tau: float = 0.05) -> dict[str, float]:
    from scipy.spatial import cKDTree
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    precision = float(np.mean(d_pg < tau))
    recall = float(np.mean(d_gp < tau))
    f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"chamfer_l2": float(np.mean(d_pg ** 2) + np.mean(d_gp ** 2)),
            "fscore": f, "precision": precision, "recall": recall}


def icp(src: np.ndarray, dst: np.ndarray, iters: int = 20) -> np.ndarray:
    """Rigid ICP (rotation + translation). Returns the transformed source points."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    cur = src.copy()
    for _ in range(iters):
        _, idx = tree.query(cur)
        m = dst[idx]
        mu_s, mu_d = cur.mean(0), m.mean(0)
        u, _, vt = np.linalg.svd((cur - mu_s).T @ (m - mu_d))
        r = (u @ vt).T
        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = (u @ vt).T
        cur = (cur - mu_s) @ r.T + mu_d
    return cur


def compare_shapes(pred_pts: np.ndarray, gt_pts: np.ndarray, tau: float = 0.05,
                   search_rotations: bool = True, refine_icp: bool = True) -> dict[str, float]:
    gt_n = normalize_points(gt_pts)
    best = None
    for r in (_rotations() if search_rotations else [np.eye(3)]):
        cand = normalize_points(pred_pts @ r.T)
        score = chamfer_fscore(cand, gt_n, tau)
        if best is None or score["chamfer_l2"] < best[0]["chamfer_l2"]:
            best = (score, cand)
    score, cand = best
    if refine_icp:
        refined = icp(cand, gt_n)
        s2 = chamfer_fscore(refined, gt_n, tau)
        if s2["chamfer_l2"] < score["chamfer_l2"]:
            score = s2
    return score
