"""Reconstruction quality report built only from measured values.

These are internal consistency measures, not accuracy against ground truth.
For ground-truth depth metrics (AbsRel, RMSE, δ1) see scripts/evaluate_depth.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..object_detection.detector import Detection


def build_report(*, depth_stats: dict, compose_stats: dict, things: list[Detection],
                 stuff: list[Detection], scene: dict, timings: dict, peak_vram: float | None,
                 device: str, fallbacks: list[str], models: dict, validation: dict,
                 mock: bool, completeness: dict | None = None) -> dict[str, Any]:
    det_scores = [d.score for d in things]
    mask_scores = [d.mask_score for d in things if d.mask_score is not None]
    methods: dict[str, int] = {}
    for d in things:
        methods[d.mask_method or "none"] = methods.get(d.mask_method or "none", 0) + 1
    return {
        "test_mode": mock,
        "scene": {"category": scene["category"], "group": scene["group"],
                  "confidence": scene["confidence"], "method": scene["method"]},
        "depth": {
            "invalid_fraction": round(depth_stats["invalid_fraction"], 4),
            "smoothness": round(depth_stats["smoothness"], 3),
            "assumed_range": [scene["near"], scene["far"]],
            "note": "Relative depth; absolute scale is assumed from the scene type.",
        },
        "geometry": compose_stats,
        "detection": {
            "objects": len(things), "regions": len(stuff),
            "mean_confidence": round(float(np.mean(det_scores)), 3) if det_scores else None,
            "labels": sorted({d.label for d in things}),
        },
        "segmentation": {
            "methods": methods,
            "mean_mask_score": round(float(np.mean(mask_scores)), 3) if mask_scores else None,
            "note": "SAM score = predicted IoU; GrabCut score = mask area / box area.",
        },
        "export": validation,
        "completeness": None if completeness is None else {
            k: completeness.get(k) for k in ("passed", "objects_total", "objects_complete",
                                             "background_closed", "issues", "possible_intersections",
                                             "out_of_bounds")
        } | {"elements": [
            {k: e.get(k) for k in ("name", "label", "kind", "status", "shape_prior", "profile",
                                   "thickness", "watertight", "boundary_edges", "non_manifold_edges",
                                   "winding_consistent", "outward_normals", "components",
                                   "textured", "uv_valid", "texture_size", "repairs", "method")}
            for e in completeness.get("elements", [])]},
        "performance": {"timings_s": timings, "peak_vram_gb": peak_vram, "device": device},
        "models": models,
        "fallbacks": fallbacks,
        "limitations": [
            "A single photo cannot show hidden surfaces; backs and sides are inferred from "
            "category shape priors and textured from the visible material. They are plausible, not measured.",
            "Depth is relative. Distances and object sizes are approximate.",
            "Thin structures and reflective or transparent surfaces often reconstruct poorly.",
        ],
    }
