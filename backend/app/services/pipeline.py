"""The SpatialAI reconstruction pipeline: photo -> composed, exportable 3D scene.

Every GPU model runs sequentially and is evicted before the next one loads.
Each stage reports real progress; any optional model that is missing or fails
is replaced by a documented fallback and recorded in the quality report.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from ..core import gpu
from ..core.config import QUALITY_MODES, settings
from ..depth_estimation import depth as depth_mod
from ..evaluation.quality import build_report
from ..export import exporter
from ..geometry import mesh as geo
from ..models.registry import manager
from ..object_detection import detector, foreground, yolo
from ..object_detection.detector import Detection
from ..object_reconstruction import sf3d
from ..scene_composition import composer
from ..scene_understanding import classifier
from ..segmentation import segmenter, semantic
from . import images, storage

log = logging.getLogger("spatialai.pipeline")

STAGES: list[tuple[str, str, float]] = [
    ("upload", "Uploading", 0.02),
    ("analysis", "Understanding image", 0.07),
    ("depth", "Estimating depth", 0.14),
    ("detection", "Detecting objects", 0.12),
    ("segmentation", "Segmenting objects", 0.10),
    ("geometry", "Generating 3D geometry", 0.10),
    ("objects", "Reconstructing hidden surfaces", 0.10),
    ("completion", "Generating backside geometry", 0.07),
    ("materials", "Applying textures", 0.06),
    ("repair", "Repairing meshes", 0.06),
    ("validation", "Validating scene", 0.04),
    ("finalize", "Exporting final 3D scene", 0.12),
]
# Stages reported by scene_composition.compose, in order.
COMPOSE_STAGES = ("geometry", "objects", "completion", "materials", "repair", "validation")


class Cancelled(Exception):
    pass


@dataclass
class Context:
    project_id: str
    quality: str
    emit: Callable[[str, float, str], None]        # stage key, fraction within stage, message
    log_line: Callable[[str, str], None]           # level, message
    cancelled: Callable[[], bool]
    timings: dict[str, float] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    models_used: dict[str, str] = field(default_factory=dict)

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled()


def run(ctx: Context) -> dict[str, Any]:
    mode = QUALITY_MODES.get(ctx.quality, QUALITY_MODES["balanced"])
    p = storage.ensure(ctx.project_id)
    original_path = next(p["input"].glob("original.*"), None)
    if original_path is None:
        raise RuntimeError("No uploaded image found for this project.")
    gpu.reset_peak()
    manager.reset_failures()
    t_all = time.time()

    # -------- analysis --------
    t = _start(ctx, "analysis", "Loading and orienting the photo")
    full_img, work_img = images.load_for_processing(original_path.read_bytes(), mode.max_side)
    rgb = images.to_array(work_img)
    h, w = rgb.shape[:2]
    work_img.save(p["output"] / "working.png")
    hfov, fov_source = geo.fov_from_exif(full_img.info.get("exif_source", full_img))
    ctx.log_line("info", f"Working resolution {w}x{h} ({mode.name} mode); FOV {hfov:.1f}° ({fov_source})")
    ctx.check()
    scene = _classify(ctx, work_img, rgb)
    ctx.log_line("info", f"Scene: {scene.category} ({scene.group}, {scene.confidence:.0%}, {scene.method})")
    _end(ctx, "analysis", t)

    # -------- depth --------
    t = _start(ctx, "depth", "Estimating depth")
    disparity = _depth(ctx, work_img)
    norm, depth_stats = depth_mod.refine_disparity(disparity, rgb)
    depth = depth_mod.to_depth(norm, scene.near, scene.far)
    np.save(p["output"] / "depth.npy", depth)
    Image.fromarray(depth_mod.depth_preview(norm)).save(p["output"] / "depth.png")
    Image.fromarray((norm * 65535).astype(np.uint16)).save(p["output"] / "depth16.png")
    manager.unload("depth")
    _end(ctx, "depth", t)
    ctx.check()

    # -------- detection --------
    t = _start(ctx, "detection", "Finding objects and regions")
    dets = _detect(ctx, work_img, scene, mode)
    things = [d for d in dets if d.kind == "thing"]
    stuff = [d for d in dets if d.kind == "stuff"]
    ctx.log_line("info", f"Detected {len(things)} objects, {len(stuff)} regions")
    manager.unload("detector")
    _end(ctx, "detection", t)
    ctx.check()

    # -------- segmentation --------
    t = _start(ctx, "segmentation", "Separating objects from the scene")
    _segment(ctx, work_img, rgb, norm, things, stuff, mode)
    things = _dedupe_masks([d for d in things if d.mask is not None])
    things = _dedupe_masks(segmenter.merge_parts_and_grow(things, norm))[: mode.max_objects]
    stuff = [d for d in stuff if d.mask is not None]
    if not things:
        things = _foreground_fallback(ctx, work_img, norm, mode)
    for k, det in enumerate(things + stuff):
        Image.fromarray(det.mask.astype(np.uint8) * 255).save(
            p["masks"] / f"{k:02d}_{composer.slug(det.label)}.png")
    manager.unload("sam")
    _end(ctx, "segmentation", t)
    ctx.check()

    # -------- geometry, hidden surfaces, completion, textures, repair, validation --------
    cam = geo.make_camera(w, h, hfov)
    object_model = None
    if mode.use_object_model:
        ok, why = sf3d.available()
        if ok:
            object_model = _sf3d_runner(ctx, rgb, p["work"])
            ctx.models_used["object_reconstruction"] = "Stable Fast 3D (+ shape-prior completion fallback)"
        else:
            ctx.fallbacks.append(f"Stable Fast 3D unavailable ({why}); objects use shape-prior completion.")
    if object_model is None:
        ctx.models_used["object_reconstruction"] = "category shape-prior completion"
    ctx.models_used["hidden_surface_texture"] = "patch-based synthesis from visible material"

    tracker: dict[str, Any] = {"key": None, "t": 0.0}

    def progress(key: str, msg: str) -> None:
        ctx.check()
        if key != tracker["key"]:
            if tracker["key"]:
                _end(ctx, tracker["key"], tracker["t"])
            tracker["key"], tracker["t"] = key, _start(ctx, key, msg)
        else:
            ctx.emit(key, 0.5, msg)
        ctx.log_line("info", msg)

    result = composer.compose(rgb, depth, cam, mode.grid_step, things, stuff, object_model, progress)
    if tracker["key"]:
        _end(ctx, tracker["key"], tracker["t"])
    for key in COMPOSE_STAGES:
        ctx.timings.setdefault(key, 0.0)
    for warn in result.warnings:
        ctx.log_line("warn", warn)
    comp_report = result.completeness
    ctx.log_line("info" if comp_report.get("passed") else "warn",
                 f"Completeness: {comp_report.get('objects_complete', 0)}/{comp_report.get('objects_total', 0)} "
                 f"objects closed, background {'closed' if comp_report.get('background_closed') else 'OPEN'}")
    for issue in comp_report.get("issues", []):
        ctx.log_line("warn", issue)
    result.fill_image.save(p["output"] / "inpainted.png")
    ctx.check()

    # -------- export --------
    t = _start(ctx, "finalize", "Writing GLB and glTF")
    glb = p["output"] / "scene.glb"
    exporter.export_glb(result.scene, glb)
    validation = exporter.validate_glb(glb, result.groups)
    if validation.get("all_closed") is False:
        bad = [k for k, v in validation["elements"].items() if not v["closed"] or v["missing_nodes"]]
        ctx.log_line("warn", f"After export, these elements are no longer closed: {', '.join(bad)}")
    else:
        ctx.log_line("info", "Reloaded the exported GLB: every element is still a closed shell")
    exporter.export_gltf_zip(result.scene, p["output"] / "scene_gltf.zip")
    assets = [
        {"id": f"{ctx.project_id}.scene.glb", "name": "Scene (GLB)", "file": "scene.glb", "type": "model/gltf-binary"},
        {"id": f"{ctx.project_id}.scene_gltf.zip", "name": "Scene (glTF zip)", "file": "scene_gltf.zip", "type": "application/zip"},
        {"id": f"{ctx.project_id}.depth.png", "name": "Depth preview", "file": "depth.png", "type": "image/png"},
        {"id": f"{ctx.project_id}.depth16.png", "name": "Depth map (16-bit)", "file": "depth16.png", "type": "image/png"},
        {"id": f"{ctx.project_id}.working.png", "name": "Source image", "file": "working.png", "type": "image/png"},
        {"id": f"{ctx.project_id}.inpainted.png", "name": "Inferred background texture", "file": "inpainted.png", "type": "image/png"},
        {"id": f"{ctx.project_id}.completeness.json", "name": "Scene completeness report", "file": "completeness.json", "type": "application/json"},
    ]
    for name, obj_scene in result.object_meshes.items():
        fname = f"{name}.glb"
        exporter.export_glb(obj_scene, p["assets"] / fname)
        assets.append({"id": f"{ctx.project_id}.assets.{fname}", "name": name, "file": f"assets/{fname}",
                       "type": "model/gltf-binary", "object": name})

    ctx.timings["total"] = round(time.time() - t_all, 2)
    report = build_report(
        depth_stats=depth_stats, compose_stats=result.stats, things=things, stuff=stuff,
        scene=scene.to_dict(), timings=ctx.timings, peak_vram=gpu.peak_vram_gb(),
        device=gpu.select_device(), fallbacks=ctx.fallbacks, models=ctx.models_used,
        validation=validation, mock=settings.mock_models, completeness=comp_report)
    metadata = {
        "project_id": ctx.project_id,
        "version": 1,
        "scene": scene.to_dict(),
        "camera": cam.to_dict(),
        "fov_source": fov_source,
        "target": result.target,
        "lighting": result.lighting,
        "ground": result.ground,
        "nodes": result.nodes,
        "assets": assets,
        "coordinate_system": "glTF: +Y up, camera at origin looking -Z, units ≈ metres (relative)",
        "notes": [
            "Geometry comes from monocular relative depth; scale and distances are approximate.",
            "Nodes with status 'inferred' or 'generated' were not visible in the photo.",
        ],
    }
    (p["output"] / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (p["output"] / "report.json").write_text(json.dumps(report, indent=2))
    (p["output"] / "completeness.json").write_text(json.dumps(
        {**comp_report, "export_check": validation.get("elements", {})}, indent=2))
    _end(ctx, "finalize", t)
    return {"metadata": metadata, "report": report, "category": scene.category}


# ---------------------------------------------------------------- stage helpers

def _start(ctx: Context, key: str, msg: str) -> float:
    ctx.emit(key, 0.0, msg)
    ctx.log_line("stage", f"{dict((k, n) for k, n, _ in STAGES)[key]}: {msg}")
    return time.time()


def _end(ctx: Context, key: str, t0: float) -> None:
    ctx.timings[key] = round(ctx.timings.get(key, 0) + time.time() - t0, 2)
    ctx.emit(key, 1.0, "Done")


def _classify(ctx: Context, image: Image.Image, rgb: np.ndarray) -> classifier.SceneInfo:
    if settings.mock_models:
        ctx.models_used["scene"] = "heuristic (test mode)"
        return classifier.classify_heuristic(rgb)
    try:
        bundle = manager.acquire("scene")
        info = classifier.classify(bundle, image)
        ctx.models_used["scene"] = settings.clip_model_id
        return info
    except Exception as exc:  # noqa: BLE001
        ctx.fallbacks.append(f"Scene classifier unavailable ({exc}); used colour heuristic.")
        ctx.log_line("warn", "Scene classifier unavailable; using heuristic")
        ctx.models_used["scene"] = "heuristic"
        return classifier.classify_heuristic(rgb)
    finally:
        manager.unload("scene")


def _depth(ctx: Context, image: Image.Image) -> np.ndarray:
    if settings.mock_models:
        ctx.models_used["depth"] = "synthetic gradient (test mode)"
        return depth_mod.mock_disparity(image)
    bundle = manager.acquire("depth")   # depth is essential: errors propagate to the user
    try:
        disp = depth_mod.predict_disparity(bundle, image)
    except Exception as exc:
        if gpu.is_oom(exc):
            ctx.log_line("warn", "Out of GPU memory during depth; retrying at lower resolution")
            gpu.release_memory()
            small = image.copy()
            small.thumbnail((max(256, image.width // 2), max(256, image.height // 2)))
            disp = depth_mod.predict_disparity(bundle, small)
            disp = cv2.resize(disp, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
            ctx.fallbacks.append("Depth ran at half resolution after a GPU out-of-memory error.")
        else:
            raise
    ctx.models_used["depth"] = _describe(settings.depth_model_id, bundle.get("source"))
    return disp


def _detect(ctx: Context, image: Image.Image, scene: classifier.SceneInfo, mode) -> list[Detection]:
    if settings.mock_models:
        ctx.models_used["detection"] = "centre box (test mode)"
        w, h = image.size
        return [Detection("object", 0.5, [w * 0.3, h * 0.3, w * 0.7, h * 0.85], "thing")]
    if not mode.use_detection:
        return []
    # 1) Scene regions from SegFormer (fine-tuned if promoted). Falls back to detector prompts.
    regions: list[Detection] = []
    try:
        bundle = manager.acquire("semantic")
        regions = semantic.regions_from_labels(semantic.label_map(bundle, image), bundle["names"])
        ctx.models_used["regions"] = _describe(settings.semantic_model_id, bundle.get("source"))
        ctx.log_line("info", f"Regions: {', '.join(r.label for r in regions) or 'none found'}")
    except Exception as exc:  # noqa: BLE001
        ctx.fallbacks.append(f"Region segmentation unavailable ({exc}); regions come from detector prompts.")
        regions = []
    finally:
        manager.unload("semantic")
    stuff_prompts = [] if regions else scene.stuff_prompts
    # 2) Objects.
    try:
        bundle = manager.acquire("detector")
        if settings.detector_backend == "yolo":
            dets = yolo.detect(bundle, image)
            ctx.models_used["detection"] = _describe(settings.yolo_weights, bundle.get("source"))
        else:
            dets = detector.detect(bundle, image, scene.thing_prompts, stuff_prompts)
            ctx.models_used["detection"] = settings.detector_model_id
        return regions + dets
    except Exception as exc:  # noqa: BLE001
        ctx.fallbacks.append(f"Object detector unavailable ({exc}); scene built without object separation.")
        ctx.log_line("warn", "Object detector unavailable; continuing without object separation")
        ctx.models_used["detection"] = "none"
        return regions


def _describe(default: str, source: dict | None) -> str:
    if source and source.get("source") == "fine-tuned":
        value = source.get("value")
        return f"fine-tuned run {source.get('run_id')} ({source.get('metric')} {value:.4f})" if value is not None \
            else f"fine-tuned run {source.get('run_id')}"
    return default


def _segment(ctx: Context, image: Image.Image, rgb: np.ndarray, norm: np.ndarray,
             things: list[Detection], stuff: list[Detection], mode) -> None:
    things.sort(key=lambda d: -d.score)
    if things and mode.use_sam and not settings.mock_models:
        try:
            bundle = manager.acquire("sam")
            segmenter.segment_sam(bundle, image, things)
            ctx.models_used["segmentation"] = settings.sam_model_id
        except Exception as exc:  # noqa: BLE001
            ctx.fallbacks.append(f"SAM unavailable ({exc}); used GrabCut segmentation.")
            ctx.log_line("warn", "SAM unavailable; using GrabCut")
    for i, det in enumerate(things):
        ctx.check()
        if det.mask is None:
            segmenter.segment_grabcut(rgb, det, norm)
            ctx.models_used.setdefault("segmentation", "GrabCut (OpenCV)")
        ctx.emit("segmentation", (i + 1) / max(len(things) + len(stuff), 1), f"Segmented {det.label}")
    for det in stuff:
        if det.mask is None:                  # SegFormer regions already carry masks
            segmenter.segment_grabcut(rgb, det, norm)


def _foreground_fallback(ctx: Context, image: Image.Image, norm: np.ndarray, mode) -> list[Detection]:
    """No objects detected: separate near-layer components so they still get full 3D geometry."""
    props = foreground.propose(norm, max_objects=mode.max_objects)
    if not props:
        ctx.log_line("warn", "No objects detected and no clear foreground layer; the scene is one closed background")
        return []
    if mode.use_sam and not settings.mock_models:
        depth_masks = [d.mask.copy() for d in props]
        try:
            bundle = manager.acquire("sam")
            segmenter.segment_sam(bundle, image, props)
            for det, dm in zip(props, depth_masks):
                # Keep SAM only when it agrees with the depth layer; otherwise keep the depth mask.
                inter = (det.mask & dm).sum() if det.mask is not None else 0
                if det.mask is None or inter / max((det.mask | dm).sum(), 1) < 0.4:
                    det.mask, det.mask_method, det.mask_score = dm, "depth-foreground", None
        except Exception as exc:  # noqa: BLE001
            ctx.log_line("warn", f"SAM refinement of foreground proposals failed ({exc})")
            for det, dm in zip(props, depth_masks):
                det.mask, det.mask_method = dm, "depth-foreground"
        finally:
            manager.unload("sam")
    ctx.fallbacks.append(f"The detector found no objects; separated {len(props)} foreground "
                         "object(s) from the depth map instead.")
    ctx.log_line("warn", f"No objects detected; using {len(props)} depth-based foreground object(s)")
    return props


def _dedupe_masks(dets: list[Detection], thr: float = 0.7) -> list[Detection]:
    keep: list[Detection] = []
    for det in sorted(dets, key=lambda d: -d.score):
        a = det.mask
        if all((a & k.mask).sum() / max(min(a.sum(), k.mask.sum()), 1) < thr for k in keep):
            keep.append(det)
    return keep


def _sf3d_runner(ctx: Context, rgb: np.ndarray, work: Path):
    def run_one(det: Detection, observed_vertices: np.ndarray, index: int):
        ctx.check()
        manager.unload_all()          # SF3D needs ~6 GB: free everything first
        crop = sf3d.make_rgba_crop(rgb, det.mask)
        mesh = sf3d.reconstruct(crop, work / f"sf3d_{index:02d}")
        return sf3d.fit_to_observation(mesh, observed_vertices)
    return run_one
