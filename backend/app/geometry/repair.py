"""Mesh cleaning, completeness validation and automatic repair.

Every reconstructed element is a set of parts (e.g. an object's observed front
plus its inferred back and sides). Parts are kept separate so the viewer can
label what was seen and what was inferred, but *together* they must form a
closed shell. This module checks that, repairs what it can, and regenerates
what it cannot repair.

Checks per element: open boundary edges, non-manifold edges, winding
consistency, outward normals (positive volume), degenerate/duplicate faces,
connected components, finite normals, valid UVs, material/texture presence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from PIL import Image


# --------------------------------------------------------------------------- cleaning

def manifold_selection(faces: np.ndarray, sel: np.ndarray, n_vertices: int,
                       max_iter: int = 12) -> tuple[np.ndarray, int]:
    """Remove "bow-tie" vertices from a face selection.

    A bow-tie vertex joins two otherwise separate pieces of surface at a single
    point (common where depth-discontinuity cuts meet diagonally). Closing such
    a surface makes the wall edge at that vertex shared by four faces, which is
    non-manifold. On a manifold boundary every vertex has exactly one outgoing
    boundary edge; faces touching vertices with more are dropped until none
    remain. Returns the new selection and the number of faces removed.
    """
    from .completion import boundary_edges
    sel = sel.copy()
    removed = 0
    for _ in range(max_iter):
        idx = np.nonzero(sel)[0]
        f = faces[idx]
        be = boundary_edges(f)
        if not len(be):
            break
        bad = np.bincount(be[:, 0], minlength=n_vertices) > 1
        if not bad.any():
            break
        touch = bad[f].any(1)
        sel[idx[touch]] = False
        removed += int(touch.sum())
    return sel, removed


# --------------------------------------------------------------------------- analysis

def merged_shell(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Concatenate parts and weld coincident vertices (ignoring UV seams)."""
    parts = [p for p in parts if p is not None and len(p.faces)]
    if not parts:
        return trimesh.Trimesh()
    verts, faces, off = [], [], 0
    for p in parts:
        verts.append(np.asarray(p.vertices, np.float64))
        faces.append(np.asarray(p.faces) + off)
        off += len(p.vertices)
    m = trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces), process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    return m


@dataclass
class ShellReport:
    faces: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    watertight: bool = False
    winding_consistent: bool = False
    outward_normals: bool = False
    degenerate_faces: int = 0
    components: int = 0
    normals_finite: bool = True
    uv_valid: bool = True
    textured: bool = True
    volume: float = 0.0
    issues: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return (self.watertight and self.winding_consistent and self.outward_normals
                and self.normals_finite and self.uv_valid and self.textured)

    def to_dict(self) -> dict[str, Any]:
        return {"faces": self.faces, "boundary_edges": self.boundary_edges,
                "non_manifold_edges": self.non_manifold_edges, "watertight": self.watertight,
                "winding_consistent": self.winding_consistent, "outward_normals": self.outward_normals,
                "degenerate_faces": self.degenerate_faces, "components": self.components,
                "normals_finite": self.normals_finite, "uv_valid": self.uv_valid,
                "textured": self.textured, "volume": round(self.volume, 5),
                "complete": self.complete, "issues": list(self.issues)}


def analyze(parts: list[trimesh.Trimesh]) -> ShellReport:
    r = ShellReport()
    parts = [p for p in parts if p is not None and len(p.faces)]
    if not parts:
        r.issues.append("no geometry")
        r.textured = r.uv_valid = False
        return r
    for p in parts:
        vis = p.visual
        if isinstance(vis, trimesh.visual.TextureVisuals):
            uv = None if vis.uv is None else np.asarray(vis.uv)
            if uv is None or len(uv) != len(p.vertices) or not np.isfinite(uv).all() \
                    or uv.min() < -1e-4 or uv.max() > 1 + 1e-4:
                r.uv_valid = False
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is None:
                r.textured = False
        elif not isinstance(vis, trimesh.visual.ColorVisuals):
            r.textured = False
        if not np.isfinite(np.asarray(p.vertex_normals)).all():
            r.normals_finite = False
    if not r.uv_valid:
        r.issues.append("invalid UV coordinates")
    if not r.textured:
        r.issues.append("missing material or texture")
    if not r.normals_finite:
        r.issues.append("non-finite normals")

    m = merged_shell(parts)
    r.faces = int(len(m.faces))
    edges = np.sort(m.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    r.boundary_edges = int((counts == 1).sum())
    r.non_manifold_edges = int((counts > 2).sum())
    r.watertight = r.boundary_edges == 0 and r.non_manifold_edges == 0
    tri = m.triangles
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    r.degenerate_faces = int((area <= 1e-12).sum())
    r.components = int(len(trimesh.graph.connected_components(
        m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1)))
    r.winding_consistent = bool(m.is_winding_consistent) if r.watertight else False
    if r.watertight and r.winding_consistent:
        r.volume = float(m.volume)
        r.outward_normals = r.volume > 0
    if r.boundary_edges:
        r.issues.append(f"{r.boundary_edges} open boundary edges (holes)")
    if r.non_manifold_edges:
        r.issues.append(f"{r.non_manifold_edges} non-manifold edges")
    if r.watertight and not r.winding_consistent:
        r.issues.append("inconsistent face winding")
    if r.watertight and r.winding_consistent and not r.outward_normals:
        r.issues.append("normals point inward")
    return r


# --------------------------------------------------------------------------- repair

def project_px(verts: np.ndarray, cam) -> np.ndarray:
    z = np.maximum(-verts[:, 2], 1e-6)
    x = verts[:, 0] * cam.fx / z + cam.cx
    y = -verts[:, 1] * cam.fy / z + cam.cy
    return np.stack([np.clip(np.round(x), 0, cam.width - 1), np.clip(np.round(y), 0, cam.height - 1)],
                    1).astype(int)


def flip_if_inward(mesh: trimesh.Trimesh) -> bool:
    """For a closed, consistently wound shell, make sure normals face outward."""
    if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume < 0:
        mesh.invert()
        return True
    return False


def convex_closure(points: np.ndarray, cam, texture: Image.Image, box: tuple[int, int, int, int],
                   name: str) -> trimesh.Trimesh:
    """Regenerated closed shell: the convex hull of the observed and inferred points.

    Used only when a relief closure cannot be repaired. Always watertight and
    outward-facing; textured by projecting into the synthesized back texture.
    """
    hull = trimesh.convex.convex_hull(points)
    hull = trimesh.Trimesh(hull.vertices, hull.faces, process=False)
    flip_if_inward(hull)
    px = project_px(np.asarray(hull.vertices), cam)
    x0, y0, x1, y1 = box
    uv = np.stack([(px[:, 0] - x0 + 0.5) / max(x1 - x0, 1), 1 - (px[:, 1] - y0 + 0.5) / max(y1 - y0, 1)], 1)
    from trimesh.visual.material import PBRMaterial
    mat = PBRMaterial(name=name, baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0,
                      doubleSided=True)
    hull.visual = trimesh.visual.TextureVisuals(uv=np.clip(uv, 0, 1).astype(np.float32), material=mat)
    return hull


def repair_single(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Repair a standalone mesh (e.g. from an image-to-3D model): weld, fill holes, fix winding."""
    actions: list[str] = []
    m = mesh.copy()
    before = analyze([m])
    if before.complete:
        return m, actions
    if before.degenerate_faces:
        m.update_faces(m.nondegenerate_faces())
        actions.append(f"removed {before.degenerate_faces} degenerate faces")
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    if not m.is_watertight:
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if trimesh.repair.fill_holes(m):
            actions.append("filled holes")
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m)
    actions.append("recomputed winding and normals")
    return m, actions


# --------------------------------------------------------------------------- scene report

def aabb_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap volume as a fraction of the smaller box."""
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va, vb = float(np.prod(a[1] - a[0])), float(np.prod(b[1] - b[0]))
    return inter / max(min(va, vb), 1e-12)


def scene_completeness(elements: list[dict[str, Any]], far: float) -> dict[str, Any]:
    """Aggregate per-element reports into the scene completeness report.

    Each element: {"name", "label", "kind", "parts": [Trimesh], "report": ShellReport, ...}.
    """
    issues: list[str] = []
    objects = [e for e in elements if e["kind"] == "object"]
    complete_objects = [e for e in objects if e["report"].complete]
    bg = next((e for e in elements if e["kind"] == "background"), None)

    out_of_bounds = []
    for e in elements:
        for p in e["parts"]:
            if p is None or not len(p.vertices):
                continue
            v = np.asarray(p.vertices)
            if (v[:, 2] > 1e-3).any() or (-v[:, 2] > far * 4).any():
                out_of_bounds.append(e["name"])
                break

    intersections = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i]["bounds"], objects[j]["bounds"]
            ov = aabb_overlap(a, b)
            if ov > 0.5:
                intersections.append({"a": objects[i]["name"], "b": objects[j]["name"],
                                      "overlap": round(ov, 3)})

    for e in elements:
        if not e["report"].complete:
            issues.append(f"{e['name']}: " + ("; ".join(e["report"].issues) or "incomplete"))
    for name in out_of_bounds:
        issues.append(f"{name}: geometry outside the expected scene bounds")
    if bg is None:
        issues.append("background: no closed background geometry")

    return {
        "passed": not issues,
        "objects_total": len(objects),
        "objects_complete": len(complete_objects),
        "background_closed": bool(bg and bg["report"].complete),
        "elements": [{"name": e["name"], "label": e.get("label"), "kind": e["kind"],
                      "shape_prior": e.get("prior"), "profile": e.get("profile"),
                      "thickness": e.get("thickness"), "method": e.get("method"),
                      "texture_size": e.get("texture_size"), "repairs": e.get("repairs", []),
                      "bridged_faces": e.get("bridged_faces", 0),
                      "status": ("complete" if e["report"].complete and not e.get("repairs")
                                 else "repaired" if e["report"].complete else "incomplete"),
                      **e["report"].to_dict()} for e in elements],
        "possible_intersections": intersections,
        "out_of_bounds": out_of_bounds,
        "issues": issues,
        "note": ("Hidden surfaces are inferred from category shape priors and the visible "
                 "material; they are plausible, not measured."),
    }
