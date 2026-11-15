"""Depth-to-geometry: unprojection, edge-aware triangulation, textured sub-meshes
and ground-plane estimation. Hidden surfaces are inferred in geometry/completion.py.

Coordinate system (glTF): +Y up, camera at the origin looking down -Z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_deg: float

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy, "hfov_deg": round(self.hfov_deg, 2),
                "vfov_deg": round(self.vfov_deg, 2), "position": [0, 0, 0],
                "look_direction": [0, 0, -1]}


def make_camera(width: int, height: int, hfov_deg: float = 60.0) -> Camera:
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return Camera(width, height, f, f, (width - 1) / 2, (height - 1) / 2, hfov_deg)


def fov_from_exif(image: Image.Image, default: float = 60.0) -> tuple[float, str]:
    """Horizontal FOV from the 35 mm-equivalent focal length when the photo has it."""
    try:
        exif = image.getexif()
        f35 = exif.get_ifd(0x8769).get(0xA405) or exif.get(0xA405)  # FocalLengthIn35mmFilm
        if f35 and float(f35) > 5:
            return math.degrees(2 * math.atan(36.0 / (2 * float(f35)))), "exif"
    except Exception:
        pass
    return default, "assumed"


@dataclass
class Grid:
    verts: np.ndarray        # (N,3) float32
    uv: np.ndarray           # (N,2) float32
    px: np.ndarray           # (N,2) int  (x, y) pixel of each vertex
    faces: np.ndarray        # (F,3) int
    face_px: np.ndarray      # (F,2) int  (x, y) pixel at face centroid
    face_ok: np.ndarray      # (F,) bool  False where the triangle spans a depth discontinuity
    rows: int
    cols: int


def build_grid(depth: np.ndarray, cam: Camera, step: int, max_ratio: float = 1.06) -> Grid:
    h, w = depth.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)
    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)
    gx, gy = np.meshgrid(xs, ys)
    z = depth[gy, gx].astype(np.float32)
    X = (gx - cam.cx) * z / cam.fx
    Y = -(gy - cam.cy) * z / cam.fy
    verts = np.stack([X, Y, -z], axis=-1).reshape(-1, 3).astype(np.float32)
    uv = np.stack([gx / max(w - 1, 1), 1.0 - gy / max(h - 1, 1)], axis=-1).reshape(-1, 2)

    rows, cols = gx.shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # Counter-clockwise when seen from the camera (normals face the viewer).
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])

    zf = z.ravel()[faces]
    z_min = np.maximum(zf.min(1), 1e-6)
    z_max = zf.max(1)
    ratios = z_max / z_min
    diffs = z_max - zf.min(1)

    # Discontinuity: must have BOTH significant depth ratio jump AND absolute depth jump.
    # A continuous slanted or curved surface (like a palm, cheek, or tabletop) has small
    # absolute depth variations across step pixels, even if the ratio is moderately high.
    # A true depth edge (foreground object against distant background) jumps both in ratio AND absolute meters.
    ratio_thr = max(1.30, 1.08 ** step)
    med_z = float(np.median(z))
    diff_thr = max(0.25, 0.12 * med_z)

    face_cut = (ratios > ratio_thr) & (diffs > diff_thr)
    face_ok = ~face_cut

    px = np.stack([gx.ravel(), gy.ravel()], 1)
    face_px = px[faces].mean(1).round().astype(int)
    return Grid(verts, uv.astype(np.float32), px, faces, face_px, face_ok, rows, cols)


def material(image: Image.Image, name: str, tint: tuple[float, float, float, float] | None = None
             ) -> PBRMaterial:
    return PBRMaterial(name=name, baseColorTexture=image,
                       baseColorFactor=list(tint) if tint else [1.0, 1.0, 1.0, 1.0],
                       metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)


def submesh(grid: Grid, face_sel: np.ndarray, mat: PBRMaterial,
            verts: np.ndarray | None = None) -> trimesh.Trimesh | None:
    faces = grid.faces[face_sel]
    if len(faces) == 0:
        return None
    used, inverse = np.unique(faces, return_inverse=True)
    v = (grid.verts if verts is None else verts)[used]
    mesh = trimesh.Trimesh(vertices=v, faces=inverse.reshape(-1, 3), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=grid.uv[used], material=mat)
    mesh.metadata["px"] = grid.px[used]          # image pixel of each vertex (for completion)
    return mesh


def fit_ground_plane(points: np.ndarray, iters: int = 300, thr: float = 0.05,
                     seed: int = 0) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane n·p + d = 0 with an upward-ish normal. Returns (n, d, inlier_ratio)."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    scale = float(np.median(np.abs(points[:, 2]))) or 1.0
    tol = thr * scale
    best = None
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[1] < 0:
            n = -n
        if n[1] < 0.6:          # must be roughly horizontal (camera roughly level)
            continue
        d = -float(n @ p[0])
        inliers = np.abs(points @ n + d) < tol
        ratio = float(inliers.mean())
        if best is None or ratio > best[2]:
            best = (n, d, ratio)
    if best is None or best[2] < 0.25:
        return None
    return best


def ground_mesh(n: np.ndarray, d: float, center: np.ndarray, size: float, color: list[int]
                ) -> trimesh.Trimesh:
    # Orthonormal basis on the plane.
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    c = center - (n @ center + d) * n
    corners = np.array([c + s * size * u + t * size * w for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    mesh = trimesh.Trimesh(vertices=corners, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    rgba = np.array(list(color[:3]) + [255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile(rgba, (2, 1)))
    return mesh
