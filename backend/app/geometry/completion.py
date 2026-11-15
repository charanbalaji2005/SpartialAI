"""Hidden-surface inference: complete, closed geometry from a single view.

A single photo shows only the camera-facing side of every surface. This module
turns each such surface into a closed solid by inferring what the camera could
not see:

* **Category-aware shape priors.** Each object label maps to a cross-section
  profile and a depth-to-width ratio (a chair is deep, a painting is thin, a
  person is rounded). Unknown labels use a neutral prior.
* **Back surface.** Back vertices lie on the same camera rays as the visible
  front. Their depth comes from the prior: ``round`` objects get an elliptical
  thickness that falls to a thin rim at the silhouette (like inflating the
  silhouette), ``box`` objects get a flat back plane, ``slab`` objects get a
  thin constant thickness. Front detail is smoothed away, so the back is a
  plausible continuation, never a copy of the front.
* **Side walls.** Every open boundary of the visible surface (the silhouette
  and any depth-discontinuity cut inside it) is stitched to the back, so the
  front, sides and back form one watertight, consistently wound shell.
* **Back texture synthesis.** The back gets its own texture: the object's
  low-frequency colour field (material colour and shading gradients) plus
  high-frequency detail re-assembled from randomly chosen patches of the
  visible material. This keeps the material looking right while removing
  recognisable front content (faces, text, logos) from the back.
* **Occlusion bridges.** Faces that span a depth discontinuity are not shown
  with the photo (that would smear it) but are kept as inferred surfaces with
  synthesized texture, so a surface has no internal holes or spikes.
* **Background closure.** The whole background becomes a relief shell: its back
  follows a smoothed copy of the front on the same camera rays, so the scene
  has no empty side from any viewing direction and cannot self-intersect.

Everything produced here is clearly labelled as *inferred* geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial


# --------------------------------------------------------------------------- shape priors

@dataclass(frozen=True)
class ShapePrior:
    name: str
    profile: str          # "round" | "box" | "slab"
    depth_ratio: float    # object depth / visible width
    min_ratio: float = 0.01   # thinnest allowed thickness as a fraction of distance to camera


_PRIORS: list[tuple[tuple[str, ...], ShapePrior]] = [
    (("person", "man", "woman", "child", "boy", "girl", "people", "face", "head", "hand", "arm", "body", "finger"),
     ShapePrior("person", "round", 0.4)),
    (("dog", "cat", "animal", "horse", "cow", "bird", "sheep", "toy", "teddy"),
     ShapePrior("animal", "round", 0.8)),
    (("tree", "plant", "bush", "flower", "vase", "pot"),
     ShapePrior("plant", "round", 0.9)),
    (("bottle", "cup", "mug", "can", "glass", "jar", "ball", "lamp"),
     ShapePrior("cylinder", "round", 1.0)),
    (("rock", "stone", "boulder"), ShapePrior("rock", "round", 0.8)),
    (("car", "truck", "bus", "vehicle", "van", "motorcycle", "bicycle", "boat", "train"),
     ShapePrior("vehicle", "round", 1.4)),
    (("sofa", "couch", "armchair"), ShapePrior("sofa", "box", 0.45)),
    (("chair", "stool", "bench"), ShapePrior("chair", "box", 0.9)),
    (("table", "desk", "counter"), ShapePrior("table", "box", 0.7)),
    (("bed",), ShapePrior("bed", "box", 1.3)),
    (("cabinet", "dresser", "wardrobe", "drawer", "refrigerator", "fridge", "oven", "box",
      "bag", "suitcase", "shoe"), ShapePrior("cabinet", "box", 0.6)),
    (("shelf", "bookcase", "bookshelf"), ShapePrior("shelf", "box", 0.35)),
    (("building", "house", "tower", "wall"), ShapePrior("building", "box", 1.0)),
    (("television", "tv", "monitor", "screen", "laptop"), ShapePrior("screen", "slab", 0.12)),
    (("painting", "picture", "poster", "frame", "door", "window", "mirror", "sign",
      "board", "rug", "carpet", "mat", "curtain"), ShapePrior("panel", "slab", 0.05)),
    (("fence", "pole", "post"), ShapePrior("pole", "slab", 0.15)),
]
DEFAULT_PRIOR = ShapePrior("generic", "round", 0.6)


def prior_for(label: str) -> ShapePrior:
    words = label.lower().replace("_", " ").split()
    for keys, prior in _PRIORS:
        if any(w in keys or w.rstrip("s") in keys for w in words):
            return prior
    return DEFAULT_PRIOR


# --------------------------------------------------------------------------- helpers

def _rays_scale(verts: np.ndarray, z_new: np.ndarray) -> np.ndarray:
    """Move each vertex along its camera ray so that its depth (-Z) becomes z_new."""
    z_old = np.maximum(-verts[:, 2], 1e-6)
    return verts * (z_new / z_old)[:, None]


def _masked_blur(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised Gaussian blur of `values` using only pixels inside `mask`."""
    m = mask.astype(np.float32)
    if values.ndim == 3:
        num = cv2.GaussianBlur(values.astype(np.float32) * m[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    else:
        num = cv2.GaussianBlur(values.astype(np.float32) * m, (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Directed boundary edges (a -> b, in the winding of the face that owns them)."""
    if len(faces) == 0:
        return np.zeros((0, 2), int)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    key = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return edges[counts[inverse.ravel()] == 1]


def _side_faces(boundary: np.ndarray, n: int) -> np.ndarray:
    """Quads joining each front boundary edge a->b to its back copy (a+n, b+n).

    The front face traverses a->b, so the side must traverse b->a for the
    shell to be consistently wound (normals outward).
    """
    if len(boundary) == 0:
        return np.zeros((0, 3), int)
    a, b = boundary[:, 0], boundary[:, 1]
    return np.concatenate([np.stack([b, a, a + n], 1), np.stack([b, a + n, b + n], 1)])


# --------------------------------------------------------------------------- texture synthesis

def synthesize_back_texture(rgb: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int],
                            seed: int = 0, max_side: int = 512, patch: int = 16
                            ) -> tuple[Image.Image, float]:
    """Texture for an unseen surface, built from the visible material inside `mask`.

    low-frequency colour field (masked blur)  +  shuffled high-frequency patches.
    Returns the image and the scale factor applied to the crop.
    """
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1].astype(bool)
    ch, cw = m.shape
    scale = min(1.0, max_side / max(ch, cw))
    if scale < 1.0:
        size = (max(2, int(round(cw * scale))), max(2, int(round(ch * scale))))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        ch, cw = m.shape
    if not m.any():
        m = np.ones_like(m)

    sigma = max(3.0, 0.12 * min(ch, cw))
    low = _masked_blur(crop, m, sigma)
    # Pixels far from any visible material: fall back to the material's mean colour.
    weight = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    mean = crop[m].mean(0)
    low = np.where(weight[..., None] > 1e-3, low, mean)

    detail_src = crop - cv2.GaussianBlur(crop, (0, 0), 2.5)
    out_detail = np.zeros_like(crop)
    p = max(4, min(patch, ch // 2, cw // 2))
    # Valid source positions: any top-left corner whose patch lies inside the material.
    # Sampling at arbitrary (not grid-aligned) positions breaks the phase of periodic and
    # location-specific detail, so front content (text, faces) is not reassembled.
    inside = cv2.erode(m.astype(np.uint8), np.ones((p, p), np.uint8), anchor=(0, 0),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)[: ch - p + 1, : cw - p + 1]
    sy_all, sx_all = np.nonzero(inside)
    if len(sy_all):
        rng = np.random.default_rng(seed)
        for yy in range(0, ch, p):
            for xx in range(0, cw, p):
                i = int(rng.integers(len(sy_all)))
                sy, sx = int(sy_all[i]), int(sx_all[i])
                hh, ww = min(p, ch - yy), min(p, cw - xx)
                out_detail[yy:yy + hh, xx:xx + ww] = detail_src[sy:sy + hh, sx:sx + ww]
        out_detail = cv2.GaussianBlur(out_detail, (3, 3), 0.8)   # soften patch seams
    tex = np.clip(low + 0.7 * out_detail, 0, 255).astype(np.uint8)
    return Image.fromarray(tex), scale


def _crop_uv(px: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    u = (px[:, 0] - x0 + 0.5) / max(x1 - x0, 1)
    v = 1.0 - (px[:, 1] - y0 + 0.5) / max(y1 - y0, 1)
    return np.clip(np.stack([u, v], 1), 0.0, 1.0).astype(np.float32)


def _material(image: Image.Image, name: str) -> PBRMaterial:
    return PBRMaterial(name=name, baseColorTexture=image, baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                       metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)


def _bbox(mask: np.ndarray, pad: int = 2) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


# --------------------------------------------------------------------------- objects

@dataclass
class Completion:
    mesh: trimesh.Trimesh      # back + side walls (front is kept as a separate, observed part)
    prior: ShapePrior
    thickness: float
    box: tuple[int, int, int, int]       # image region the back texture is synthesized from
    method: str
    texture_size: tuple[int, int] | None = None


def apply_texture(c: Completion, rgb: np.ndarray, mask: np.ndarray, name: str, seed: int = 0,
                  patch: int = 16) -> None:
    """Synthesize the unseen-surface texture and assign it (UVs are already set)."""
    tex, _ = synthesize_back_texture(rgb, mask, c.box, seed=seed, patch=patch)
    c.mesh.visual = trimesh.visual.TextureVisuals(uv=c.mesh.visual.uv, material=_material(tex, name))
    c.texture_size = tex.size


def _assemble(front: trimesh.Trimesh, back: np.ndarray, px: np.ndarray,
              box: tuple[int, int, int, int], occluded: np.ndarray | None = None) -> trimesh.Trimesh:
    """Inferred part of a closed shell around the full surface `front`.

    Faces: the occluded front faces (depth-discontinuity bridges the camera could
    not see, flagged by `occluded`), the back cap, and side walls along the
    boundary. Together with the observed front faces this is watertight.
    """
    v = np.asarray(front.vertices, np.float64)
    n = len(v)
    bridge = front.faces[occluded] if occluded is not None else np.zeros((0, 3), int)
    faces = np.concatenate([bridge, front.faces[:, ::-1] + n,
                            _side_faces(boundary_edges(front.faces), n)])
    verts = np.concatenate([v, back])
    used = np.unique(faces)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    uv = _crop_uv(np.concatenate([px, px]), box)
    mesh = trimesh.Trimesh(vertices=verts[used], faces=remap[faces], process=False)
    placeholder = Image.new("RGB", (2, 2), (128, 128, 128))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv[used], material=_material(placeholder, "untextured"))
    return mesh


def typical_width(verts: np.ndarray, px: np.ndarray, mask: np.ndarray) -> float:
    """Typical horizontal width of the object in metres: the median silhouette row width.

    Unlike the bounding-box width this ignores protrusions (outstretched arms,
    a car's mirrors, a lamp's arm), which would otherwise inflate the inferred depth.
    """
    rows = mask.sum(1)
    rows = rows[rows > 0]
    width_px = float(np.median(rows)) if len(rows) else 1.0
    span_px = max(float(np.ptp(px[:, 0])), 1.0)
    return width_px * float(np.ptp(verts[:, 0])) / span_px


def infer_back_depth(front_z: np.ndarray, px: np.ndarray, mask: np.ndarray, depth: np.ndarray,
                     prior: ShapePrior, width: float) -> tuple[np.ndarray, float]:
    """Depth (positive, metres) of the hidden back surface for each front vertex."""
    zmed = float(np.median(front_z))
    t_min = max(prior.min_ratio * zmed, 0.004)
    thickness = float(np.clip(prior.depth_ratio * width, 3 * t_min, 3.0 * zmed))
    xs, ys = px[:, 0], px[:, 1]

    if prior.profile == "slab":
        return front_z + max(thickness, t_min), thickness

    if prior.profile == "box":
        back_plane = float(np.percentile(front_z, 10)) + thickness
        return np.maximum(back_plane, front_z + t_min), thickness

    # round: smooth rounded cross-section from the silhouette's interior distance.
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(float(dist.max()), 1.0)
    s = np.clip(dist[ys, xs] / radius, 0.0, 1.0)
    profile = np.sin(0.5 * np.pi * s)          # 0 at the rim, 1 at the core (smooth finite slope)
    ys0, xs0 = np.nonzero(mask)
    obj_size = max(int(np.ptp(ys0)) if len(ys0) else 1, int(np.ptp(xs0)) if len(xs0) else 1, 1)
    smooth = _masked_blur(depth, mask, max(2.0, 0.08 * obj_size))[ys, xs]
    return np.maximum(smooth + thickness * profile, front_z + t_min), thickness


def close_object(front: trimesh.Trimesh, mask: np.ndarray, depth: np.ndarray, label: str,
                 rgb: np.ndarray | None = None, seed: int = 0,
                 occluded: np.ndarray | None = None,
                 behind: np.ndarray | None = None) -> Completion:
    """Inferred back, sides and occluded bridges for an object's surface.

    `front` is the object's full surface (observed faces plus, optionally, the
    depth-discontinuity faces flagged by `occluded`) and must carry per-vertex
    pixel coordinates in ``front.metadata["px"]`` (set by
    :func:`geometry.mesh.submesh`). With `rgb` the back is textured immediately;
    otherwise call :func:`apply_texture` later.
    """
    prior = prior_for(label)
    v = np.asarray(front.vertices, np.float64)
    px = np.asarray(front.metadata["px"])
    front_z = -v[:, 2]
    back_z, thickness = infer_back_depth(front_z, px, mask, depth, prior, typical_width(v, px, mask))
    if behind is not None:
        # An object cannot extend into the surface behind it (`behind`: depth of the
        # background hidden behind the object). Clamp, keeping a minimal thickness.
        t_min = max(prior.min_ratio * float(np.median(front_z)), 0.004)
        limit = behind[px[:, 1], px[:, 0]] * 0.99
        back_z = np.maximum(np.minimum(back_z, limit), front_z + t_min)
    box = _bbox(mask)
    mesh = _assemble(front, _rays_scale(v, back_z), px, box, occluded)
    c = Completion(mesh, prior, thickness, box,
                   f"{prior.profile} profile ({prior.name} prior), synthesized back texture")
    if rgb is not None:
        apply_texture(c, rgb, mask, f"back_{seed:02d}", seed)
    return c


# --------------------------------------------------------------------------- background

def close_background(front: trimesh.Trimesh, depth: np.ndarray, rgb: np.ndarray | None = None,
                     seed: int = 0, occluded: np.ndarray | None = None,
                     thickness_ratio: float = 0.12) -> Completion:
    """Close the whole background surface into a solid relief shell.

    `front` is the full background surface (observed faces plus the
    depth-discontinuity bridges flagged by `occluded`). The back follows a
    smoothed copy of the front at a thickness of `thickness_ratio` x the median
    distance, on the *same camera rays*. Because back and front share rays and
    the back is always deeper, the solid cannot fold or self-intersect, and the
    border walls stay short instead of flaring out towards a far plane.
    """
    v = np.asarray(front.vertices, np.float64)
    px = np.asarray(front.metadata["px"])
    front_z = -v[:, 2]
    zmed = float(np.median(front_z))
    thickness = thickness_ratio * zmed
    h, w = depth.shape
    smooth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 0.04 * max(h, w))[px[:, 1], px[:, 0]]
    back_z = np.maximum(smooth + thickness, front_z + 0.02 * zmed)
    box = (0, 0, w, h)
    mesh = _assemble(front, _rays_scale(v, back_z), px, box, occluded)
    c = Completion(mesh, ShapePrior("background", "box", thickness_ratio), thickness, box,
                   "relief shell (smoothed back on the same camera rays), occlusion bridges, "
                   "synthesized back texture")
    if rgb is not None:
        apply_texture(c, rgb, np.ones((h, w), bool), "background_back", seed, patch=24)
    return c
