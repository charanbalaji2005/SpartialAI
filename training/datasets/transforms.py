"""Paired augmentations that keep geometry and labels consistent.

Rules:
- Every geometric change (crop, scale, flip) is applied identically to the image and
  its target. Images resize with bilinear/Lanczos; depth and masks with NEAREST, so
  real depth measurements and class ids are never blended across object edges.
- Photometric changes (brightness, contrast, saturation, hue, blur) touch the image only.
- Depth values are not rescaled when the image is resized: depth is a property of the
  scene, not the pixel grid. (The depth loss is scale/shift-invariant anyway.)
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize(img: Image.Image) -> np.ndarray:
    """HWC uint8 -> CHW float32 with ImageNet statistics."""
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1).copy()


def photometric(img: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    if strength <= 0:
        return img
    ops = [
        lambda im: ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-0.25, 0.25) * strength),
        lambda im: ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-0.25, 0.25) * strength),
        lambda im: ImageEnhance.Color(im).enhance(1 + rng.uniform(-0.3, 0.3) * strength),
        lambda im: _hue(im, rng.uniform(-0.03, 0.03) * strength),
    ]
    rng.shuffle(ops)
    for op in ops:
        if rng.random() < 0.8:
            img = op(img)
    if rng.random() < 0.1 * strength:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    return img


def _hue(img: Image.Image, shift: float) -> Image.Image:
    hsv = np.asarray(img.convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(shift * 255)) % 256
    return Image.fromarray(hsv, "HSV").convert("RGB")


def _resize_target(t: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize for depth (float) or masks (uint8). size = (W, H)."""
    mode = "F" if t.dtype == np.float32 else None
    return np.array(Image.fromarray(t, mode=mode).resize(size, Image.NEAREST))


# --------------------------------------------------------------------------- depth
def depth_train(img: Image.Image, depth: np.ndarray, size_hw: tuple[int, int], rng: random.Random,
                scale: tuple[float, float] = (0.6, 1.0), flip: bool = True, color: float = 1.0
                ) -> tuple[np.ndarray, np.ndarray]:
    h_out, w_out = size_hw
    W, H = img.size
    aspect = w_out / h_out
    s = rng.uniform(*scale)
    cw = min(W, int(round(np.sqrt(s * W * H * aspect))))
    ch = min(H, int(round(cw / aspect)))
    cw = min(cw, int(round(ch * aspect)))
    x0, y0 = rng.randint(0, W - cw), rng.randint(0, H - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w_out, h_out), Image.BILINEAR)
    depth = _resize_target(depth[y0:y0 + ch, x0:x0 + cw].astype(np.float32), (w_out, h_out))
    if flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth[:, ::-1]
    img = photometric(img, rng, color)
    return normalize(img), np.ascontiguousarray(depth, np.float32)


def depth_eval(img: Image.Image, depth: np.ndarray, size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h_out, w_out = size_hw
    img = img.resize((w_out, h_out), Image.BILINEAR)
    depth = _resize_target(depth.astype(np.float32), (w_out, h_out))
    return normalize(img), np.ascontiguousarray(depth, np.float32)


# --------------------------------------------------------------------------- segmentation
def seg_train(img: Image.Image, mask: np.ndarray, crop: int, rng: random.Random, ignore: int = 255,
              scale: tuple[float, float] = (0.5, 2.0), flip: bool = True, color: float = 1.0
              ) -> tuple[np.ndarray, np.ndarray]:
    W, H = img.size
    s = rng.uniform(*scale) * crop / min(W, H)
    nw, nh = max(1, int(W * s)), max(1, int(H * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    mask = _resize_target(mask.astype(np.uint8), (nw, nh))
    # Pad with ignore so small images still yield a full crop.
    pw, ph = max(0, crop - nw), max(0, crop - nh)
    if pw or ph:
        canvas = Image.new("RGB", (nw + pw, nh + ph), (124, 116, 104))
        canvas.paste(img, (0, 0))
        img = canvas
        m = np.full((nh + ph, nw + pw), ignore, np.uint8)
        m[:nh, :nw] = mask
        mask = m
    W2, H2 = img.size
    x0, y0 = rng.randint(0, W2 - crop), rng.randint(0, H2 - crop)
    img = img.crop((x0, y0, x0 + crop, y0 + crop))
    mask = mask[y0:y0 + crop, x0:x0 + crop]
    if flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask[:, ::-1]
    img = photometric(img, rng, color)
    return normalize(img), np.ascontiguousarray(mask, np.int64)


def seg_eval(img: Image.Image, mask: np.ndarray, short_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize so the short side matches training scale; the full mask is kept for scoring."""
    W, H = img.size
    s = short_side / min(W, H)
    img = img.resize((max(32, int(round(W * s / 32)) * 32), max(32, int(round(H * s / 32)) * 32)), Image.BILINEAR)
    return normalize(img), np.ascontiguousarray(mask, np.int64)
