"""Image ingestion: content-based validation, EXIF orientation, resizing."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..core.config import settings

ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


class ImageError(ValueError):
    """Raised with a user-facing message."""


def validate_upload(data: bytes) -> tuple[str, int, int]:
    """Check real file contents (not the extension). Returns (ext, width, height)."""
    if not data:
        raise ImageError("The file is empty.")
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise ImageError(f"The file is larger than {settings.max_upload_mb} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageError("The file isn't a readable image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageError("Use a JPG, PNG or WEBP image.")
    if width * height > settings.max_input_pixels:
        raise ImageError("The image has too many pixels. Use one under 40 megapixels.")
    if min(width, height) < settings.min_input_side:
        raise ImageError(f"The image is too small. Use at least {settings.min_input_side}px per side.")
    return ALLOWED_FORMATS[fmt], width, height


def load_for_processing(data: bytes, max_side: int) -> tuple[Image.Image, Image.Image]:
    """Return (original_oriented_rgb, working_rgb) with EXIF rotation applied."""
    img = Image.open(io.BytesIO(data))
    exif_source = img.copy()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.info["exif_source"] = exif_source  # keep EXIF (focal length) reachable
    work = img.copy()
    work.thumbnail((max_side, max_side), Image.LANCZOS)
    # Even dimensions keep the depth model's patching happy.
    w, h = work.size
    work = work.resize((w - w % 2, h - h % 2), Image.LANCZOS)
    return img, work


def to_array(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
