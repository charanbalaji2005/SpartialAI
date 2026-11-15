"""Shared dataset utilities: manifests, file validation, hashing and leakage-free splits."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..common.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Depth maps are stored as 16-bit PNG in millimetres (0 = invalid): lossless, compact,
# readable everywhere. 65.535 m max covers indoor and most outdoor data.
DEPTH_SCALE = 1000.0
DEPTH_MAX_M = 65.535
IGNORE_INDEX = 255


# --------------------------------------------------------------------------- manifests
def write_manifest(rows: list[dict], path: Path, fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    fields = fields or list(rows[0].keys())
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


def read_manifest(path: Path, split: str | None = None) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if split is None or r.get("split") == split]


def rel(path: Path, base: Path = DATA_DIR) -> str:
    """Manifest paths are stored relative to training/data with forward slashes (portable)."""
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def abs_path(stored: str, base: Path = DATA_DIR) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else base / p


# --------------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    total: int = 0
    kept: int = 0
    problems: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def reject(self, reason: str, item: str) -> None:
        self.problems[reason].append(item)

    def summary(self) -> dict:
        return {"total": self.total, "kept": self.kept, "rejected": self.total - self.kept,
                "problems": {k: {"count": len(v), "examples": v[:5]} for k, v in self.problems.items()}}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))


def check_image(path: Path, min_side: int = 32) -> tuple[bool, str, tuple[int, int] | None]:
    """Decode the image fully (catches truncated files, not just bad headers)."""
    if not path.exists():
        return False, "missing_file", None
    try:
        with Image.open(path) as im:
            im.load()
            size = im.size
    except (UnidentifiedImageError, OSError, SyntaxError):
        return False, "corrupt_image", None
    if min(size) < min_side:
        return False, "too_small", size
    return True, "ok", size


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


# --------------------------------------------------------------------------- splits
def group_split(groups: Iterable[str], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                seed: int = 0) -> dict[str, str]:
    """Assign whole groups (scenes, object models) to splits so nothing leaks across them."""
    unique = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = max(n_val, 1)
    out = {}
    for i, g in enumerate(unique):
        out[g] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def check_leakage(rows: list[dict], group_key: str) -> dict[str, list[str]]:
    """Groups that appear in more than one split (should be empty)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        seen[r[group_key]].add(r["split"])
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}


def split_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    return dict(counts)


# --------------------------------------------------------------------------- files
def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link when possible (no admin rights needed on NTFS, no extra disk), else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_depth_png(depth_m: np.ndarray, path: Path) -> None:
    d = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d[(d <= 0) | (d > DEPTH_MAX_M)] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(d * DEPTH_SCALE).astype(np.uint16)).save(path)


def load_depth_png(path: Path) -> np.ndarray:
    """Metres, float32; 0 = invalid."""
    return np.asarray(Image.open(path), dtype=np.float32) / DEPTH_SCALE
