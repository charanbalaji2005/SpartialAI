"""PyTorch Datasets reading the CSV manifests produced by the prepare_* scripts."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import transforms as T
from .common import abs_path, load_depth_png, read_manifest


class DepthDataset(Dataset):
    def __init__(self, manifests: list[Path], split: str, size_hw: tuple[int, int], train: bool,
                 aug: dict | None = None, limit: int | None = None, seed: int = 0) -> None:
        self.rows = [r for m in manifests for r in read_manifest(m, split)]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No '{split}' rows in {manifests}")
        self.size_hw, self.train, self.aug, self.seed = tuple(size_hw), train, aug or {}, seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        img = Image.open(abs_path(r["image"])).convert("RGB")
        depth = load_depth_png(abs_path(r["depth"]))
        if self.train:
            rng = random.Random(np.random.randint(0, 2**31 - 1))
            x, d = T.depth_train(img, depth, self.size_hw, rng, tuple(self.aug.get("scale", (0.6, 1.0))),
                                 self.aug.get("flip", True), self.aug.get("color", 1.0))
        else:
            x, d = T.depth_eval(img, depth, self.size_hw)
        return {"pixel_values": torch.from_numpy(x), "depth": torch.from_numpy(d),
                "valid": torch.from_numpy(d > 1e-3), "source": r["source"]}


class SegmentationDataset(Dataset):
    def __init__(self, manifest: Path, split: str, train: bool, crop: int = 512, aug: dict | None = None,
                 limit: int | None = None, ignore_index: int = 255) -> None:
        self.rows = read_manifest(manifest, split)
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No '{split}' rows in {manifest}")
        self.train, self.crop, self.aug, self.ignore = train, crop, aug or {}, ignore_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        img = Image.open(abs_path(r["image"])).convert("RGB")
        mask = np.asarray(Image.open(abs_path(r["mask"])))
        if self.train:
            rng = random.Random(np.random.randint(0, 2**31 - 1))
            x, m = T.seg_train(img, mask, self.crop, rng, self.ignore, tuple(self.aug.get("scale", (0.5, 2.0))),
                               self.aug.get("flip", True), self.aug.get("color", 1.0))
        else:
            x, m = T.seg_eval(img, mask, self.crop)
        return {"pixel_values": torch.from_numpy(x), "labels": torch.from_numpy(m)}
