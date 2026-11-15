"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except ImportError:
        pass


def worker_init(worker_id: int) -> None:
    """Seed each DataLoader worker so augmentations differ per worker but stay reproducible."""
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed((seed + worker_id) % 2**32)
    random.seed(seed + worker_id)
