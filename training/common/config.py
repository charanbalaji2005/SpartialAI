"""YAML configs with dotted command-line overrides (`--set train.lr=1e-4`)."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]          # SpatialAI/
TRAINING_DIR = ROOT / "training"
DATA_DIR = TRAINING_DIR / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = TRAINING_DIR / "configs" / path
    cfg = yaml.safe_load(path.read_text()) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value: {item}")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    cfg["_config_path"] = str(path)
    return cfg


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def get(cfg: dict, key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve(path: str | Path) -> Path:
    """Relative paths in configs are relative to the SpatialAI root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def cli(description: str, default_config: str) -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.epochs=5 train.batch_size=1")
    parser.add_argument("--resume", default=None, help="run id to resume, or 'last' for the newest run")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    return args, load_config(args.config, args.set)


def dump(cfg: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False))
