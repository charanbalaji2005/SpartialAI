"""Per-project storage with strict path containment."""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from ..core.config import settings

ID_RE = re.compile(r"^[a-f0-9]{32}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(pid: str) -> bool:
    return bool(ID_RE.match(pid))


def project_dir(pid: str) -> Path:
    if not valid_id(pid):
        raise ValueError("Invalid project id")
    return settings.storage_dir / "projects" / pid


def paths(pid: str) -> dict[str, Path]:
    root = project_dir(pid)
    return {
        "root": root,
        "input": root / "input",
        "work": root / "work",
        "output": root / "output",
        "assets": root / "output" / "assets",
        "masks": root / "output" / "masks",
    }


def ensure(pid: str) -> dict[str, Path]:
    p = paths(pid)
    for key in ("input", "work", "output", "assets", "masks"):
        p[key].mkdir(parents=True, exist_ok=True)
    return p


def safe_child(base: Path, name: str) -> Path:
    """Resolve `name` inside `base`, refusing anything that escapes it."""
    if not NAME_RE.match(name):
        raise ValueError("Invalid file name")
    target = (base / name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError("Path escapes project directory")
    return target


def clear_outputs(pid: str) -> None:
    p = paths(pid)
    for key in ("work", "output"):
        shutil.rmtree(p[key], ignore_errors=True)
    ensure(pid)


def remove(pid: str) -> None:
    shutil.rmtree(project_dir(pid), ignore_errors=True)


def sanitize_display_name(name: str) -> str:
    stem = Path(name or "photo").stem
    clean = re.sub(r"[^\w\s.-]", "", stem).strip()[:80]
    return clean or "Untitled scene"
