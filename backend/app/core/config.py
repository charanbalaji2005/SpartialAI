"""SpatialAI configuration.

All settings can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]          # SpatialAI/
BACKEND_DIR = ROOT_DIR / "backend"

# Configure Hugging Face cache and options across all processes
os.environ.setdefault("HF_HOME", str(ROOT_DIR / "models" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QualityMode:
    name: str
    max_side: int          # longest image side used for inference + geometry
    grid_step: int         # mesh vertex every N pixels
    max_objects: int       # objects extracted as separate meshes
    use_detection: bool
    use_sam: bool
    use_object_model: bool  # Stable Fast 3D / TripoSR when installed


QUALITY_MODES: dict[str, QualityMode] = {
    "fast": QualityMode("fast", 768, 2, 6, True, True, False),
    "balanced": QualityMode("balanced", 1024, 2, 10, True, True, False),
    "best": QualityMode("best", 1536, 1, 16, True, True, True),
}


@dataclass
class Settings:
    storage_dir: Path = field(default_factory=lambda: _env_path("SPATIALAI_STORAGE", ROOT_DIR / "storage"))
    model_cache_dir: Path = field(default_factory=lambda: _env_path("SPATIALAI_MODEL_CACHE", ROOT_DIR / "models"))
    database_path: Path = field(default_factory=lambda: _env_path("SPATIALAI_DB", ROOT_DIR / "storage" / "spatialai.db"))

    device_preference: str = os.getenv("SPATIALAI_DEVICE", "auto")          # auto | cuda | cpu
    use_fp16: bool = _env_bool("SPATIALAI_FP16", True)
    max_upload_mb: int = int(os.getenv("SPATIALAI_MAX_UPLOAD_MB", "25"))
    max_input_pixels: int = int(os.getenv("SPATIALAI_MAX_PIXELS", str(40_000_000)))
    min_input_side: int = 64

    # Models (Hugging Face ids)
    depth_model_id: str = os.getenv("SPATIALAI_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
    detector_model_id: str = os.getenv("SPATIALAI_DETECTOR_MODEL", "IDEA-Research/grounding-dino-tiny")
    sam_model_id: str = os.getenv("SPATIALAI_SAM_MODEL", "facebook/sam-vit-base")
    clip_model_id: str = os.getenv("SPATIALAI_CLIP_MODEL", "openai/clip-vit-base-patch32")
    semantic_model_id: str = os.getenv("SPATIALAI_SEMANTIC_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512")
    # Object detector: grounding_dino (open vocabulary) or yolo (closed set, fine-tunable on COCO)
    detector_backend: str = os.getenv("SPATIALAI_DETECTOR", "grounding_dino")
    yolo_weights: str = os.getenv("SPATIALAI_YOLO_WEIGHTS", "yolo11n.pt")

    # Fine-tuned models promoted by training/promote.py live in checkpoints/production/<task>/
    checkpoint_dir: Path = field(default_factory=lambda: _env_path("SPATIALAI_CHECKPOINTS", ROOT_DIR / "checkpoints"))
    use_finetuned: bool = _env_bool("SPATIALAI_USE_FINETUNED", True)

    # Optional external object reconstructor (Stable Fast 3D in its own venv)
    sf3d_python: str | None = os.getenv("SPATIALAI_SF3D_PYTHON") or None     # e.g. C:\SpatialAI\sf3d-env\Scripts\python.exe
    sf3d_repo: str | None = os.getenv("SPATIALAI_SF3D_REPO") or None         # e.g. C:\SpatialAI\stable-fast-3d
    sf3d_texture_resolution: int = int(os.getenv("SPATIALAI_SF3D_TEXTURE", "512"))
    sf3d_timeout_s: int = int(os.getenv("SPATIALAI_SF3D_TIMEOUT", "300"))

    # Test mode: replaces neural networks with deterministic stand-ins. Never enable for real use.
    mock_models: bool = _env_bool("SPATIALAI_MOCK_MODELS", False)

    allowed_origins: list[str] = field(default_factory=lambda: os.getenv(
        "SPATIALAI_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(","))

    def ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "projects").mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep Hugging Face downloads inside the project unless the user set HF_HOME.
        os.environ.setdefault("HF_HOME", str(self.model_cache_dir / "huggingface"))
        # Prevent hf_xet from throttling downloads on Windows.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


settings = Settings()


def model_source(task: str, default: str) -> tuple[str, dict]:
    """Where to load a model from: the promoted fine-tuned checkpoint if present, else `default`.

    Returns (path_or_hf_id, info) where info describes the source for the UI and reports.
    """
    import json
    prod = settings.checkpoint_dir / "production" / task
    manifest = prod / "manifest.json"
    if settings.use_finetuned and manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            return str(prod), {"source": "fine-tuned", "run_id": data.get("run_id"),
                               "metric": data.get("metric"), "value": data.get("value"),
                               "base_model": data.get("model_name"), "promoted_at": data.get("promoted_at")}
        except (OSError, ValueError):
            pass
    return default, {"source": "pretrained", "base_model": default}
