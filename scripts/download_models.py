"""Download all model weights into the local cache so SpatialAI can run offline.

Run while online:  python scripts\\download_models.py
Weights go to ./models/huggingface (or HF_HOME / SPATIALAI_MODEL_CACHE).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings  # noqa: E402

settings.ensure_dirs()
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError  # noqa: E402

MODELS = [
    (settings.depth_model_id, "Depth Anything V2 (required)", "Apache-2.0 for Small; Base/Large are CC-BY-NC-4.0"),
    (settings.clip_model_id, "CLIP scene classifier", "MIT"),
    (settings.detector_model_id, "Grounding DINO detector", "Apache-2.0"),
    (settings.sam_model_id, "Segment Anything", "Apache-2.0"),
    (settings.semantic_model_id, "SegFormer scene regions", "NVIDIA SegFormer license (research/non-commercial)"),
]
PATTERNS = ["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.py"]

failed = 0
for repo, name, lic in MODELS:
    print(f"\n{name}\n  {repo}  [{lic}]")
    try:
        path = snapshot_download(repo, allow_patterns=PATTERNS)
        print(f"  ok -> {path}")
    except GatedRepoError:
        failed += 1
        print("  Needs access: accept the license on huggingface.co, then run `huggingface-cli login`.")
    except (HfHubHTTPError, OSError) as exc:
        failed += 1
        print(f"  FAILED: {exc}")

print("\nStable Fast 3D is optional and installed separately (see docs/models.md).")
print(f"\nCache: {os.environ.get('HF_HOME')}")
if failed:
    print(f"{failed} model(s) failed. SpatialAI falls back where it can; depth is required.")
    sys.exit(1)
print("All models ready. You can now run offline (set HF_HUB_OFFLINE=1 if you like).")
