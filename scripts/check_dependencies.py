"""Check every Python package SpatialAI needs.  Run:  python scripts\\check_dependencies.py"""
import importlib
import sys

REQUIRED = [("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("multipart", "python-multipart"),
            ("pydantic", "pydantic"), ("torch", "torch (CUDA build)"), ("transformers", "transformers"),
            ("huggingface_hub", "huggingface_hub"), ("PIL", "pillow"), ("numpy", "numpy"),
            ("cv2", "opencv-contrib-python"), ("scipy", "scipy"), ("trimesh", "trimesh")]

missing = []
for module, package in REQUIRED:
    try:
        mod = importlib.import_module(module)
        print(f"  ok       {package:<24} {getattr(mod, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  MISSING  {package:<24} ({exc.__class__.__name__})")
        missing.append(package)

try:
    import cv2
    cv2.ximgproc  # noqa: B018
    print("  ok       OpenCV contrib (edge-aware depth filtering)")
except AttributeError:
    print("  note     opencv-contrib not found; a bilateral filter will be used instead")

if missing:
    print(f"\nInstall the missing packages:  pip install {' '.join(p.split(' ')[0] for p in missing)}")
    sys.exit(1)
print("\nAll dependencies present.")
