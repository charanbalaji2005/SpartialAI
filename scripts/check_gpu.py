"""Verify the NVIDIA GPU, driver and CUDA-enabled PyTorch.  Run:  python scripts\\check_gpu.py"""
import shutil
import subprocess
import sys


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]} ({'64' if sys.maxsize > 2**32 else '32'}-bit)")
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        print(f"nvidia-smi: {out or 'no GPU reported'}")
    else:
        print("nvidia-smi: not found. Install or update the NVIDIA driver.")
        ok = False
    try:
        import torch
    except ImportError:
        print("PyTorch: NOT INSTALLED. Run setup_environment.ps1.")
        return 1
    print(f"PyTorch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA available: NO")
        if torch.version.cuda is None:
            print("  This is the CPU-only build. Reinstall with the CUDA index URL (see README).")
        else:
            print("  The driver may be too old for this CUDA build. Update the NVIDIA driver.")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA available: yes. GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM, "
          f"compute {props.major}.{props.minor}")
    a = torch.rand(2048, 2048, device="cuda", dtype=torch.float16)
    b = (a @ a).float().sum().item()
    print(f"FP16 matmul on GPU: ok ({b:.1f})")
    if props.total_memory / 1024**3 < 7.5:
        print("Note: under 8 GB VRAM. Use Fast or Balanced mode; Stable Fast 3D may not fit.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
