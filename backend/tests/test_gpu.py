from app.core import gpu
from app.models.manager import ModelManager


def test_gpu_info_without_cuda():
    info = gpu.gpu_info()
    assert set(["cuda_available", "device", "total_vram_gb"]) <= info.keys()
    assert gpu.select_device() in ("cpu", "cuda")


def test_oom_detection():
    assert gpu.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not gpu.is_oom(RuntimeError("something else"))


def test_manager_sequential_and_failures():
    m = ModelManager()
    loaded = []
    m.register("a", "x/a", "role", "MIT", lambda dev: loaded.append("a") or {"m": "a"})
    m.register("b", "x/b", "role", "MIT", lambda dev: loaded.append("b") or {"m": "b"})
    def broken(dev):
        raise OSError("You are trying to access a gated repo")
    m.register("c", "x/c", "role", "MIT", broken)
    m.acquire("a")
    m.acquire("b")   # evicts a
    states = {s["name"]: s["status"] for s in m.status()}
    assert states == {"a": "not_loaded", "b": "loaded", "c": "not_loaded"}
    try:
        m.acquire("c")
    except RuntimeError as exc:
        assert "huggingface-cli login" in str(exc)
    assert {s["name"]: s["status"] for s in m.status()}["c"] == "failed"
