import time
import zipfile
import io

import trimesh
from fastapi.testclient import TestClient

from app.main import app


def _wait(client, pid, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/reconstruction/{pid}/status").json()
        if s["status"] in ("completed", "failed", "cancelled"):
            return s
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_full_workflow(jpeg_bytes):
    with TestClient(app) as client:
        assert client.get("/api/health").json()["test_mode"] is True
        assert "cuda_available" in client.get("/api/system/gpu").json()
        assert len(client.get("/api/models/status").json()["models"]) >= 5

        r = client.post("/api/reconstruction/upload",
                        files={"file": ("living room.jpg", jpeg_bytes, "image/jpeg")})
        assert r.status_code == 200, r.text
        project = r.json()
        pid = project["id"]
        assert project["name"] == "living room" and project["image_width"] == 320

        assert client.get(f"/api/projects/{pid}/image").status_code == 200
        r = client.post(f"/api/reconstruction/{pid}/generate", json={"quality": "fast"})
        assert r.status_code == 200
        s = _wait(client, pid)
        assert s["status"] == "completed", s
        assert all(st["state"] == "done" for st in s["stages"])
        assert s["progress"] == 1.0

        scene = client.get(f"/api/scenes/{pid}").json()
        assert scene["report"]["test_mode"] is True
        statuses = {n["status"] for n in scene["nodes"]}
        assert {"observed", "inferred"} <= statuses
        # Complete 360° geometry: every object and the background are closed shells, also after export.
        comp = scene["report"]["completeness"]
        assert comp["passed"] and comp["background_closed"], comp["issues"]
        assert comp["objects_complete"] == comp["objects_total"] >= 1
        assert scene["report"]["export"]["all_closed"]
        assert all(n["closed"] for n in scene["nodes"] if n["node"] != "ground_plane")
        assert any(n["node"] == "bg_shell" for n in scene["nodes"])
        assert any(n["node"].endswith("_back") for n in scene["nodes"])
        keys = [st["key"] for st in s["stages"]]
        assert keys.index("objects") < keys.index("repair") < keys.index("validation") < keys.index("finalize")
        objs = client.get(f"/api/scenes/{pid}/objects").json()
        assert all(o["kind"] == "object" for o in objs)

        glb = client.get(f"/api/export/{pid}/glb")
        assert glb.status_code == 200 and glb.content[:4] == b"glTF"
        loaded = trimesh.load(io.BytesIO(glb.content), file_type="glb", force="scene")
        assert len(loaded.geometry) == len(scene["nodes"])

        gltf = client.get(f"/api/export/{pid}/gltf")
        assert any(n.endswith(".gltf") for n in zipfile.ZipFile(io.BytesIO(gltf.content)).namelist())

        for a in scene["assets"]:
            assert client.get(f"/api/assets/{a['id']}").status_code == 200, a

        projects = client.get("/api/projects").json()
        assert any(p["id"] == pid and p["has_scene"] for p in projects)
        assert client.get(f"/api/projects/{pid}").json()["report"]["geometry"]["objects"] >= 0

        assert client.delete(f"/api/projects/{pid}").status_code == 200
        assert client.get(f"/api/projects/{pid}").status_code == 404


def test_websocket_progress(jpeg_bytes):
    with TestClient(app) as client:
        pid = client.post("/api/reconstruction/upload",
                          files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}).json()["id"]
        client.post(f"/api/reconstruction/{pid}/generate", json={"quality": "fast"})
        seen = []
        with client.websocket_connect(f"/api/reconstruction/{pid}/ws") as ws:
            while True:
                msg = ws.receive_json()
                seen.append(msg["status"])
                if msg["status"] in ("completed", "failed", "cancelled"):
                    break
        assert seen[-1] == "completed"


def test_cancel(jpeg_bytes):
    with TestClient(app) as client:
        pid = client.post("/api/reconstruction/upload",
                          files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}).json()["id"]
        client.post(f"/api/reconstruction/{pid}/generate", json={"quality": "fast"})
        client.post(f"/api/reconstruction/{pid}/cancel")
        s = _wait(client, pid)
        assert s["status"] in ("cancelled", "completed")   # may finish before cancel lands


def test_errors_and_security(jpeg_bytes):
    with TestClient(app) as client:
        r = client.post("/api/reconstruction/upload",
                        files={"file": ("evil.jpg", b"<?php system($_GET['x']); ?>", "image/jpeg")})
        assert r.status_code == 422 and "readable image" in r.json()["detail"]
        assert client.get("/api/projects/..%2F..%2Fetc%2Fpasswd").status_code == 404
        # When the built frontend is served, traversal through the SPA route must not read files.
        leak = client.get("/..%2F..%2Fbackend%2Fapp%2Fmain.py")
        assert "FastAPI" not in leak.text
        assert client.get("/api/does-not-exist").status_code == 404
        assert client.get("/api/projects/not-a-real-id").status_code == 404
        assert client.get("/api/assets/..%2F..%2Fspatialai.db").status_code == 404
        pid = client.post("/api/projects", json={"name": "x"}).json()["id"]
        assert client.post(f"/api/reconstruction/{pid}/generate").status_code == 400
        assert client.get(f"/api/scenes/{pid}").status_code == 404
        assert client.post(f"/api/reconstruction/{pid}/generate",
                           json={"quality": "ultra"}).status_code == 422
        assert client.post(f"/api/reconstruction/{pid}/cancel").status_code == 409
