"""REST + WebSocket API."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import (APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core import gpu
from ..core.config import QUALITY_MODES, settings
from ..models.registry import manager
from ..object_reconstruction import sf3d
from ..services import images, storage

router = APIRouter(prefix="/api")


class ProjectCreate(BaseModel):
    name: str = Field("Untitled scene", max_length=120)


class GenerateRequest(BaseModel):
    quality: Literal["fast", "balanced", "best"] = "balanced"


def _db(request: Request):
    return request.app.state.db


def _jobs(request: Request):
    return request.app.state.jobs


def _project_or_404(request: Request, pid: str) -> dict:
    if not storage.valid_id(pid):
        raise HTTPException(404, "Project not found")
    project = _db(request).get(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


def _public(project: dict) -> dict:
    pid = project["id"]
    out = {k: v for k, v in project.items() if k != "report"}
    out["image_url"] = f"/api/projects/{pid}/image" if project.get("image_ext") else None
    out["has_scene"] = (storage.paths(pid)["output"] / "scene.glb").exists()
    return out


# ------------------------------------------------------------------ system
@router.get("/health")
def health():
    return {"status": "ok", "test_mode": settings.mock_models}


@router.get("/system/gpu")
def system_gpu():
    return gpu.gpu_info()


@router.get("/models/status")
def models_status():
    ok, why = sf3d.available()
    return {
        "models": manager.status() + [{
            "name": "sf3d", "model_id": "stabilityai/stable-fast-3d", "role": "Object reconstruction (Best mode)",
            "license": "Stability AI Community License", "status": "available" if ok else "disabled",
            "error": None if ok else why, "device": None, "load_seconds": None}],
        "quality_modes": {k: vars(v) for k, v in QUALITY_MODES.items()},
        "test_mode": settings.mock_models,
    }


# ------------------------------------------------------------------ training
@router.get("/training/registry")
def training_registry():
    """Training runs and promoted (production) models recorded by the training pipeline."""
    path = settings.checkpoint_dir / "registry.json"
    data = json.loads(path.read_text()) if path.exists() else {"version": 1, "tasks": {}}
    tasks = {}
    for task in ("depth", "segmentation", "detection"):
        t = data.get("tasks", {}).get(task, {})
        runs = [{"run_id": k, **{kk: vv for kk, vv in v.items() if kk != "metrics"},
                 "metrics": v.get("metrics", {})} for k, v in t.get("runs", {}).items()]
        runs.sort(key=lambda r: r.get("updated_at") or 0, reverse=True)
        tasks[task] = {"production": t.get("production"), "runs": runs[:20]}
    return {"checkpoint_dir_exists": settings.checkpoint_dir.exists(), "use_finetuned": settings.use_finetuned,
            "tasks": tasks}


# ------------------------------------------------------------------ projects
@router.post("/projects")
def create_project(body: ProjectCreate, request: Request):
    pid = storage.new_id()
    storage.ensure(pid)
    return _public(_db(request).create(pid, storage.sanitize_display_name(body.name)))


@router.get("/projects")
def list_projects(request: Request):
    return [_public(p) for p in _db(request).list()]


@router.get("/projects/{project_id}")
def get_project(project_id: str, request: Request):
    project = _project_or_404(request, project_id)
    data = _public(project)
    data["report"] = project.get("report")
    return data


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, request: Request):
    _project_or_404(request, project_id)
    job = _jobs(request).get(project_id)
    if job and job.status in ("queued", "running"):
        raise HTTPException(409, "Cancel the running reconstruction before deleting this project.")
    storage.remove(project_id)
    _db(request).delete(project_id)
    return {"deleted": project_id}


@router.get("/projects/{project_id}/image")
def project_image(project_id: str, request: Request):
    project = _project_or_404(request, project_id)
    path = storage.paths(project_id)["input"] / f"original.{project['image_ext']}"
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


# ------------------------------------------------------------------ reconstruction
@router.post("/reconstruction/upload")
async def upload(request: Request, file: UploadFile = File(...),
                 project_id: str | None = Form(None)):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        ext, width, height = images.validate_upload(data)
    except images.ImageError as exc:
        raise HTTPException(422, str(exc)) from exc

    db = _db(request)
    if project_id:
        project = _project_or_404(request, project_id)
        job = _jobs(request).get(project_id)
        if job and job.status in ("queued", "running"):
            raise HTTPException(409, "This project is reconstructing. Cancel it first.")
        pid = project["id"]
        storage.clear_outputs(pid)
        for old in storage.paths(pid)["input"].glob("original.*"):
            old.unlink()
    else:
        pid = storage.new_id()
        db.create(pid, storage.sanitize_display_name(file.filename or "photo"))
    p = storage.ensure(pid)
    (p["input"] / f"original.{ext}").write_bytes(data)
    db.update(pid, status="uploaded", stage="upload", progress=0, error=None, message="Image uploaded",
              original_filename=storage.sanitize_display_name(file.filename or "photo") + f".{ext}",
              image_ext=ext, image_width=width, image_height=height, report=None, scene_category=None)
    return _public(db.get(pid))


@router.post("/reconstruction/{project_id}/generate")
def generate(project_id: str, request: Request, body: GenerateRequest | None = None):
    project = _project_or_404(request, project_id)
    if not project.get("image_ext"):
        raise HTTPException(400, "Upload an image before generating.")
    job = _jobs(request).submit(project_id, (body or GenerateRequest()).quality)
    return job.snapshot()


@router.get("/reconstruction/{project_id}/status")
def status(project_id: str, request: Request):
    project = _project_or_404(request, project_id)
    job = _jobs(request).get(project_id)
    if job:
        return job.snapshot()
    return {"project_id": project_id, "status": project["status"], "stage": project.get("stage"),
            "progress": project.get("progress") or 0, "message": project.get("message"),
            "error": project.get("error"), "stages": [], "logs": [], "version": 0}


@router.post("/reconstruction/{project_id}/cancel")
def cancel(project_id: str, request: Request):
    _project_or_404(request, project_id)
    if not _jobs(request).cancel(project_id):
        raise HTTPException(409, "Nothing is running for this project.")
    return {"cancelling": project_id}


@router.websocket("/reconstruction/{project_id}/ws")
async def progress_ws(websocket: WebSocket, project_id: str):
    await websocket.accept()
    jobs = websocket.app.state.jobs
    last = -1
    try:
        while True:
            job = jobs.get(project_id)
            if job and job.version != last:
                last = job.version
                await websocket.send_text(json.dumps(job.snapshot()))
                if job.status in ("completed", "failed", "cancelled"):
                    break
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return
    await websocket.close()


# ------------------------------------------------------------------ scenes & export
def _metadata(project_id: str) -> dict:
    path = storage.paths(project_id)["output"] / "metadata.json"
    if not path.exists():
        raise HTTPException(404, "No scene has been generated for this project yet.")
    return json.loads(path.read_text())


@router.get("/scenes/{project_id}")
def scene(project_id: str, request: Request):
    _project_or_404(request, project_id)
    meta = _metadata(project_id)
    meta["glb_url"] = f"/api/export/{project_id}/glb"
    report = storage.paths(project_id)["output"] / "report.json"
    meta["report"] = json.loads(report.read_text()) if report.exists() else None
    return meta


@router.get("/scenes/{project_id}/objects")
def scene_objects(project_id: str, request: Request):
    _project_or_404(request, project_id)
    return [n for n in _metadata(project_id)["nodes"] if n["kind"] == "object"]


def _send(project_id: str, rel: str, filename: str, media: str) -> FileResponse:
    out = storage.paths(project_id)["output"]
    parts = rel.split("/")
    base = out / parts[0] if len(parts) == 2 else out
    path = storage.safe_child(base, parts[-1])
    if not path.exists():
        raise HTTPException(404, "File not found. Generate the scene first.")
    return FileResponse(path, media_type=media, filename=filename)


@router.get("/export/{project_id}/glb")
def export_glb(project_id: str, request: Request):
    project = _project_or_404(request, project_id)
    return _send(project_id, "scene.glb", f"{_fname(project)}.glb", "model/gltf-binary")


@router.get("/export/{project_id}/gltf")
def export_gltf(project_id: str, request: Request):
    project = _project_or_404(request, project_id)
    return _send(project_id, "scene_gltf.zip", f"{_fname(project)}_gltf.zip", "application/zip")


@router.get("/export/{project_id}/metadata")
def export_metadata(project_id: str, request: Request):
    project = _project_or_404(request, project_id)
    return _send(project_id, "metadata.json", f"{_fname(project)}_metadata.json", "application/json")


@router.get("/assets/{asset_id}")
def asset(asset_id: str, request: Request):
    pid = asset_id.split(".", 1)[0]
    _project_or_404(request, pid)
    match = next((a for a in _metadata(pid)["assets"] if a["id"] == asset_id), None)
    if not match:
        raise HTTPException(404, "Asset not found")
    return _send(pid, match["file"], match["file"].split("/")[-1], match["type"])


def _fname(project: dict) -> str:
    return storage.sanitize_display_name(project.get("name") or "scene").replace(" ", "_")
