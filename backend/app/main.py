"""SpatialAI FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .core.config import ROOT_DIR, settings
from .database.db import Database
from .services.jobs import JobManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spatialai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    db = Database(settings.database_path)
    interrupted = db.mark_interrupted()
    if interrupted:
        log.warning("Marked %d interrupted job(s) as failed", interrupted)
    app.state.db = db
    app.state.jobs = JobManager(db)
    if settings.mock_models:
        log.warning("TEST MODE: neural networks are replaced by deterministic stand-ins")
    yield


app = FastAPI(title="SpatialAI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception):
    log.exception("Unhandled error")
    # Never leak internal paths or stack traces to the client.
    return JSONResponse(status_code=500, content={"detail": "Internal error. Check the backend log."})


# Serve the built frontend (production) when present.
DIST = ROOT_DIR / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="static-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        candidate = (DIST / path).resolve()
        if path and DIST.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(Path(DIST / "index.html"))
