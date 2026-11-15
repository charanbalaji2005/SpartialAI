# API

Base URL `http://127.0.0.1:8000`. Interactive docs at `/docs`. Errors return `{"detail": "<message for the user>"}`.

| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | Server status and whether test mode is on |
| GET | /api/system/gpu | GPU name, VRAM total/used/free, utilisation, driver, PyTorch version |
| GET | /api/models/status | Each model's role, id, license, status (`not_loaded`, `loaded`, `failed`…) and error, plus quality modes |
| GET | /api/training/registry | Training runs and the promoted (production) model per task: depth, segmentation, detection |
| POST | /api/projects | Create an empty project `{"name": "..."}` |
| GET | /api/projects | List projects, newest first |
| GET | /api/projects/{id} | One project including its quality report |
| DELETE | /api/projects/{id} | Delete a project and its files (409 while running) |
| GET | /api/projects/{id}/image | The uploaded photo |
| POST | /api/reconstruction/upload | Multipart `file` (+ optional `project_id` to replace a photo). Creates a project if none given. 422 for invalid images |
| POST | /api/reconstruction/{id}/generate | Start or queue `{"quality": "fast" \| "balanced" \| "best"}` |
| GET | /api/reconstruction/{id}/status | Status, stage, progress 0–1, per-stage states, recent log lines |
| WS | /api/reconstruction/{id}/ws | Pushes the same status object on every change; closes when finished |
| POST | /api/reconstruction/{id}/cancel | Cancel at the next stage boundary (409 if nothing is running) |
| GET | /api/scenes/{id} | Scene metadata: nodes, camera, lighting, ground, assets, report, `glb_url` |
| GET | /api/scenes/{id}/objects | Object nodes only |
| GET | /api/export/{id}/glb | Scene as GLB |
| GET | /api/export/{id}/gltf | Scene as glTF + bin + textures in a zip |
| GET | /api/export/{id}/metadata | metadata.json |
| GET | /api/assets/{asset_id} | Any asset listed in the scene metadata (depth maps, per-object GLBs, textures) |

Status values: `created`, `uploaded`, `queued`, `running`, `completed`, `failed`, `cancelled`.

Node status values in scene metadata: `observed` (seen in the photo), `inferred` (filled by inpainting, symmetry or plane fitting), `generated` (made by an object model).
