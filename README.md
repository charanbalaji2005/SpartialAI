# SpatialAI

SpatialAI turns one photograph into an explorable, exportable 3D scene, running entirely on your own computer. Upload a photo of a room, street, landscape, vehicle or object; SpatialAI estimates depth, finds and separates objects, builds textured geometry, infers what the camera couldn't see, and opens the result in an interactive 3D editor.

Every surface in the output is labelled by where it came from: **seen in the photo**, **inferred** (filled in by inpainting or symmetry), or **AI-generated** (by an object reconstruction model). The viewer's "Seen vs inferred" mode colours the scene by that label, so a reconstruction never pretends to be a scan.

## What it does

The pipeline runs ten stages, each reported live in the UI: upload, image analysis (orientation and scene type), depth estimation, object detection, segmentation, geometry reconstruction, object generation, scene composition, textures and materials, and final export. The result is a single GLB scene with named parts (each background region, each object's visible surface, each object's inferred back, the inferred background behind objects, and an estimated ground plane), plus per-object GLB files, depth maps, masks, metadata and a quality report.

The editor is laid out like the Unreal Engine 5 editor. A menu bar (File, Edit, Window, Tools, Build, Select, Help) sits above a toolbar with Import, a green Generate button, Stop, a quality picker and Export. The viewport has Perspective, view-mode (Unlit, Lit, Clay, Seen vs inferred) and Show menus, a Scalability chip, and move, rotate and scale gizmos with snapping and world or local space. On the right, the Outliner lists every actor (SM_Wall, SM_Chair_01, SM_Chair_01_Back …) with visibility toggles and source and type columns, and the Details panel shows coloured X/Y/Z Location, Rotation and Scale fields, a mesh thumbnail cut from your photo, detection data and a Scene Settings tab. Along the bottom are the Content Drawer (Ctrl+Space), the Output Log and a console (`help`, `view truth`, `stat gpu`, `export glb` …). Scenes are opened from a Project Browser.

SpatialAI also includes a complete training pipeline (`training/`) for fine-tuning the depth, region-segmentation and detection models on public datasets with your RTX 4050, evaluating them, and promoting better models into the app. See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) and [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md).

## Requirements

Built for Windows 11 with an NVIDIA RTX 4050 (8 GB VRAM). Other NVIDIA cards with 6 GB or more work; CPU-only works but is slow (minutes per image).

You need Python 3.11 (64-bit), Node.js 20 LTS or newer, a current NVIDIA driver, about 6 GB of disk for models and dependencies, and 16 GB RAM (32 GB is more comfortable). Git and Visual Studio 2022 C++ build tools are only needed for the optional Stable Fast 3D setup.

## Install

```bat
install_windows.bat
```

This runs `setup_environment.ps1`, which checks Python, Node.js and the GPU, creates `backend\.venv`, installs CUDA-enabled PyTorch, installs backend and frontend dependencies, creates `.env`, verifies the GPU with a real FP16 computation, and downloads the model weights (about 2 GB) into `models\`.

PyTorch's CUDA build is the step most likely to need attention. The script uses the `cu128` wheels and falls back to `cu126`. If neither suits your driver, pick a command at https://pytorch.org/get-started/locally/ and run it inside `backend\.venv`, or pass a tag: `install_windows.bat -Cuda cu126`. Then confirm with:

```bat
backend\.venv\Scripts\python scripts\check_gpu.py
```

It should print `CUDA available: yes` and your GPU name.

## Run

```bat
start_spatialai.bat
```

This opens the backend (http://127.0.0.1:8000, API docs at `/docs`) and the frontend (http://localhost:5173) in two windows and launches your browser. `start_spatialai.bat prod` instead builds the frontend once and serves everything from port 8000.

## Using it

1. Drop a photo on the Scenes page (JPG, PNG or WEBP, up to 25 MB).
2. Pick a quality mode and choose **Generate 3D scene**.
3. Watch progress, GPU memory and the log. Cancel stops at the next step boundary.
4. Explore the scene. Press `V` to return to the photo's exact viewpoint, then orbit away to see depth.
5. Export from the bottom panel: the scene as GLB or glTF (zip), metadata JSON, single objects, depth maps, or your edited scene.

| Mode | Resolution | Objects | Segmentation | Object model |
|---|---|---|---|---|
| Fast | 512 px | up to 4 | GrabCut | relief backs |
| Balanced | 768 px | up to 8 | Segment Anything | relief backs |
| Best quality | 1024 px | up to 12 | Segment Anything | Stable Fast 3D if installed |

Keyboard (Unreal-style): `Q` select, `W` move, `E` rotate, `R` scale, `F` focus selection, `V` photo viewpoint, `H` hide selection, `G` grid, `O` orthographic, `1`–`4` view modes, `T` seen-vs-inferred, `Ctrl+Space` Content Drawer, `Ctrl+G` generate, `` ` `` console, `Esc` deselect. Help > Keyboard shortcuts lists them all.

## Models

| Role | Model | License |
|---|---|---|
| Depth | Depth Anything V2 Small | Apache-2.0 |
| Scene type | CLIP ViT-B/32 | MIT |
| Detection | Grounding DINO Tiny | Apache-2.0 |
| Segmentation | SAM ViT-Base | Apache-2.0 |
| Scene regions | SegFormer-B0 (ADE20K) | NVIDIA SegFormer license |
| Detection (optional) | YOLO11n | AGPL-3.0 |
| Object reconstruction (optional) | Stable Fast 3D | Stability AI Community License |

When you have fine-tuned and promoted a model, the app loads it from `checkpoints/production/` instead of the pretrained weights; the Models window shows which is in use. Only one model sits in GPU memory at a time; each is released before the next loads. If an optional model is missing or fails, SpatialAI uses a documented fallback and lists it in the quality report. Depth is the only model that must work. Setup details for Stable Fast 3D, licenses and citations are in [docs/models.md](docs/models.md).

## Testing

```bat
cd backend
.venv\Scripts\python -m pytest -v
set SPATIALAI_REAL=1 && .venv\Scripts\python -m pytest tests\test_real_models.py -v
cd ..
python -m pytest training\tests -v
cd frontend && npm test
```

The default backend suite runs on CPU in test mode, where neural networks are replaced by simple deterministic stand-ins, and covers image validation, depth conversion, unprojection, discontinuity cutting, relief closure, ground-plane fitting, segmentation, composition, GLB/glTF export and reload, the full API workflow, WebSocket progress, cancellation, model lifecycle and path-traversal protection. `test_real_models.py` runs the real depth and detection models on your GPU. See [docs/testing.md](docs/testing.md), including what has and hasn't been verified.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA available: NO` | You have the CPU build of PyTorch or an old driver. Update the NVIDIA driver, then reinstall PyTorch with the CUDA index URL. |
| "Not enough GPU memory" | Use Fast mode, a smaller photo, or close games, browsers with heavy tabs, and other GPU apps. Depth retries at half resolution automatically. |
| A model shows "Unavailable" | Run `scripts\download_models.py` while online. For gated models, accept the license on huggingface.co and run `huggingface-cli login`. |
| "Can't reach the SpatialAI backend" | The backend window closed or crashed. Check its window for the error and run `start_spatialai.bat` again. |
| Scene looks like a stretched sheet | Thin structures and depth jumps are hard for monocular depth. Try Balanced or Best mode; turn on "Seen vs inferred" to see what was guessed. |
| PowerShell blocks the script | Run `install_windows.bat` (it bypasses the policy for that one script only). |

## Known limitations

A single photo cannot show hidden surfaces. Object backs and sides are inferred from category shape priors (or generated by Stable Fast 3D) with textures synthesized from the visible material, the background is closed into a relief shell, and the background behind objects is inpainted. Every element is validated as a closed shell (see completeness.json); the hidden geometry is plausible, not measured. Depth is relative, so absolute scale comes from an assumed range per scene type (indoor 1–8 m, outdoor 2–60 m, object 1–3 m) and is approximate. Field of view is read from EXIF when the photo has it and otherwise assumed to be 60°. Reflective, transparent and very thin surfaces reconstruct poorly. Stable Fast 3D object placement uses approximate orientation, and its Windows support is experimental upstream.

## Project layout

```
backend/app/        FastAPI app and the reconstruction engine
  api/              REST + WebSocket routes
  core/             configuration, GPU detection and memory
  models/           sequential model lifecycle manager
  scene_understanding/, depth_estimation/, object_detection/, segmentation/
  geometry/         unprojection, triangulation, relief closure, ground plane
  object_reconstruction/  Stable Fast 3D integration
  scene_composition/      assembles observed + inferred parts
  export/, evaluation/, database/, services/ (pipeline, jobs, storage)
backend/tests/      pytest suite
frontend/src/       React + TypeScript editor (Unreal-style) and the Three.js viewer engine
scripts/            check_gpu, check_dependencies, download_models
training/           dataset download + preparation, trainers, evaluation, checkpoints
  configs/          depth.yaml, segmentation.yaml, detection.yaml, pipeline.yaml
  datasets/         download, prepare_depth/detection/segmentation/reconstruction, validate, transforms
  trainers/         engine (AMP, accumulation, resume), train_depth, train_segmentation, train_detection
  evaluation/       metrics, evaluate_depth/segmentation/detection/reconstruction
  promote.py, train_all.py, tests/
checkpoints/        training runs, registry.json and promoted production models
docs/               architecture, pipeline, API, models, testing, user guide
storage/            projects, outputs and SQLite database (created at runtime)
models/             downloaded model weights
```

Two deliberate choices differ from the original brief: styling uses plain CSS design tokens instead of Tailwind, and the viewer drives Three.js directly instead of through React Three Fiber, because the shading modes need precise control over materials.
