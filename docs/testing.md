# Testing

## Running

```bat
cd backend
.venv\Scripts\python -m pytest -v                        :: CPU, test mode, no weights needed
set SPATIALAI_REAL=1
.venv\Scripts\python -m pytest tests\test_real_models.py -v   :: real models on your GPU
cd ..
python -m pytest training\tests -v                         :: dataset prep, metrics, real tiny-model training
cd frontend
npm test                                                  :: Vitest
npm run build                                             :: TypeScript type-check + production build
```

## What the suites cover

`test_images.py`: accepts JPEG; rejects empty files, non-images (including a PHP payload named .jpg), BMP and tiny images; applies EXIF rotation; flattens PNG alpha.

`test_geometry.py`: disparity-to-depth is monotonic and hits near/far; NaNs are repaired; the centre pixel unprojects onto the optical axis; flat planes keep every triangle while depth steps are cut; EXIF field of view; relief closure produces a watertight shell behind the visible surface; RANSAC recovers a known ground plane.

`test_composer.py`: GrabCut recovers a synthetic object (IoU > 0.8); composition produces observed, inferred and region nodes with the object in front of the inferred background; GLB and glTF export and reload with matching node names; NMS and IoU.

`test_api.py`: the full workflow (upload, generate, poll, scene, objects, GLB/glTF/asset downloads, project list, delete); WebSocket progress to completion; cancellation; invalid uploads, path traversal attempts, bad ids, generating without an image, invalid quality, cancelling when idle.

`test_gpu.py`: GPU info without CUDA, out-of-memory detection, and that the model manager evicts other models and reports gated-repo failures with login instructions.

`test_real_models.py`: Depth Anything V2 runs and ranks a near object closer than the wall; Grounding DINO runs.

## Verification status of this release

Verified in a Linux CPU environment: 27 backend tests passed (the 2 real-model tests were skipped), 17 training tests passed (they really train small stand-in models, resume them and promote them), 4 frontend tests passed, the TypeScript build passed, and the Unreal-style editor was exercised in headless Chromium against the backend in test mode (project browser, outliner selection, move gizmo, view modes, console).

Not verified, because that environment had no GPU and no access to Hugging Face or the datasets: full-size training on NYU, ADE20K and COCO; real inference with CLIP, Depth Anything V2, Grounding DINO and SAM; Stable Fast 3D integration; GPU memory behaviour on an RTX 4050; and the Windows `.bat` and `.ps1` scripts. Run `scripts\check_gpu.py`, `test_real_models.py` and one reconstruction of your own photo first, and report what you see.

## Depth accuracy against ground truth

Use `training.evaluation.evaluate_depth` (and the segmentation, detection and reconstruction evaluators) on prepared test splits; see TRAINING_GUIDE.md. The in-app quality report is different: it contains internal consistency measures, not accuracy.

`test_finetuned.py`: promoted checkpoints are preferred over pretrained weights (and can be switched off), SegFormer classes merge into SpatialAI regions, and the training registry endpoint.

`training/tests`: every dataset converter on synthetic data (NYU .mat-free paths, the Kaggle NYU 8/16-bit encodings, Hypersim ray-to-planar depth, COCO-to-YOLO including crowd, tiny and clipped boxes and class subsets, ADE20K label remapping, Pix3D filtering and model-level splits), the validator catching a corrupted file, augmentations keeping image and depth aligned, depth/segmentation/shape metrics against known answers, the scale/shift-invariant depth loss, depth and segmentation training end to end (segmentation must reach mIoU > 0.5), exact resume, and promotion rules including refusing a worse model, forced promotion, failed smoke tests and rollback.
