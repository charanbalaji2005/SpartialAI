# Training guide

SpatialAI is a hybrid system: it starts from pretrained models and lets you fine-tune the ones that benefit most from your data, all on the RTX 4050. Nothing is trained from scratch.

| Stage | Model | Fine-tuned on | Chosen by | Notes |
|---|---|---|---|---|
| Depth | Depth Anything V2 Small | NYU Depth V2, Hypersim, your RGB-D pairs | lowest val AbsRel | biggest effect on scene quality |
| Scene regions | SegFormer-B0 | ADE20K | highest val mIoU | walls, floors, sky, ground |
| Objects | YOLO11n (optional) | COCO (scene-relevant classes) | highest val mAP50-95 | alternative to Grounding DINO |
| 3D objects | Stable Fast 3D | not trained (inference only) | evaluated on Pix3D | too large to fine-tune on 8 GB |

## Setup

```bat
install_windows.bat -Training
```

This adds `requirements-training.txt` to the normal install. All commands below use `train.bat`, which runs Python modules inside the SpatialAI environment. Prepare data first; see DATASET_GUIDE.md.

## Check the whole setup in a few minutes

```bat
train.bat training.train_all --quick
```

This trains every model that has prepared data for 1 epoch on a small subset, without promoting anything. If it finishes, your GPU, drivers, data and checkpoints all work.

## Train

```bat
train.bat training.trainers.train_depth --config depth.yaml
train.bat training.trainers.train_segmentation --config segmentation.yaml
train.bat training.trainers.train_detection --config detection.yaml
```

Change settings without editing files: `--set train.epochs=4 train.batch_size=1 data.limit_train=1000`. Configs live in `training/configs/`. Watch progress live with `tensorboard --logdir checkpoints/runs`.

To continue after a crash, power cut or Ctrl+C:

```bat
train.bat training.trainers.train_depth --resume last
```

Resuming restores the model, optimizer, learning-rate schedule, mixed-precision scaler, random state and epoch. A recovery checkpoint is written every 500 optimizer steps, so you lose at most a few minutes.

## How each model is trained

**Depth.** Depth Anything V2 predicts relative depth whose overall scale and offset are unknown. The loss therefore first finds, for each image, the scale and shift that best match the ground truth (in inverse depth), then penalises what remains. The largest 20 % of errors are ignored, which absorbs sensor noise at edges and on glass, and a multi-scale gradient term keeps object boundaries sharp. The pretrained DINOv2 backbone is frozen for the first epoch so the new data first adapts the head, then the whole network trains with a 10× lower learning rate on the backbone than on the head. This keeps the model's broad knowledge while adapting it to your data.

**Segmentation.** Cross-entropy on 512-pixel crops with multi-scale augmentation, optionally focusing on the hardest pixels (`loss.ohem`). Besides overall mIoU the trainer reports `region_miou`, the score over the classes SpatialAI actually turns into scene regions.

**Detection.** Ultralytics handles YOLO's training loop (mosaic, colour and flip augmentation, EMA weights). SpatialAI adds 8 GB-safe defaults, records mAP in the registry and exports weights the backend can load. Set `SPATIALAI_DETECTOR=yolo` in `.env` to use YOLO in the app. Note that YOLO and its weights are AGPL-3.0.

## Settings for 8 GB VRAM

| Setting | Depth | Segmentation | Detection |
|---|---|---|---|
| Batch size | 2 | 2 | 4 |
| Gradient accumulation | 4 (effective 8) | 4 (effective 8) | n/a |
| Input size | 378×504 | 512 crops | 640 |
| Precision | FP16 autocast | FP16 autocast | AMP |

These are starting points, not tuned optima. If a batch runs out of memory, training skips it, clears the cache and logs a message. If that repeats, lower the batch size (and raise accumulation to keep the effective batch), reduce the input size, or set `model.gradient_checkpointing: true` for depth. Close games and browsers with heavy GPU use while training, since the laptop GPU's memory is shared with the display.

## Checkpoints and promotion

```
checkpoints/
  runs/<task>/<run_id>/      last.pt, best.pt, recovery.pt, export/, config.yaml, metrics.jsonl, train.log
  production/<task>/         what the app loads
  production_previous/<task>/ automatic rollback copy
  registry.json              every run and the current production model
```

A finished run never replaces the app's model by itself. Promote it explicitly:

```bat
train.bat training.promote depth
train.bat training.promote depth --run 20261113-101500
train.bat training.promote depth --rollback
train.bat training.promote --list
```

Promotion loads the exported model and runs it once (a smoke test). It refuses to promote when the monitored metric isn't better than production unless you add `--force`. The swap is atomic, so the app never sees a half-copied model. Restart the backend afterwards. The Models window (Settings > Models and training) then shows "Fine-tuned" with the run and metric. Set `SPATIALAI_USE_FINETUNED=0` in `.env` to go back to the pretrained models without deleting anything.

## Evaluate

Always compare against the pretrained baseline on the same test split:

```bat
train.bat training.evaluation.evaluate_depth --manifest training/data/manifests/depth_nyu_kaggle.csv --model pretrained
train.bat training.evaluation.evaluate_depth --manifest training/data/manifests/depth_nyu_kaggle.csv --model production
train.bat training.evaluation.evaluate_segmentation --manifest training/data/manifests/segmentation_ade20k.csv
train.bat training.evaluation.evaluate_detection --data training/data/processed/detection/coco_yolo_spatialai/data.yaml
train.bat training.evaluation.evaluate_reconstruction --method relief --limit 100
```

Depth evaluation runs the app's own inference code at full resolution, aligns scale and shift in inverse depth, applies the standard Eigen crop to NYU frames, and reports AbsRel, SqRel, RMSE, RMSE-log and δ1–δ3 averaged per image. Reconstruction evaluation compares meshes against Pix3D CAD models with Chamfer-L2 and F-score@0.05 after normalising size and searching orientation; report that protocol alongside the numbers. Results are saved in `training/logs/`.

## What has and hasn't been verified

The training code is tested automatically with small stand-in models on synthetic data: data preparation for every format, augmentation alignment, metrics against hand-computed values, real training that measurably learns, exact resume, promotion rules and rollback (17 tests). Full-size training on the real datasets has not been run yet, so there are no real accuracy numbers in this project. They will come from your first runs on the RTX 4050.
