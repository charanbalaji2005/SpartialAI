# Dataset guide

SpatialAI uses public datasets to fine-tune and evaluate the models behind each stage of the reconstruction pipeline. You don't need any of them to *use* the app; they matter only if you want to improve or measure the models.

| Dataset | Used for | Size to start with | License (check before use) |
|---|---|---|---|
| NYU Depth V2 | Depth fine-tuning and evaluation (indoor) | 2.8 GB labeled set, or the ~4 GB Kaggle export | Research use |
| Hypersim | Extra depth data (synthetic indoor) | a few scenes, ~1–5 GB | CC BY-SA 3.0 |
| ADE20K (SceneParsing) | Region segmentation (wall, floor, sky…) | 0.9 GB | Research use |
| COCO 2017 | Object detection (YOLO) | 1.3 GB val+annotations; 19 GB with train | CC BY 4.0 annotations, Flickr image terms |
| Pix3D | 3D reconstruction evaluation | 3.6 GB | Research use |
| Objaverse | 3D asset library and checks | your choice of categories | Per-object CC licenses |

Don't redistribute any of these datasets, and check each license before commercial use.

## Folder layout

```
training/data/
  raw/<dataset>/          downloads, untouched
  processed/<task>/...    converted images, depth maps, masks, YOLO labels
  manifests/*.csv         one row per sample, with its split
  manifests/*_report.json what was rejected and why
```

Raw data is never modified. Every preparation step writes to `processed/` and `manifests/`, so you can delete and regenerate those at any time.

## 1. Download

```bat
train.bat training.datasets.download list
train.bat training.datasets.download coco --parts val annotations
train.bat training.datasets.download ade20k
train.bat training.datasets.download nyu
train.bat training.datasets.download pix3d
train.bat training.datasets.download objaverse --categories chair table sofa lamp --per-category 20
train.bat training.datasets.download hypersim
```

Downloads resume after interruptions, and zip files are checked for unsafe paths before extraction. Kaggle mirrors work too: extract them under `training/data/raw/<dataset>/`. The NYU "nyu_data" export used by DenseDepth is the most common Kaggle version and is supported directly.

## 2. Prepare

| Task | Command |
|---|---|
| Depth (Kaggle NYU) | `train.bat training.datasets.prepare_depth --source nyu_kaggle --input training/data/raw/nyu_depth/nyu_data` |
| Depth (official NYU) | `train.bat training.datasets.prepare_depth --source nyu_mat --input training/data/raw/nyu/nyu_depth_v2_labeled.mat --splits-mat training/data/raw/nyu/splits.mat` |
| Depth (Hypersim) | `train.bat training.datasets.prepare_depth --source hypersim --input training/data/raw/hypersim` |
| Depth (your own) | `train.bat training.datasets.prepare_depth --source folder --input path\to\pairs` |
| Segmentation | `train.bat training.datasets.prepare_segmentation --input training/data/raw/ade20k/ADEChallengeData2016` |
| Detection | `train.bat training.datasets.prepare_detection --coco training/data/raw/coco --preset spatialai` |
| Pix3D | `train.bat training.datasets.prepare_reconstruction pix3d --input training/data/raw/pix3d` |
| Objaverse | `train.bat training.datasets.prepare_reconstruction objaverse --input training/data/raw/objaverse` |

Add `--max-images 500` (segmentation) or `--max-train 2000 --max-val 300` (detection) for a quick first experiment.

## 3. Validate

```bat
train.bat training.datasets.validate_dataset training/data/manifests/depth_nyu_kaggle.csv
train.bat training.datasets.validate_dataset training/data/manifests/segmentation_ade20k.csv --num-classes 150
```

The validator decodes every file and checks sizes, depth ranges, class ids, duplicate images (by content hash) and split leakage. It exits with code 1 if anything is wrong.

## What preprocessing does, and why

**Integrity.** Every image is fully decoded, not just opened, so truncated downloads are caught. Samples with missing partners, mismatched sizes, mostly-invalid depth (under 30 % valid pixels by default) or almost-unlabelled masks are rejected and listed in the report with examples.

**Depth units and storage.** All depth is converted to metres and stored as 16-bit PNG in millimetres, with 0 meaning "no measurement". That is lossless to 1 mm up to 65 m and opens in any image tool. The Kaggle NYU export stores training depth as 8-bit (0–255 for 0–10 m) and test depth as 16-bit millimetres; both are decoded correctly. Hypersim stores the distance along each camera ray, which is converted to planar depth with the camera's focal length so it matches the other datasets.

**Alignment.** Depth and masks are always resized with nearest-neighbour sampling. Bilinear resizing would invent depth values halfway between a chair and the wall behind it, and class ids that don't exist.

**Labels.** ADE20K's "0 = other" becomes 255 (ignored in the loss and metrics), and classes 1–150 become 0–149. COCO's non-contiguous category ids (1–90) become 0–79. Crowd boxes and boxes under 2 pixels are dropped, and boxes are clipped to the image. Images without kept objects become empty label files, so YOLO also learns from backgrounds.

**Leakage-free splits.** Frames of the same room look almost identical, so a random split would leak training data into the test set and inflate the scores. Depth splits are by scene, Pix3D splits by 3D model, and Objaverse by object. Official test sets (NYU's 654 frames, ADE20K's validation set) are kept as `test` and used only for final reports. A held-out part of the training data is used as `val` to choose checkpoints.

**Augmentation (during training).** Geometric changes (random crops, scale, horizontal flips) are applied identically to the image and its target. Colour changes (brightness, contrast, saturation, hue, occasional blur) touch only the image. Depth values are not changed when an image is resized, because depth is a property of the scene, not of the pixel grid.
