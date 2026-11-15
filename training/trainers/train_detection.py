"""Fine-tune a YOLO detector (Ultralytics) on the COCO data prepared by prepare_detection.

Ultralytics owns the inner loop (mosaic/HSV/flip augmentation, AMP, EMA, cosine LR,
its own last/best weights). This wrapper adds what SpatialAI needs around it:
8 GB-safe defaults, a run in checkpoints/runs/detection/, validation mAP50 and
mAP50-95 recorded in the registry, and an export folder the backend can load.

  pip install ultralytics
  python -m training.trainers.train_detection --config detection.yaml
  python -m training.trainers.train_detection --config detection.yaml --set train.epochs=1 train.fraction=0.05
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..common.checkpoints import CheckpointManager, MonitorSpec
from ..common.config import CHECKPOINT_DIR, cli, dump, get, resolve
from ..common.logger import RunLogger

ULTRALYTICS_KEYS = ["epochs", "imgsz", "batch", "workers", "patience", "lr0", "lrf", "optimizer",
                    "cos_lr", "close_mosaic", "fraction", "amp", "cache", "freeze", "seed", "mosaic",
                    "mixup", "degrees", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v"]


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed: pip install ultralytics") from exc


def smoke_test(export_dir: Path) -> None:
    YOLO = _require_ultralytics()
    import numpy as np
    model = YOLO(str(export_dir / "best.pt"))
    model.predict(np.zeros((320, 320, 3), np.uint8), verbose=False)


def run(cfg: dict, resume: str | None = None, run_name: str | None = None) -> dict:
    YOLO = _require_ultralytics()
    data_yaml = resolve(get(cfg, "data.yaml"))
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found. Run prepare_detection first.")
    ckpt = CheckpointManager("detection", MonitorSpec("val/map50_95", "max"), run_name, root=resolve(get(cfg, "checkpoint_root", CHECKPOINT_DIR)), resume=resume)
    dump(cfg, ckpt.run_dir / "config.yaml")
    log = RunLogger(ckpt.run_dir, use_tensorboard=False)
    train_args = {k: v for k, v in (get(cfg, "train", {}) or {}).items() if k in ULTRALYTICS_KEYS}
    device = get(cfg, "device", 0)
    weights = get(cfg, "model.name", "yolo11n.pt")
    ckpt.write_run(model_name=weights, dataset_version=str(data_yaml), status="running")

    if resume:
        last = ckpt.run_dir / "ultralytics" / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Nothing to resume: {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        log.text(f"Training {weights} on {data_yaml} with {train_args}")
        model = YOLO(weights)
        model.train(data=str(data_yaml), device=device, project=str(ckpt.run_dir), name="ultralytics",
                    exist_ok=True, plots=True, verbose=True, **train_args)

    best = ckpt.run_dir / "ultralytics" / "weights" / "best.pt"
    if not best.exists():
        ckpt.write_run(status="failed")
        raise RuntimeError("Ultralytics finished without best.pt")
    metrics_obj = YOLO(str(best)).val(data=str(data_yaml), device=device, imgsz=train_args.get("imgsz", 640),
                                      batch=train_args.get("batch", 4), plots=False, verbose=False)
    metrics = {"val/map50": float(metrics_obj.box.map50), "val/map50_95": float(metrics_obj.box.map),
               "val/precision": float(metrics_obj.box.mp), "val/recall": float(metrics_obj.box.mr)}
    export = ckpt.export_dir
    export.mkdir(exist_ok=True)
    shutil.copy2(best, export / "best.pt")
    names = YOLO(str(best)).names
    (export / "spatialai.json").write_text(json.dumps({"task": "detection", "base_model": weights,
                                                       "class_names": [names[i] for i in sorted(names)],
                                                       "imgsz": train_args.get("imgsz", 640)}, indent=2))
    ckpt.best_value = metrics["val/map50_95"]
    ckpt.write_run(status="completed", best_value=ckpt.best_value, metrics=metrics,
                   monitor={"metric": "val/map50_95", "mode": "max"},
                   export_dir=str(export.relative_to(ckpt.root)))
    log.log(0, metrics)
    log.text("Validation: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log.close()
    return {"run_id": ckpt.run_id, "metrics": metrics, "run_dir": str(ckpt.run_dir)}


def main() -> None:
    args, cfg = cli(__doc__.splitlines()[0], "detection.yaml")
    print(json.dumps(run(cfg, args.resume, args.run_name), indent=2))


if __name__ == "__main__":
    main()
