"""Evaluate a YOLO model (mAP50, mAP50-95, precision, recall) on a prepared data.yaml.

  python -m training.evaluation.evaluate_detection --weights checkpoints/production/detection/best.pt \
      --data training/data/processed/detection/coco_yolo_spatialai/data.yaml
  python -m training.evaluation.evaluate_detection --weights yolo11n.pt --data ...   # pretrained baseline
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="checkpoints/production/detection/best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights).val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device, plots=True)
    names = m.names
    per_class = {names[c]: round(float(v), 4) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
    print(json.dumps({"weights": a.weights, "map50": float(m.box.map50), "map50_95": float(m.box.map),
                      "precision": float(m.box.mp), "recall": float(m.box.mr), "per_class_map50_95": per_class}, indent=2))


if __name__ == "__main__":
    main()
