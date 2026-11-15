"""Closed-set detection with a YOLO model (Ultralytics), e.g. one fine-tuned by
training/trainers/train_detection.py and promoted to checkpoints/production/detection/."""
from __future__ import annotations

from typing import Any

from PIL import Image

from ..core.config import model_source, settings
from .detector import Detection, nms


def load(device: str) -> dict[str, Any]:
    from ultralytics import YOLO
    path, info = model_source("detection", settings.yolo_weights)
    weights = f"{path}/best.pt" if info["source"] == "fine-tuned" else path
    model = YOLO(weights)
    return {"model": model, "device": 0 if device == "cuda" else "cpu", "source": info}


def detect(bundle: dict[str, Any], image: Image.Image, conf: float = 0.3) -> list[Detection]:
    res = bundle["model"].predict(image, conf=conf, device=bundle["device"], verbose=False)[0]
    names = res.names
    dets = [Detection(names[int(c)], float(s), [float(v) for v in b], "thing")
            for b, s, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]
    return nms(dets)
