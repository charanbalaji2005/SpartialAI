"""Scene understanding: classify the photo and choose a reconstruction strategy.

Uses CLIP zero-shot classification. Categories guide prompts and depth range;
unknown scenes still go through the general pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from PIL import Image

from ..core.config import settings

CATEGORIES: dict[str, str] = {
    "living_room": "a photo of a living room",
    "bedroom": "a photo of a bedroom",
    "kitchen": "a photo of a kitchen",
    "office": "a photo of an office interior",
    "indoor_room": "a photo of an indoor room",
    "building_exterior": "a photo of the outside of a building",
    "urban_street": "a photo of a city street",
    "road": "a photo of a road",
    "landscape": "a photo of an open landscape",
    "forest": "a photo of a forest",
    "garden": "a photo of a garden",
    "mountain": "a photo of mountains",
    "beach": "a photo of a beach",
    "vehicle": "a photo of a single vehicle",
    "furniture": "a product photo of a single piece of furniture",
    "product": "a product photo of a single object",
    "general_object": "a close-up photo of an object",
    "people": "a photo of people",
    "portrait": "a portrait photo of a person",
    "poster": "a movie poster or graphic artwork with people",
}

GROUPS = {
    "indoor": {"living_room", "bedroom", "kitchen", "office", "indoor_room"},
    "outdoor": {"building_exterior", "urban_street", "road", "landscape", "forest",
                "garden", "mountain", "beach"},
    "object": {"vehicle", "furniture", "product", "general_object"},
    "people": {"people", "portrait", "poster"},
}

# Region ("stuff") and object ("thing") prompts per group for the open-vocabulary detector.
STUFF = {
    "indoor": ["wall", "floor", "ceiling"],
    "outdoor": ["sky", "road", "ground", "grass"],
    "object": [],
    "people": [],
}
THINGS = {
    "indoor": ["sofa", "chair", "table", "bed", "lamp", "cabinet", "shelf", "door", "window",
               "plant", "television", "rug", "person", "painting", "hand"],
    "outdoor": ["building", "tree", "car", "person", "bicycle", "bench", "fence", "pole",
                "plant", "rock", "house"],
    "object": ["hand", "arm", "car", "chair", "sofa", "table", "bottle", "shoe", "bag", "toy", "plant",
               "item", "tool", "product", "device", "object"],
    "people": ["person", "hand", "arm", "body", "face", "dog", "cat", "chair", "table", "car", "bag", "plant"],
}
# Searched for in every scene: scene classification can be wrong (a poster of people read as
# a vehicle photo), and a missed person or animal leaves the scene without separable objects.
UNIVERSAL_THINGS = ["person", "hand", "arm", "animal", "car", "chair", "table", "plant"]
# Approximate depth range (scene units ~ metres) used to turn relative depth into geometry.
DEPTH_RANGE = {"indoor": (1.0, 8.0), "outdoor": (2.0, 60.0), "object": (1.0, 3.0),
               "people": (1.0, 6.0)}


@dataclass
class SceneInfo:
    category: str
    group: str
    confidence: float
    top3: list[dict[str, float]]
    method: str
    stuff_prompts: list[str]
    thing_prompts: list[str]
    near: float
    far: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load(device: str) -> dict[str, Any]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(settings.clip_model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(settings.clip_model_id)
    return {"model": model, "processor": processor, "device": device, "torch": torch}


def classify(bundle: dict[str, Any], image: Image.Image) -> SceneInfo:
    torch = bundle["torch"]
    keys = list(CATEGORIES)
    inputs = bundle["processor"](text=[CATEGORIES[k] for k in keys], images=image,
                                 return_tensors="pt", padding=True)
    inputs = {k: v.to(bundle["device"]) for k, v in inputs.items()}
    with torch.inference_mode():
        probs = bundle["model"](**inputs).logits_per_image.softmax(dim=-1)[0].float().cpu().numpy()
    order = np.argsort(-probs)
    best = keys[order[0]]
    return _info(best, float(probs[order[0]]),
                 [{"category": keys[i], "p": round(float(probs[i]), 3)} for i in order[:3]], "clip")


def classify_heuristic(rgb: np.ndarray) -> SceneInfo:
    """Fallback when CLIP is unavailable: bright, blue-ish top band suggests outdoors."""
    h = rgb.shape[0]
    top = rgb[: max(1, h // 5)].reshape(-1, 3).astype(np.float32)
    blue = float(np.mean((top[:, 2] > top[:, 0] + 10) & (top[:, 2] > 120)))
    bright = float(top.mean() / 255)
    if blue > 0.35 or bright > 0.75:
        return _info("landscape" if blue > 0.6 else "urban_street", 0.4, [], "heuristic")
    return _info("indoor_room", 0.35, [], "heuristic")


def _info(category: str, conf: float, top3: list, method: str) -> SceneInfo:
    group = next((g for g, members in GROUPS.items() if category in members), "indoor")
    # Low-confidence results use the general strategy but keep the best guess.
    near, far = DEPTH_RANGE[group]
    things = list(dict.fromkeys(THINGS[group] + UNIVERSAL_THINGS))
    return SceneInfo(category, group, round(conf, 3), top3, method,
                     STUFF[group], things, near, far)
