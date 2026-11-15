"""Fine-tune SegFormer-B0 for semantic segmentation (ADE20K, 150 classes).

SpatialAI uses the segmentation model to split a photo's background into named
regions (wall, floor, ceiling, sky, road, grass ...), so mIoU on those "stuff"
classes matters most. The trainer reports overall mIoU, pixel accuracy and a
SpatialAI-region mIoU.

Loss: cross-entropy (ignore 255) on logits upsampled to label resolution, with
optional OHEM-style hard-pixel weighting. Defaults suit 8 GB VRAM: 512 crops,
batch 2, grad accumulation 4, FP16.

  python -m training.trainers.train_segmentation --config segmentation.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..common.checkpoints import CheckpointManager, MonitorSpec
from ..common.config import CHECKPOINT_DIR, cli, dump, get, resolve
from ..common.logger import RunLogger
from ..common.seed import seed_everything, worker_init
from ..datasets.torch_datasets import SegmentationDataset
from ..evaluation.metrics import ConfusionMatrix
from .engine import Trainer, TrainSettings

# ADE20K classes SpatialAI turns into scene regions (names as in the model's id2label).
REGION_CLASSES = ["wall", "floor", "ceiling", "sky", "road", "grass", "sidewalk", "earth", "mountain",
                  "water", "sea", "field", "sand", "building", "tree", "rug", "path", "dirt track"]


class TinySegNet(torch.nn.Module):
    """Small CNN used only by the automated tests (`model.name: tiny-test`)."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, stride=4, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Conv2d(8, num_labels, 1)
        self.num_labels = num_labels

    def forward(self, pixel_values: torch.Tensor):
        return type("Out", (), {"logits": self.head(self.backbone(pixel_values))})()


def build_model(cfg: dict, num_labels: int, class_names: list[str]) -> torch.nn.Module:
    name = get(cfg, "model.name")
    if name == "tiny-test":
        return TinySegNet(num_labels)
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        name, num_labels=num_labels, id2label=dict(enumerate(class_names)),
        label2id={n: i for i, n in enumerate(class_names)}, ignore_mismatched_sizes=True)


def seg_loss(logits: torch.Tensor, labels: torch.Tensor, ignore: int = 255, ohem: float = 0.0) -> torch.Tensor:
    logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    if ohem <= 0:
        return F.cross_entropy(logits, labels, ignore_index=ignore)
    per_pixel = F.cross_entropy(logits, labels, ignore_index=ignore, reduction="none")[labels != ignore]
    k = max(1, int(per_pixel.numel() * ohem))
    return torch.topk(per_pixel, k).values.mean()


def export(model: torch.nn.Module, out: Path, cfg: dict, class_names: list[str]) -> None:
    info = {"task": "segmentation", "base_model": get(cfg, "model.name"), "num_labels": len(class_names),
            "class_names": class_names, "region_classes": REGION_CLASSES, "ignore_index": 255}
    if isinstance(model, TinySegNet):
        torch.save(model.state_dict(), out / "tiny.pt")
    else:
        model.save_pretrained(out, safe_serialization=True)
        from transformers import AutoImageProcessor
        try:
            AutoImageProcessor.from_pretrained(get(cfg, "model.name")).save_pretrained(out)
        except Exception:
            AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").save_pretrained(out)
    (out / "spatialai.json").write_text(json.dumps(info, indent=2))


def smoke_test(export_dir: Path) -> None:
    info = json.loads((export_dir / "spatialai.json").read_text())
    x = torch.randn(1, 3, 128, 128)
    if (export_dir / "tiny.pt").exists():
        m = TinySegNet(info["num_labels"])
        m.load_state_dict(torch.load(export_dir / "tiny.pt", map_location="cpu"))
    else:
        from transformers import SegformerForSemanticSegmentation
        m = SegformerForSemanticSegmentation.from_pretrained(export_dir)
    with torch.no_grad():
        logits = m.eval()(pixel_values=x).logits
    if logits.shape[1] != info["num_labels"] or not torch.isfinite(logits).all():
        raise RuntimeError("Exported segmentation model produced invalid logits")


def run(cfg: dict, resume: str | None = None, run_name: str | None = None) -> dict:
    seed_everything(get(cfg, "seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and get(cfg, "device", "auto") != "cpu" else "cpu")
    manifest = resolve(get(cfg, "data.manifest"))
    names_path = resolve(get(cfg, "data.class_names", str(manifest).replace(".csv", "_classes.json")))
    class_names = json.loads(Path(names_path).read_text()) if Path(names_path).exists() else \
        [f"class_{i}" for i in range(get(cfg, "data.num_classes", 150))]
    num_labels = len(class_names)
    crop = get(cfg, "data.crop", 512)
    train_ds = SegmentationDataset(manifest, "train", True, crop, get(cfg, "augment"), get(cfg, "data.limit_train"))
    val_ds = SegmentationDataset(manifest, "val", False, crop, limit=get(cfg, "data.limit_val"))
    bs, workers = get(cfg, "train.batch_size", 2), get(cfg, "data.workers", 2)
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=workers, drop_last=len(train_ds) > bs,
                              pin_memory=device.type == "cuda", worker_init_fn=worker_init,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=workers)

    model = build_model(cfg, num_labels, class_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=get(cfg, "train.lr", 6e-5),
                                  weight_decay=get(cfg, "train.weight_decay", 0.01))
    ckpt = CheckpointManager("segmentation", MonitorSpec("val/miou", "max", 1e-4), run_name, root=resolve(get(cfg, "checkpoint_root", CHECKPOINT_DIR)), resume=resume)
    dump(cfg, ckpt.run_dir / "config.yaml")
    log = RunLogger(ckpt.run_dir, get(cfg, "log.tensorboard", True))
    log.text(f"Segmentation fine-tuning on {device}: {len(train_ds)} train / {len(val_ds)} val, {num_labels} classes")
    ckpt.write_run(model_name=get(cfg, "model.name"), dataset_version=manifest.name,
                   train_samples=len(train_ds), val_samples=len(val_ds))
    region_ids = [i for i, n in enumerate(class_names) if n in REGION_CLASSES]

    def validate(m: torch.nn.Module) -> dict[str, float]:
        cm = ConfusionMatrix(num_labels)
        for b in val_loader:
            x = b["pixel_values"].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = m(pixel_values=x).logits
            logits = F.interpolate(logits.float(), size=b["labels"].shape[-2:], mode="bilinear", align_corners=False)
            cm.update(logits.argmax(1)[0].cpu().numpy(), b["labels"][0].numpy())
        res = cm.result(class_names)
        out = {"miou": res["miou"], "pixel_acc": res["pixel_acc"], "mean_acc": res["mean_acc"]}
        regions = [res["per_class_iou"][class_names[i]] for i in region_ids if class_names[i] in res["per_class_iou"]]
        if regions:
            out["region_miou"] = sum(regions) / len(regions)
        return out

    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader,
                      loss_fn=lambda m, b: seg_loss(m(pixel_values=b["pixel_values"]).logits, b["labels"],
                                                    ohem=get(cfg, "loss.ohem", 0.0)),
                      validate=validate, settings=TrainSettings.from_cfg(get(cfg, "train", {})), ckpt=ckpt,
                      logger=log, device=device, export_fn=lambda m, d: export(m, d, cfg, class_names))
    if resume:
        state = ckpt.load_for_resume()
        if state:
            trainer.load_state(state)
    result = trainer.fit()
    log.close()
    return {**result, "run_id": ckpt.run_id}


def main() -> None:
    args, cfg = cli(__doc__.splitlines()[0], "segmentation.yaml")
    result = run(cfg, args.resume, args.run_name)
    print(json.dumps({k: v for k, v in result.items() if k != "best"}, indent=2))


if __name__ == "__main__":
    main()
