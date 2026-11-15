"""Fine-tune Depth Anything V2 on prepared RGB-D data (NYU Depth V2, Hypersim, ...).

Depth Anything V2 (relative) predicts affine-invariant disparity, so we train with the
MiDaS-style objective: per image, solve the least-squares scale/shift that best maps the
prediction onto ground-truth disparity, then penalise the remaining error (L1, with the
largest 20 % residuals trimmed) plus a multi-scale gradient-matching term that keeps
edges sharp. This preserves the model's ability to generalise across scenes instead of
over-fitting it to NYU's metric range.

Memory plan for an 8 GB RTX 4050: Small model, 378x504 input, batch 2, grad accumulation
4 (effective batch 8), FP16 autocast, frozen DINOv2 backbone for the first epochs, and
optional gradient checkpointing.

  python -m training.trainers.train_depth --config depth.yaml
  python -m training.trainers.train_depth --config depth.yaml --set train.epochs=2 data.limit_train=200
  python -m training.trainers.train_depth --resume last
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..common.checkpoints import CheckpointManager, MonitorSpec
from ..common.config import CHECKPOINT_DIR, cli, dump, get, resolve
from ..common.logger import RunLogger
from ..common.seed import seed_everything, worker_init
from ..datasets.torch_datasets import DepthDataset
from ..evaluation.metrics import DepthMeter, evaluate_depth_pair
from .engine import Trainer, TrainSettings


# --------------------------------------------------------------------------- losses
def _ls_scale_shift(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image closed-form least squares for s, t in s*pred + t ≈ target (B,H,W)."""
    m = mask.float()
    a00 = (m * pred * pred).sum((1, 2))
    a01 = (m * pred).sum((1, 2))
    a11 = m.sum((1, 2))
    b0 = (m * pred * target).sum((1, 2))
    b1 = (m * target).sum((1, 2))
    det = a00 * a11 - a01 * a01
    ok = det.abs() > 1e-6
    s = torch.where(ok, (a11 * b0 - a01 * b1) / det.clamp(min=1e-6), torch.ones_like(det))
    t = torch.where(ok, (-a01 * b0 + a00 * b1) / det.clamp(min=1e-6), torch.zeros_like(det))
    return s, t


def gradient_loss(residual: torch.Tensor, mask: torch.Tensor, scales: int = 4) -> torch.Tensor:
    total = residual.new_zeros(())
    for k in range(scales):
        step = 2 ** k
        r, m = residual[:, ::step, ::step], mask[:, ::step, ::step].float()
        gx = (r[:, :, 1:] - r[:, :, :-1]).abs() * m[:, :, 1:] * m[:, :, :-1]
        gy = (r[:, 1:, :] - r[:, :-1, :]).abs() * m[:, 1:, :] * m[:, :-1, :]
        total = total + (gx.sum() + gy.sum()) / m.sum().clamp(min=1)
    return total / scales


def ssi_loss(pred_disp: torch.Tensor, gt_depth: torch.Tensor, valid: torch.Tensor,
             trim: float = 0.2, grad_weight: float = 0.5) -> torch.Tensor:
    pred = pred_disp.float()
    target = torch.zeros_like(gt_depth)
    target[valid] = 1.0 / gt_depth[valid]
    # Per-image median normalisation keeps the target scale comparable across datasets.
    med = torch.stack([target[i][valid[i]].median() if valid[i].any() else target.new_ones(())
                       for i in range(target.shape[0])])
    target = target / med.clamp(min=1e-6)[:, None, None]
    s, t = _ls_scale_shift(pred, target, valid)
    aligned = s[:, None, None] * pred + t[:, None, None]
    residual = (aligned - target) * valid
    err = residual.abs()[valid]
    if trim > 0 and err.numel() > 10:
        k = int(err.numel() * (1 - trim))
        err = torch.topk(err, k, largest=False).values
    return err.mean() + grad_weight * gradient_loss(residual, valid)


# --------------------------------------------------------------------------- model
class TinyDepthNet(torch.nn.Module):
    """Small CNN used only by the automated tests (`model.name: tiny-test`)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, pixel_values: torch.Tensor):
        out = F.relu(self.head(self.backbone(pixel_values))).squeeze(1)
        return type("Out", (), {"predicted_depth": out})()


def build_model(cfg: dict) -> torch.nn.Module:
    name = get(cfg, "model.name")
    if name == "tiny-test":
        return TinyDepthNet()
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(name)
    if get(cfg, "model.gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
        except Exception:  # not every backbone supports it
            print("Gradient checkpointing not supported by this model; continuing without it")
    return model


def param_groups(model: torch.nn.Module, cfg: dict) -> list[dict]:
    backbone, head = [], []
    for n, p in model.named_parameters():
        (backbone if n.startswith("backbone") else head).append(p)
    return [{"params": head, "lr": get(cfg, "train.lr_head", 5e-5), "name": "head"},
            {"params": backbone, "lr": get(cfg, "train.lr_backbone", 5e-6), "name": "backbone"}]


def export(model: torch.nn.Module, out: Path, cfg: dict) -> None:
    name = get(cfg, "model.name")
    info = {"task": "depth", "base_model": name, "input_size_hw": get(cfg, "data.input_size"),
            "output": "relative disparity (larger = nearer)"}
    if isinstance(model, TinyDepthNet):
        torch.save(model.state_dict(), out / "tiny.pt")
    else:
        model.save_pretrained(out, safe_serialization=True)
        from transformers import AutoImageProcessor
        AutoImageProcessor.from_pretrained(name).save_pretrained(out)
    (out / "spatialai.json").write_text(json.dumps(info, indent=2))


def smoke_test(export_dir: Path) -> None:
    """Load an exported model and run one forward pass (used before promotion)."""
    info = json.loads((export_dir / "spatialai.json").read_text())
    h, w = info.get("input_size_hw") or (378, 504)
    x = torch.randn(1, 3, h, w)
    if (export_dir / "tiny.pt").exists():
        m = TinyDepthNet()
        m.load_state_dict(torch.load(export_dir / "tiny.pt", map_location="cpu"))
    else:
        from transformers import AutoModelForDepthEstimation
        m = AutoModelForDepthEstimation.from_pretrained(export_dir)
    with torch.no_grad():
        out = m.eval()(pixel_values=x).predicted_depth
    if not torch.isfinite(out).all() or out.shape[-2:] != (h, w):
        raise RuntimeError(f"Exported depth model produced invalid output {tuple(out.shape)}")


# --------------------------------------------------------------------------- run
def run(cfg: dict, resume: str | None = None, run_name: str | None = None) -> dict:
    seed_everything(get(cfg, "seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and get(cfg, "device", "auto") != "cpu" else "cpu")
    manifests = [resolve(m) for m in get(cfg, "data.manifests")]
    size = tuple(get(cfg, "data.input_size", [378, 504]))
    if size[0] % 14 or size[1] % 14:
        raise ValueError("data.input_size must be multiples of 14 for the DINOv2 backbone")
    train_ds = DepthDataset(manifests, "train", size, True, get(cfg, "augment"), get(cfg, "data.limit_train"))
    val_ds = DepthDataset(manifests, "val", size, False, limit=get(cfg, "data.limit_val"))
    bs, workers = get(cfg, "train.batch_size", 2), get(cfg, "data.workers", 2)
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=workers, drop_last=len(train_ds) > bs,
                              pin_memory=device.type == "cuda", worker_init_fn=worker_init,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=workers)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=get(cfg, "train.weight_decay", 0.01))
    ckpt = CheckpointManager("depth", MonitorSpec("val/abs_rel", "min", 1e-4), run_name, root=resolve(get(cfg, "checkpoint_root", CHECKPOINT_DIR)), resume=resume)
    dump(cfg, ckpt.run_dir / "config.yaml")
    log = RunLogger(ckpt.run_dir, get(cfg, "log.tensorboard", True))
    log.text(f"Depth fine-tuning on {device}: {len(train_ds)} train / {len(val_ds)} val images, input {size}")
    ckpt.write_run(model_name=get(cfg, "model.name"), dataset_version=[m.name for m in manifests],
                   train_samples=len(train_ds), val_samples=len(val_ds))

    freeze_epochs = get(cfg, "train.freeze_backbone_epochs", 1)

    def on_epoch(epoch: int, m: torch.nn.Module, _opt) -> None:
        frozen = epoch < freeze_epochs
        for n, p in m.named_parameters():
            if n.startswith("backbone"):
                p.requires_grad = not frozen
        if epoch in (0, freeze_epochs):
            log.text("Backbone frozen" if frozen else "Backbone unfrozen")

    max_depth = get(cfg, "eval.max_depth", 10.0)

    def validate(m: torch.nn.Module) -> dict[str, float]:
        meter = DepthMeter()
        for b in val_loader:
            x = b["pixel_values"].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = m(pixel_values=x).predicted_depth.float()
            if pred.shape[-2:] != b["depth"].shape[-2:]:
                pred = F.interpolate(pred[:, None], size=b["depth"].shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            meter.update(evaluate_depth_pair(pred[0].cpu().numpy(), b["depth"][0].numpy(), max_depth=max_depth))
        return meter.result()

    def loss_fn(m: torch.nn.Module, b: dict) -> torch.Tensor:
        pred = m(pixel_values=b["pixel_values"]).predicted_depth
        return ssi_loss(pred, b["depth"], b["valid"], get(cfg, "loss.trim", 0.2), get(cfg, "loss.grad_weight", 0.5))

    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader, loss_fn=loss_fn,
                      validate=validate, settings=TrainSettings.from_cfg(get(cfg, "train", {})), ckpt=ckpt,
                      logger=log, device=device, export_fn=lambda m, d: export(m, d, cfg), on_epoch_start=on_epoch)
    if resume:
        state = ckpt.load_for_resume()
        if state:
            trainer.load_state(state)
    result = trainer.fit()
    log.close()
    return {**result, "run_id": ckpt.run_id}


def main() -> None:
    args, cfg = cli(__doc__.splitlines()[0], "depth.yaml")
    result = run(cfg, args.resume, args.run_name)
    print(json.dumps({k: v for k, v in result.items() if k != "best"}, indent=2))


if __name__ == "__main__":
    main()
