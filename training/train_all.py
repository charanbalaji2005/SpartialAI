"""Run the configured training steps in order, then promote models that improved.

  python -m training.train_all                       # uses configs/pipeline.yaml
  python -m training.train_all --only depth
  python -m training.train_all --quick               # 1 epoch on small subsets: checks the whole setup

Steps whose prepared data is missing are skipped with a clear message, so you can
prepare datasets one at a time. Only one model trains at a time (8 GB VRAM).
"""
from __future__ import annotations

import argparse
import json
import traceback

from .common.checkpoints import promote
from .common.config import get, load_config, resolve
from .promote import SMOKE

QUICK = {
    "depth": ["train.epochs=1", "data.limit_train=64", "data.limit_val=16", "train.warmup_steps=5"],
    "segmentation": ["train.epochs=1", "data.limit_train=64", "data.limit_val=16", "train.warmup_steps=5"],
    "detection": ["train.epochs=1", "train.fraction=0.02"],
}


def data_ready(name: str, cfg: dict) -> tuple[bool, str]:
    if name == "depth":
        missing = [m for m in get(cfg, "data.manifests", []) if not resolve(m).exists()]
        return (not missing, f"missing {missing}; run prepare_depth")
    if name == "segmentation":
        m = get(cfg, "data.manifest")
        return (resolve(m).exists(), f"missing {m}; run prepare_segmentation")
    if name == "detection":
        y = get(cfg, "data.yaml")
        return (resolve(y).exists(), f"missing {y}; run prepare_detection")
    return False, "unknown step"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", default="pipeline.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-promote", action="store_true")
    a = ap.parse_args()
    pipeline = load_config(a.pipeline)
    summary = {}
    for step in pipeline["steps"]:
        name = step["name"]
        if a.only and name not in a.only:
            continue
        cfg = load_config(step["config"], QUICK.get(name, []) if a.quick else [])
        ok, why = data_ready(name, cfg)
        if not ok:
            print(f"\n== {name}: skipped ({why})")
            summary[name] = {"skipped": why}
            continue
        print(f"\n== {name}: training")
        try:
            module = __import__(f"training.trainers.train_{name}", fromlist=["run"])
            result = module.run(cfg)
            summary[name] = {"run_id": result["run_id"], "best_value": result.get("best_value"),
                             "metrics": result.get("metrics")}
            if step.get("promote") and not a.no_promote and not a.quick:
                summary[name]["promotion"] = promote(name, result["run_id"], SMOKE[name])
        except SystemExit as exc:
            summary[name] = {"error": str(exc)}
        except Exception as exc:  # keep going with the next model
            traceback.print_exc()
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
