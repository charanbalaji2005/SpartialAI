"""Promote a validated training run to production (what the SpatialAI backend loads).

  python -m training.promote depth                 # best run of the task, if it beats production
  python -m training.promote depth --run 20261113-101500
  python -m training.promote segmentation --force  # replace even if the metric is not better
  python -m training.promote depth --rollback      # restore the previous production model
  python -m training.promote --list

Before swapping, the exported model is loaded and run once (smoke test). The previous
production model is kept in checkpoints/production_previous/<task>/ for rollback.
Restart the backend afterwards so it loads the new model.
"""
from __future__ import annotations

import argparse
import json

from .common.checkpoints import Registry, promote, rollback

SMOKE = {
    "depth": lambda d: __import__("training.trainers.train_depth", fromlist=["smoke_test"]).smoke_test(d),
    "segmentation": lambda d: __import__("training.trainers.train_segmentation", fromlist=["smoke_test"]).smoke_test(d),
    "detection": lambda d: __import__("training.trainers.train_detection", fromlist=["smoke_test"]).smoke_test(d),
}


def best_run(task: str) -> str:
    runs = Registry().task(task)["runs"]
    done = {k: v for k, v in runs.items() if v.get("best_value") is not None and v.get("export_dir")}
    if not done:
        raise SystemExit(f"No finished {task} runs with an exported model.")
    mode = next(iter(done.values())).get("monitor", {}).get("mode", "min")
    return (min if mode == "min" else max)(done, key=lambda k: done[k]["best_value"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", choices=list(SMOKE))
    ap.add_argument("--run", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.task:
        print(json.dumps(Registry().load(), indent=2))
        return
    if a.rollback:
        print(json.dumps(rollback(a.task), indent=2))
        return
    run_id = a.run or best_run(a.task)
    print(json.dumps(promote(a.task, run_id, SMOKE[a.task], force=a.force), indent=2))


if __name__ == "__main__":
    main()
