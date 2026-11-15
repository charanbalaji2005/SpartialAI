"""Download datasets into training/data/raw/<name>/ (resumable, with extraction).

  python -m training.datasets.download list
  python -m training.datasets.download coco --parts val annotations     # ~1.3 GB, good first test
  python -m training.datasets.download coco --parts train               # ~18 GB
  python -m training.datasets.download ade20k                           # ~0.9 GB
  python -m training.datasets.download nyu                              # ~2.8 GB labeled .mat + splits
  python -m training.datasets.download pix3d                            # ~3.6 GB
  python -m training.datasets.download objaverse --per-category 20 --categories chair table sofa lamp
  python -m training.datasets.download hypersim                         # prints instructions

Each dataset has its own license and terms of use. Read them (`list` prints where)
before downloading, and don't redistribute the data. URLs are the publishers' official
links at the time of writing; if one moves, download it manually into the same folder.
Kaggle mirrors work too: put the extracted files under training/data/raw/<name>/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from .common import RAW_DIR

SOURCES = {
    "coco": {
        "license": "Annotations CC BY 4.0; images under Flickr terms. https://cocodataset.org/#termsofuse",
        "parts": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "val": "http://images.cocodataset.org/zips/val2017.zip",
            "train": "http://images.cocodataset.org/zips/train2017.zip",
        },
    },
    "ade20k": {
        "license": "Research use; see https://groups.csail.mit.edu/vision/datasets/ADE20K/terms/",
        "parts": {"scene_parsing": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"},
    },
    "nyu": {
        "license": "Research use; see https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
        "parts": {
            "labeled": "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
            "splits": "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
        },
    },
    "pix3d": {
        "license": "Research use; see http://pix3d.csail.mit.edu/",
        "parts": {"all": "http://pix3d.csail.mit.edu/data/pix3d.zip"},
    },
    "objaverse": {"license": "Per-object Creative Commons licenses; see each object's metadata.", "parts": {}},
    "hypersim": {"license": "CC BY-SA 3.0; see https://github.com/apple/ml-hypersim", "parts": {}},
}


def fetch(url: str, dest: Path, retries: int = 5) -> Path:
    """Resumable download using HTTP Range requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return dest
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "SpatialAI"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:      # server ignored Range: restart
                    have = 0
                    part.unlink(missing_ok=True)
                total = int(resp.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                with part.open(mode) as fh:
                    done, t0 = have, time.time()
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            rate = (done - have) / max(time.time() - t0, 1e-3) / 1e6
                            sys.stdout.write(f"\r  {dest.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB  {rate:.1f} MB/s")
                            sys.stdout.flush()
            print()
            part.rename(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"\n  attempt {attempt} failed: {exc}")
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Could not download {url}. Download it manually to {dest}.")


def extract(archive: Path, into: Path) -> None:
    marker = into / f".extracted_{archive.name}"
    if marker.exists():
        return
    print(f"  extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():                 # guard against zip-slip paths
            target = (into / member.filename).resolve()
            if into.resolve() not in target.parents and target != into.resolve():
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(into)
    marker.touch()


def download_objaverse(categories: list[str], per_category: int, out: Path) -> None:
    try:
        import objaverse
    except ImportError:
        raise SystemExit("pip install objaverse   (then run this command again)")
    lvis = objaverse.load_lvis_annotations()           # category -> [uids]
    wanted: dict[str, str] = {}
    for cat in categories:
        matches = [k for k in lvis if k.lower() == cat.lower() or k.lower().startswith(cat.lower() + "_")]
        for k in matches:
            for uid in lvis[k][:per_category]:
                wanted[uid] = k
    if not wanted:
        raise SystemExit(f"No LVIS categories matched {categories}")
    print(f"  downloading {len(wanted)} objects ...")
    paths = objaverse.load_objects(uids=list(wanted), download_processes=4)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for uid, src in paths.items():
        dst = out / f"{uid}.glb"
        if not dst.exists():
            shutil.copy2(src, dst)
        meta[uid] = {"path": dst.name, "category": wanted[uid]}
    (out / "objects.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["list", *SOURCES])
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--categories", nargs="*", default=["chair", "table", "sofa", "lamp", "bed", "car", "plant"])
    ap.add_argument("--per-category", type=int, default=20)
    a = ap.parse_args()

    if a.dataset == "list":
        for name, info in SOURCES.items():
            print(f"{name:10s} parts: {', '.join(info['parts']) or '-'}\n           license: {info['license']}")
        return
    out = RAW_DIR / a.dataset
    print(f"{a.dataset}: {SOURCES[a.dataset]['license']}")
    if a.dataset == "objaverse":
        download_objaverse(a.categories, a.per_category, out)
    elif a.dataset == "hypersim":
        print("Hypersim is ~1.9 TB in full. Download only a few scenes with Apple's script:\n"
              "  git clone https://github.com/apple/ml-hypersim\n"
              "  python ml-hypersim/code/python/tools/dataset_download_images.py "
              f"--downloads_dir {out} --decompress_dir {out}\n"
              "Stop it after a few scenes (ai_001_001 ...). You need the *_final_preview JPGs and "
              "*_geometry_hdf5 depth_meters files. Then run prepare_depth --source hypersim.")
    else:
        parts = a.parts or list(SOURCES[a.dataset]["parts"])
        for p in parts:
            url = SOURCES[a.dataset]["parts"][p]
            f = fetch(url, out / "_downloads" / url.rsplit("/", 1)[-1])
            if f.suffix == ".zip":
                extract(f, out)
            else:
                shutil.copy2(f, out / f.name) if not (out / f.name).exists() else None
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
