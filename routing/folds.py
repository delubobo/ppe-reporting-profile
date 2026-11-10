"""
Five-fold models: score every photo with all five detectors in one pass.

  python -m routing.folds source              the 700 source photos (data/css_v30; 656 remain after duplicate cleaning)
  python -m routing.folds target --name chv   a target dataset already cached with the v1 model
  python -m routing.folds reference           the source photos scored by the published v1 model
  python -m routing.folds import --src dell_results.zip --size m
                                              detections scored on the Dell for a larger model size

The environment variable PPE_VARIANT (for example "m") makes every fold cache read below come
from that model size, so `PPE_VARIANT=m python -m routing.experiments_cv` reruns the whole
analysis for it. Unset, the YOLOX-Nano folds are used.

Each model k writes outputs/cache/folds/<name>_f<k>.pkl, a list of photo records in the same
layout as run_study.cache. Source records also carry "fold": the fold in which the photo
was held out, so the out-of-fold detections of photo i are those of model i["fold"].
"""
import argparse
import json
import os
import pickle

import cv2

from . import datasets
from .detect import Detector
from .run_study import CACHE_DIR, dhash, load_cache

FOLD_DIR = "models/folds"
FOLD_CACHE = os.path.join(CACHE_DIR, "folds")
N_FOLDS = 5
SOURCE_ROOT = "data/css_v30"
VARIANT = os.environ.get("PPE_VARIANT", "")


def source_photos():
    """Ground truth for the 700 source photos (656 after duplicate cleaning), from the held-out label file of the fold each belongs to."""
    with open(os.path.join(FOLD_DIR, "folds.json"), encoding="utf8") as f:
        split = {p["file_name"]: p["roboflow_split"] for p in json.load(f)["photos"]}
    mapping = datasets.REGISTRY["css_v27"]["mapping"]
    photos = []
    for k in range(N_FOLDS):
        images = datasets.load_coco(os.path.join(FOLD_DIR, f"fold{k}", "heldout_annotations.json"), "", mapping)
        for im in images:
            name = os.path.basename(im["path"])
            photos.append({**im, "path": os.path.join(SOURCE_ROOT, split[name], name), "fold": k})
    return photos


def score(name, images, model_dirs=None, tags=None):
    model_dirs = model_dirs or [os.path.join(FOLD_DIR, f"fold{k}") for k in range(N_FOLDS)]
    tags = tags or [f"f{k}" for k in range(len(model_dirs))]
    detectors = [Detector(d, threads=4) for d in model_dirs]
    kept = [[] for _ in detectors]
    for n, im in enumerate(images):
        img = cv2.imread(im["path"])
        if img is None:
            continue
        base = {k: v for k, v in im.items() if not k.startswith("det_")}
        base.update({"size": img.shape[:2], "dhash": dhash(img)})
        for k, detector in enumerate(detectors):
            boxes, scores, cls = detector(img)
            kept[k].append({**base, "det_boxes": boxes, "det_scores": scores, "det_cls": cls})
        if n % 500 == 0:
            print(f"  {name}: {n}/{len(images)}", flush=True)
    os.makedirs(FOLD_CACHE, exist_ok=True)
    for tag, rows in zip(tags, kept):
        with open(os.path.join(FOLD_CACHE, f"{name}_{tag}.pkl"), "wb") as f:
            pickle.dump(rows, f)
    print(f"{name}: scored {len(kept[0])} of {len(images)} photos with {len(detectors)} models", flush=True)


def load_fold_cache(name, k):
    """k is a fold number, or a tag such as "v1"."""
    tag = k if not isinstance(k, int) else (f"{VARIANT}_f{k}" if VARIANT else f"f{k}")
    with open(os.path.join(FOLD_CACHE, f"{name}_{tag}.pkl"), "rb") as f:
        return pickle.load(f)


def import_detections(src, size):
    """Merge detections scored elsewhere (dell_kit/score.py) with the ground truth already cached here.

    src is dell_results.zip or its unpacked folder. Photo order is that of the manifest shipped
    in the package, which must match the photos of the Nano caches one for one.
    """
    import io
    import zipfile

    import numpy as np

    if os.path.isdir(src):
        read = lambda rel: open(os.path.join(src, rel), "rb").read()
    else:
        archive = zipfile.ZipFile(src)
        read = archive.read
    manifest = json.loads(read("manifest.json").decode("utf8"))
    for name, rel_paths in manifest.items():
        with open(os.path.join(FOLD_CACHE, f"{name}_f0.pkl"), "rb") as f:
            base = pickle.load(f)
        local = [im["path"].replace(chr(92), "/").split("data/", 1)[1] for im in base]
        if local != rel_paths:
            raise SystemExit(f"{name}: the photo list in the package does not match the local cache ({len(rel_paths)} against {len(local)} photos).")
        for k in range(N_FOLDS):
            z = np.load(io.BytesIO(read(f"detections/{size}/{name}_f{k}.npz")))
            counts = z["counts"]
            if len(counts) != len(base) or counts.sum() != len(z["scores"]):
                raise SystemExit(f"{name} fold {k}: {len(counts)} photos and {len(z['scores'])} detections do not add up.")
            ends = np.cumsum(counts)
            boxes, scores, cls = z["boxes"].astype(np.float64), z["scores"].astype(np.float64), z["cls"].astype(np.int64)  # an NpzFile decompresses on every access
            rows = []
            for im, start, end in zip(base, ends - counts, ends):
                row = {key: v for key, v in im.items() if not key.startswith("det_")}
                row.update({"det_boxes": boxes[start:end], "det_scores": scores[start:end], "det_cls": cls[start:end]})
                rows.append(row)
            with open(os.path.join(FOLD_CACHE, f"{name}_{size}_f{k}.pkl"), "wb") as f:
                pickle.dump(rows, f)
        print(f"{name}: imported {len(base)} photos x {N_FOLDS} folds for size {size}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("source")
    sub.add_parser("reference")
    t = sub.add_parser("target"); t.add_argument("--name", required=True)
    i = sub.add_parser("import"); i.add_argument("--src", required=True); i.add_argument("--size", required=True)
    args = ap.parse_args()
    if args.cmd == "source":
        score("css_v30", source_photos())
    elif args.cmd == "import":
        import_detections(args.src, args.size)
    elif args.cmd == "reference":
        from .run_study import MODEL_DIR
        score("css_v30", source_photos(), [MODEL_DIR], ["v1"])
    else:
        score(args.name, load_cache(args.name))  # reuses the parsed ground truth; detections are replaced


if __name__ == "__main__":
    main()
