"""Export the cached fold detections of the two released datasets to compressed npz files.

  python tools/export_detections.py --study <study repository>

Reads outputs/cache/folds/<name>[_<size>]_f<k>.pkl for css_v30 (the source) and hard_hat_workers,
for every YOLOX size present (no size tag is Nano), and writes

  detections/<name>/yolox_<size>_fold<k>.npz   detections of fold model k on every photo
  data/manifests/<name>.csv                    file, fold (source only), height, width

Each npz holds the photos in the cached order:
  file        base name of the photo (no directory)
  fold        fold in which the photo was held out (source only)
  size_hw     image height and width
  gt_boxes, gt_cls, gt_offsets           ground truth of photo i is rows gt_offsets[i]:gt_offsets[i+1]
  det_boxes, det_scores, det_cls, det_offsets  detections of photo i, likewise
Boxes are x1, y1, x2, y2 in pixels; classes index the unified classes helmet, head, vest, person.
Nothing else from the cached record is stored: no directory and no image hash. Every file is read
back and compared with the cache before it is accepted.
"""
import argparse
import datetime
import io
import os
import pickle
import shutil
import zipfile

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = datetime.date.today().isoformat()
NAMES = ("css_v30", "hard_hat_workers")
SIZES = (("", "nano"), ("s", "s"), ("m", "m"), ("l", "l"))
N_FOLDS = 5
LIMIT = 50 * 1024 * 1024


def pack(records, source):
    def cat(key, shape, offsets=False):
        parts = [np.asarray(im[key]).reshape(shape) for im in records]
        lens = np.array([len(p) for p in parts], dtype=np.int64)
        arr = np.concatenate(parts) if parts else np.zeros((0,) + shape[1:])
        return arr, np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out = {"file": np.array([os.path.basename(im["path"]) for im in records]),
           "size_hw": np.array([tuple(im["size"]) for im in records], dtype=np.int64)}
    if source:
        out["fold"] = np.array([im["fold"] for im in records], dtype=np.int64)
    out["gt_boxes"], out["gt_offsets"] = cat("gt_boxes", (-1, 4))
    out["gt_cls"], off = cat("gt_cls", (-1,))
    assert np.array_equal(off, out["gt_offsets"])
    out["det_boxes"], out["det_offsets"] = cat("det_boxes", (-1, 4))
    out["det_scores"], off = cat("det_scores", (-1,))
    assert np.array_equal(off, out["det_offsets"])
    out["det_cls"], off = cat("det_cls", (-1,))
    assert np.array_equal(off, out["det_offsets"])
    return out


def unpack(z):
    """The inverse of pack: one dict per photo, in the layout routing.core.Records.from_images reads."""
    go, do = z["gt_offsets"], z["det_offsets"]
    rows = []
    for i, name in enumerate(z["file"]):
        im = {"file": str(name), "size": tuple(int(v) for v in z["size_hw"][i]),
              "gt_boxes": z["gt_boxes"][go[i]:go[i + 1]], "gt_cls": z["gt_cls"][go[i]:go[i + 1]],
              "det_boxes": z["det_boxes"][do[i]:do[i + 1]], "det_scores": z["det_scores"][do[i]:do[i + 1]],
              "det_cls": z["det_cls"][do[i]:do[i + 1]]}
        if "fold" in z:
            im["fold"] = int(z["fold"][i])
        rows.append(im)
    return rows


def same(records, rows):
    if len(records) != len(rows):
        return False
    for a, b in zip(records, rows):
        if os.path.basename(a["path"]) != b["file"] or tuple(a["size"]) != b["size"] or a.get("fold") != b.get("fold"):
            return False
        for key, shape in (("gt_boxes", (-1, 4)), ("gt_cls", (-1,)), ("det_boxes", (-1, 4)), ("det_scores", (-1,)), ("det_cls", (-1,))):
            if not np.array_equal(np.asarray(a[key]).reshape(shape), b[key]):
                return False
    return True


def npz_bytes(arrays):
    """A compressed npz archive, as numpy.savez_compressed writes it, with fixed member dates so that
    reruns give identical bytes (numpy.savez_compressed itself cannot take an array named 'file')."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)
    return buf.getvalue()


def put_npz(rel, data, log):
    put_bytes(rel, data, log)


def put_bytes(rel, data, log):
    dst = os.path.join(PKG, rel)
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == data:
                log.append(f"unchanged  {rel}")
                return
        arc = os.path.join(PKG, "archive", f"{rel}_{TODAY}")
        os.makedirs(os.path.dirname(arc), exist_ok=True)
        shutil.move(dst, arc)
        log.append(f"archived   {rel}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    log.append(f"written    {rel}")


def put_text(rel, text, log):
    dst = os.path.join(PKG, rel)
    data = text.encode("utf8")
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == data:
                log.append(f"unchanged  {rel}")
                return
        arc = os.path.join(PKG, "archive", f"{rel}_{TODAY}")
        os.makedirs(os.path.dirname(arc), exist_ok=True)
        shutil.move(dst, arc)
        log.append(f"archived   {rel}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    log.append(f"written    {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    args = ap.parse_args()
    cache = os.path.join(os.path.abspath(args.study), "outputs", "cache", "folds")
    log, summary, total = [], [], 0
    for name in NAMES:
        reference = None
        for tag, size in SIZES:
            paths = [os.path.join(cache, f"{name}_{tag + '_' if tag else ''}f{k}.pkl") for k in range(N_FOLDS)]
            if not all(os.path.exists(p) for p in paths):
                summary.append(f"{name} {size}: not present, skipped")
                continue
            for k, p in enumerate(paths):
                with open(p, "rb") as f:
                    records = pickle.load(f)
                arrays = pack(records, source=(name == "css_v30"))
                data = npz_bytes(arrays)
                with np.load(io.BytesIO(data)) as z:
                    if not same(records, unpack(z)):
                        raise SystemExit(f"read-back check failed for {name} {size} fold {k}")
                manifest = (list(arrays["file"]), arrays["size_hw"].tolist(), arrays.get("fold", np.zeros(0)).tolist())
                if reference is None:
                    reference = manifest
                elif manifest != reference:
                    raise SystemExit(f"{name} {size} fold {k}: photo list differs from the first cache")
                rel = f"detections/{name}/yolox_{size}_fold{k}.npz"
                put_npz(rel, data, log)
                nbytes = os.path.getsize(os.path.join(PKG, rel))
                if nbytes >= LIMIT:
                    raise SystemExit(f"{rel} is {nbytes:,} bytes, over the 50 MB limit")
                total += nbytes
                summary.append(f"{name} {size} fold {k}: {len(records)} photos, {len(arrays['gt_cls'])} labels, "
                               f"{len(arrays['det_cls'])} detections, {nbytes:,} bytes")
        files, sizes, folds = reference
        assert not any("," in fn or '"' in fn for fn in files), "file names must not contain commas or quotes"
        lines = ["file,fold,height,width"] + [f"{fn},{folds[i] if folds else ''},{h},{w}" for i, (fn, (h, w)) in enumerate(zip(files, sizes))]
        put_text(f"data/manifests/{name}.csv", "\n".join(lines) + "\n", log)
    print("\n".join(log))
    print("\n".join(summary))
    print(f"detections total: {total:,} bytes")


if __name__ == "__main__":
    main()
