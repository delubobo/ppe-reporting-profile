"""Write the near-duplicate exclusion lists that routing.experiments_cv.clean_sets computes in memory.

  python tools/export_duplicate_lists.py --study <study repository>

Run with PPE_VARIANT unset: the YOLOX Nano fold caches define the lists. The loop below is the one
in clean_sets (difference hashes, at most 4 differing bits):
  source   pairs of source photos held out in different folds; both photos of a pair are dropped
  targets  each target in the order of experiments.TARGETS, then sh17, against the source and then
           against every previously kept target; the target photo (file_b) is dropped

One CSV per comparison is written to data/duplicates/ with columns
dataset_a, file_a, dataset_b, file_b, hash_distance (file names only, never directories), and
data/duplicates/summary.json holds the counts. The script stops unless those counts equal the
cleaning block of outputs/results/experiments_cv.json.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = datetime.date.today().isoformat()
BITS = 4


def put(rel, text, log):
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


def rows_csv(rows):
    lines = ["dataset_a,file_a,dataset_b,file_b,hash_distance"]
    for r in rows:
        assert not any("," in str(v) for v in r)
        lines.append(",".join(str(v) for v in r))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    args = ap.parse_args()
    if os.environ.get("PPE_VARIANT", ""):
        raise SystemExit("unset PPE_VARIANT: the YOLOX Nano caches define the lists")
    study = os.path.abspath(args.study)
    sys.path.insert(0, study)
    os.chdir(study)                       # the study code reads its caches by relative path
    import numpy as np
    from routing.experiments_cv import photo_group, target_names
    from routing.folds import N_FOLDS, load_fold_cache
    from routing.run_study import RESULT_DIR, near_duplicates

    base = lambda im: os.path.basename(im["path"])
    log, files = [], {}
    photos = load_fold_cache("css_v30", 0)
    fold = np.array([im["fold"] for im in photos])
    pairs = [(i, j, d) for i, j, d in near_duplicates(photos, photos, BITS) if i != j and fold[i] != fold[j]]
    leaked = {i for i, j, _ in pairs} | {j for _, j, _ in pairs}
    keep = [i for i in range(len(photos)) if i not in leaked]
    files["css_v30_vs_css_v30.csv"] = [("css_v30", base(photos[i]), "css_v30", base(photos[j]), d) for i, j, d in pairs]
    count = {"source_photos": len(photos), "source_dropped_duplicate_in_another_fold": len(leaked), "source_kept": len(keep),
             "source_kept_by_fold": [int((fold[keep] == k).sum()) for k in range(N_FOLDS)],
             "source_video_frames_kept": int(sum(photo_group(photos[i]["path"]) == "video frame" for i in keep))}
    kept_ref = {}
    for name in target_names():
        images = load_fold_cache(name, 0)
        drop = set()
        for other_name, other in [("source", photos)] + list(kept_ref.items()):
            found = near_duplicates(other, images, BITS)
            hits = {j for _, j, _ in found}
            count[f"{name}_dropped_vs_{other_name}"] = len(hits - drop)
            drop |= hits
            a = "css_v30" if other_name == "source" else other_name
            files[f"{a}_vs_{name}.csv"] = [(a, base(other[i]), name, base(images[j]), d) for i, j, d in found]
        kept_ref[name] = [images[j] for j in range(len(images)) if j not in drop]
        count[f"{name}_kept"] = len(kept_ref[name])

    with open(os.path.join(RESULT_DIR, "experiments_cv.json"), encoding="utf8") as f:
        saved = json.load(f)["cleaning"]
    if count != saved:
        print("computed counts", json.dumps(count))
        print("saved counts   ", json.dumps(saved))
        raise SystemExit("counts differ from the saved cleaning block")
    for fname, rows in files.items():
        put(f"data/duplicates/{fname}", rows_csv(rows), log)
    summary = {"bits": BITS, "detector_caches": "YOLOX Nano fold caches (PPE_VARIANT unset)",
               "cleaning": count, "pairs_per_file": {k: len(v) for k, v in files.items()},
               "equals_experiments_cv_json_cleaning_block": True}
    put("data/duplicates/summary.json", json.dumps(summary, indent=2) + "\n", log)
    print("\n".join(log))
    print("counts equal the saved cleaning block:", json.dumps(count))
    print("pairs per file:", json.dumps(summary["pairs_per_file"]))


if __name__ == "__main__":
    main()
