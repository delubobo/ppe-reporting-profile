"""
Study runner.

  cache     run the detector once over a dataset and store detections next to the ground truth
  analyze   score a cached target dataset under the rule shipped with the model, and run the
            label-budget experiment
  dedupe    look for near-duplicate images between two cached datasets (leakage check)

Examples:
  python -m routing.run_study cache --name shwd --root data/shwd
  python -m routing.run_study analyze --name shwd
  python -m routing.run_study dedupe --a shwd --b hard_hat_workers
"""
import argparse
import json
import os
import pickle

import numpy as np

from . import core, datasets

MODEL_DIR = "models/site-photo-check-v1"
CACHE_DIR = "outputs/cache"
RESULT_DIR = "outputs/results"


# ---------------------------------------------------------------- cache
def cache(name, images, model_dir=MODEL_DIR):
    """images: list of dicts with path, gt_boxes, gt_cls (from routing.datasets loaders)."""
    import cv2

    from .detect import Detector

    detector = Detector(model_dir)
    kept = []
    for n, im in enumerate(images):
        img = cv2.imread(im["path"])
        if img is None:
            continue
        boxes, scores, cls = detector(img)
        kept.append({**im, "det_boxes": boxes, "det_scores": scores, "det_cls": cls, "size": img.shape[:2], "dhash": dhash(img)})
        if n % 500 == 0:
            print(f"  {name}: {n}/{len(images)}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(kept, f)
    print(f"{name}: cached {len(kept)} of {len(images)} images")
    return kept


def load_cache(name):
    with open(os.path.join(CACHE_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- leakage check
def dhash(img_bgr):
    """64-bit difference hash; near-identical images differ in only a few bits."""
    import cv2

    g = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def near_duplicates(a, b, max_bits=4):
    """Pairs (index in a, index in b, differing bits) whose hashes differ by at most max_bits."""
    ha = np.array([im["dhash"] for im in a], dtype=np.uint64)
    hb = np.array([im["dhash"] for im in b], dtype=np.uint64)
    pairs = []
    for start in range(0, len(ha), 512):
        dist = np.bitwise_count(ha[start:start + 512, None] ^ hb[None, :])
        for i, j in zip(*np.where(dist <= max_bits)):
            pairs.append((start + int(i), int(j), int(dist[i, j])))
    return pairs


# ---------------------------------------------------------------- analysis
def shipped_rule(model_dir=MODEL_DIR):
    """The calibration map and thresholds published with the model, exactly as the page applies them."""
    with open(os.path.join(model_dir, "calibration.json"), encoding="utf8") as f:
        c = json.load(f)
    cal = core.Platt()
    cal.a, cal.b = c["a"], c["b"]
    return cal, c["suggested_thresholds"]["route_to_person_low"], c["suggested_thresholds"]["act"]


def analyze(name, cached=None, budgets=(10, 25, 50, 100, 200, 400), reps=200, alpha=0.10, beta=0.10, delta=0.10, seed=0, rule="iou"):
    cached = load_cache(name) if cached is None else cached
    ids = datasets.evaluable_ids(name)
    rec = core.Records.from_images(cached, floor=0.05, class_ids=ids, rule=rule)
    rec_ap = core.Records.from_images(cached, floor=0.01, class_ids=ids, rule=rule)
    cal, t_low, t_high = shipped_rule()
    p = cal.predict(rec.det_score, rec.det_cls)

    out = {
        "dataset": name, "match_rule": rule, "n_images": rec.n_images, "evaluable": [core.CLASSES[i] for i in ids],
        "n_gt": {core.CLASSES[i]: int((rec.gt_cls == i).sum()) for i in ids},
        "n_detections": int(len(rec.det_score)),
        "ap50": {core.CLASSES[i]: core.average_precision(rec_ap, i) for i in ids},
        "ece_raw": core.ece(rec.det_score, rec.det_tp)[0],
        "ece_shipped": core.ece(p, rec.det_tp)[0],
        "reliability_shipped": core.ece(p, rec.det_tp)[1],
        "shipped_rule": core.routing_metrics(rec, p, t_low, t_high),
        "shipped_rule_by_class": {},
        "targets": {"alpha": alpha, "beta": beta, "delta": delta},
    }
    for i in ids:
        m = rec.det_cls == i
        rep = m & (p >= t_high)
        out["shipped_rule_by_class"][core.CLASSES[i]] = {
            "n_report": int(rep.sum()),
            "false_report_rate": float(1 - rec.det_tp[rep].mean()) if rep.any() else None,
            "ece_shipped": core.ece(p[m], rec.det_tp[m])[0],
        }
    # What a rule selected on all of the target data would be: the best a site could do with every label.
    tl_all, th_all = core.select_thresholds(rec, p, alpha, beta, delta)
    tl_all = t_low if tl_all is None else tl_all  # keep the shipped hide threshold when no miss target is certifiable
    out["selected_on_all_target_data"] = {"t_low": tl_all, "t_high": th_all, **core.routing_metrics(rec, p, tl_all, th_all)}
    out["miss_control_feasible"] = {str(b): core.select_t_low(rec, p, b) for b in (0.1, 0.2, 0.3, 0.5)}

    result = core.budget_curve(None, rec, budgets, reps=reps, alpha=alpha, beta=beta, delta=delta, seed=seed, rule=(cal, t_low, t_high))
    out["budget_source_rule"] = core.summarize_budget(result, "source")
    out["budget_local_thresholds"] = core.summarize_budget(result, "target")
    ece_rows = {}
    for n in sorted({r["n"] for r in result["rows"]}):
        rows = [r for r in result["rows"] if r["n"] == n]
        ece_rows[n] = {k: float(np.mean([r[k] for r in rows])) for k in ("ece_source", "ece_refit_platt", "ece_refit_isotonic")}
    out["budget_ece"] = ece_rows

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{name}_{rule}.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze"); a.add_argument("--name", required=True); a.add_argument("--reps", type=int, default=200)
    d = sub.add_parser("dedupe"); d.add_argument("--a", required=True); d.add_argument("--b", required=True); d.add_argument("--bits", type=int, default=4)
    c = sub.add_parser("cache"); c.add_argument("--name", required=True); c.add_argument("--root", required=True)
    args = ap.parse_args()
    if args.cmd == "cache":
        cache(args.name, datasets.load(args.name, args.root))
    elif args.cmd == "analyze":
        out = analyze(args.name, reps=args.reps)
        print(json.dumps({k: v for k, v in out.items() if k != "reliability_shipped"}, indent=2))
    elif args.cmd == "dedupe":
        a_imgs, b_imgs = load_cache(args.a), load_cache(args.b)
        pairs = near_duplicates(a_imgs, b_imgs, args.bits)
        print(f"{len(pairs)} near-duplicate pairs between {args.a} ({len(a_imgs)}) and {args.b} ({len(b_imgs)})")
        for i, j, dist in pairs[:20]:
            print(f"  {dist} bits  {a_imgs[i]['path']}  <->  {b_imgs[j]['path']}")


if __name__ == "__main__":
    main()
