"""
Paper experiments on cached detections. Run after `run_study cache` for every dataset.

  python -m routing.experiments            all experiments, results in outputs/results/experiments.json

E1  Transfer. Fit calibration and thresholds on a random half of the clean source held-out
    photos, score the other half and every target. Repeated over random halves.
E2  Sensitivity of the whole-head datasets to the containment matching parameters.
E3  Label budget on the cleaned targets, with the asymptotic detection-level test and the
    exact photo-level test.
E4  Whether any hide threshold can meet a miss-rate target for the violation class.
E5  Whether stretching target photos to a square, as the source export was, changes the picture.

E2 to E5 use one fixed "source rule": calibration and report threshold fitted on all clean
source held-out photos. The hide threshold is fixed at 0.50 when no miss-rate target can be
certified (which is the case here; see E4).

Cleaning
  source held-out   validation + test images whose Roboflow original does not also appear in
                    the training folder (the published split repeats ten originals)
  targets           images that near-duplicate a source image are dropped; duplicates between
                    targets are dropped from the lower-priority set (CHV > Hard Hat Workers > SHWD,
                    because CHV is the only target with vests and whole persons)
"""
import json
import os
import pickle

import numpy as np

from . import core, datasets
from .run_study import CACHE_DIR, RESULT_DIR, load_cache, near_duplicates

TARGETS = ("chv", "hard_hat_workers", "shwd")          # dedupe priority order; "sh17" is appended by experiments_cv when its cache exists
MATCH = {"chv": "iou", "hard_hat_workers": "contain", "shwd": "contain",
         "sh17": {core.CLASSES.index("head"): "contain"}}   # SH17: whole-head labels for bare heads (containment), hat and body boxes by IoU; roadmap Section 10
ALPHA, BETA, DELTA = 0.10, 0.10, 0.10
ALPHA_PHOTO = 0.20
T_LOW_DEFAULT = 0.5
ALL_IDS = tuple(range(len(core.CLASSES)))


def stem(path):
    return os.path.basename(path).split(".rf.")[0]


def clean_sets(bits=4):
    train, valid, test = (load_cache(f"css_v27_{s}") for s in ("train", "valid", "test"))
    train_stems = {stem(im["path"]) for im in train}
    seen, source = set(), []
    for im in valid + test:
        s = stem(im["path"])
        if s not in train_stems and s not in seen:
            seen.add(s); source.append(im)
    log = {"source_heldout": len(source), "source_dropped_as_repeated_in_training": len(valid) + len(test) - len(source)}
    kept = {}
    for name in TARGETS:
        images = load_cache(name)
        drop = set()
        for other_name, other in [("source", valid + test + train)] + [(k, v) for k, v in kept.items()]:
            hits = {j for _, j, _ in near_duplicates(other, images, bits)}
            log[f"{name}_dropped_vs_{other_name}"] = len(hits - drop)
            drop |= hits
        kept[name] = [im for j, im in enumerate(images) if j not in drop]
        log[f"{name}_kept"] = len(kept[name])
    return source, kept, log


def records(name, images, rule=None):
    return core.Records.from_images(images, class_ids=datasets.evaluable_ids(name), rule=rule or MATCH[name])


def class_rates(rec, p, t_high, ids):
    out = {}
    for i in ids:
        rep = (rec.det_cls == i) & (p >= (np.inf if t_high is None else t_high))
        out[core.CLASSES[i]] = {"n_report": int(rep.sum()), "false_report_rate": float(1 - rec.det_tp[rep].mean()) if rep.any() else None}
    return out


def spread(values):
    v = np.array([x for x in values if x is not None and x == x], dtype=float)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def source_rule(source, method="cluster", alpha=ALPHA):
    """Calibration and thresholds fitted on all clean source held-out photos."""
    src = core.Records.from_images(source, class_ids=ALL_IDS)
    cal = core.make_calibrator("platt").fit(src.det_score, src.det_tp)
    p = cal.predict(src.det_score)
    t_high = core.select_t_high(src, p, alpha, DELTA, method=method)
    t_low = core.select_t_low(src, p, BETA)
    info = {"method": method, "alpha": alpha, "n_photos": src.n_images, "platt_a": cal.a, "platt_b": cal.b, "t_high": t_high,
            "t_low_certified": t_low, "t_low_used": T_LOW_DEFAULT if t_low is None else t_low,
            "ece_raw": core.ece(src.det_score, src.det_tp)[0], "ece_calibrated": core.ece(p, src.det_tp)[0]}
    return (cal, info["t_low_used"], t_high), info


def e1_transfer(source, targets, reps=50, method="cluster", seed=0):
    rng = np.random.default_rng(seed)
    src = core.Records.from_images(source, class_ids=ALL_IDS)
    tgt = {n: records(n, ims) for n, ims in targets.items()}
    keys = ("false_report_rate", "false_alarm_images", "share_routed", "routed_per_image", "miss_rate", "n_report")
    rows = {n: {k: [] for k in keys} for n in ("source", *tgt)}
    per_class = {n: {} for n in ("source", *tgt)}
    ece = {n: {"raw": [], "calibrated": []} for n in ("source", *tgt)}
    chosen = {"t_high": [], "platt_a": [], "platt_b": [], "no_report_threshold": 0}
    for _ in range(reps):
        perm = rng.permutation(src.n_images)
        fit, held = src.subset(perm[: src.n_images // 2]), src.subset(perm[src.n_images // 2:])
        cal = core.make_calibrator("platt").fit(fit.det_score, fit.det_tp)
        th = core.select_t_high(fit, cal.predict(fit.det_score), ALPHA, DELTA, method=method)
        chosen["platt_a"].append(cal.a); chosen["platt_b"].append(cal.b)
        if th is None:
            chosen["no_report_threshold"] += 1
            continue
        chosen["t_high"].append(th)
        for n, rec in (("source", held), *tgt.items()):
            p = cal.predict(rec.det_score)
            m = core.routing_metrics(rec, p, T_LOW_DEFAULT, th)
            for k in keys:
                rows[n][k].append(m[k])
            ece[n]["raw"].append(core.ece(rec.det_score, rec.det_tp)[0]); ece[n]["calibrated"].append(core.ece(p, rec.det_tp)[0])
            ids = ALL_IDS if n == "source" else datasets.evaluable_ids(n)
            for c, v in class_rates(rec, p, th, ids).items():
                per_class[n].setdefault(c, []).append(v["false_report_rate"])
    out = {"method": method, "reps": reps, "alpha": ALPHA, "delta": DELTA, "fit_photos": src.n_images // 2,
           "chosen": {k: (spread(v) if isinstance(v, list) else v) for k, v in chosen.items()}, "datasets": {}}
    for n in rows:
        out["datasets"][n] = {k: spread(v) for k, v in rows[n].items()}
        out["datasets"][n]["by_class_false_report"] = {c: spread(v) for c, v in per_class[n].items()}
        out["datasets"][n]["ece_raw"] = spread(ece[n]["raw"]); out["datasets"][n]["ece_calibrated"] = spread(ece[n]["calibrated"])
        out["datasets"][n]["share_of_runs_within_alpha"] = float(np.mean(np.array(rows[n]["false_report_rate"]) <= ALPHA)) if rows[n]["false_report_rate"] else None
    return out


def e2_containment(targets, rule):
    cal, tl, th = rule
    out = {}
    for n in ("hard_hat_workers", "shwd"):
        ids = datasets.evaluable_ids(n)
        out[n] = {}
        for label, match in (("iou 0.5", "iou"), ("inside 0.5, iou 0.1", ("contain", 0.5, 0.1)), ("inside 0.7, iou 0.2", ("contain", 0.7, 0.2)),
                             ("inside 0.7, iou 0.3", ("contain", 0.7, 0.3)), ("inside 0.9, iou 0.2", ("contain", 0.9, 0.2))):
            rec = records(n, targets[n], match)
            p = cal.predict(rec.det_score)
            m = core.routing_metrics(rec, p, tl, th)
            out[n][label] = {"false_report_rate": m["false_report_rate"], "miss_rate": m["miss_rate"],
                             "by_class": {c: v["false_report_rate"] for c, v in class_rates(rec, p, th, ids).items()}}
    return out


def e3_budget(targets, rules, budgets=(10, 25, 50, 100, 200, 400), reps=100):
    out = {}
    for n, ims in targets.items():
        rec = records(n, ims)
        out[n] = {}
        for method, alpha, risk in (("cluster", ALPHA, "false_report_rate"), ("image", ALPHA_PHOTO, "false_alarm_images")):
            res = core.budget_curve(None, rec, budgets, reps=reps, alpha=alpha, beta=BETA, delta=DELTA, method=method, rule=rules[method], seed=1)
            out[n][method] = {"alpha": alpha, "risk": risk, "local": core.summarize_budget(res, "target", risk), "source_rule": core.summarize_budget(res, "source", risk)}
    return out


def e4_miss_feasibility(source, targets, rule):
    """Highest hide threshold that meets each miss-rate target for bare heads, or None when none does."""
    cal = rule[0]
    out = {}
    sets = {"source": core.Records.from_images(source, class_ids=ALL_IDS), **{n: records(n, ims) for n, ims in targets.items()}}
    for n, rec in sets.items():
        if not np.isin(rec.gt_cls, core.VIOLATION_IDS).any():
            continue
        p = cal.predict(rec.det_score)
        floor_miss = core.routing_metrics(rec, p, 0.0, None)["miss_rate"]   # nothing hidden: what the detector never saw
        out[n] = {"undetected_share": floor_miss, **{str(b): core.select_t_low(rec, p, b) for b in (0.1, 0.2, 0.3, 0.5)}}
    return out


def e5_stretch(targets, rule):
    """Letterboxed (as deployed) against stretched-to-square preprocessing, on the cleaned photos."""
    cal, tl, th = rule
    out = {}
    for n in ("chv", "shwd"):
        path = os.path.join(CACHE_DIR, f"{n}_stretch.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            stretched = pickle.load(f)
        keep = {im["path"] for im in targets[n]}
        stretched = [im for im in stretched if im["path"] in keep]
        ids = datasets.evaluable_ids(n)
        out[n] = {}
        for label, ims in (("letterbox", targets[n]), ("stretch", stretched)):
            rec = records(n, ims)
            rec_ap = core.Records.from_images(ims, floor=0.01, class_ids=ids, rule=MATCH[n])
            p = cal.predict(rec.det_score)
            m = core.routing_metrics(rec, p, tl, th)
            out[n][label] = {"n_photos": rec.n_images, "false_report_rate": m["false_report_rate"],
                             "by_class": {c: v["false_report_rate"] for c, v in class_rates(rec, p, th, ids).items()},
                             "ap50": {core.CLASSES[i]: core.average_precision(rec_ap, i) for i in ids}}
    return out


def main():
    source, targets, log = clean_sets()
    rule, info = source_rule(source, "cluster", ALPHA)
    rule_photo, info_photo = source_rule(source, "image", ALPHA_PHOTO)
    result = {"cleaning": log, "source_rule": info, "source_rule_photo": info_photo,
              "e1_cluster": e1_transfer(source, targets, method="cluster"),
              "e2_containment": e2_containment(targets, rule),
              "e3_budget": e3_budget(targets, {"cluster": rule, "image": rule_photo}),
              "e4_miss": e4_miss_feasibility(source, targets, rule),
              "e5_stretch": e5_stretch(targets, rule)}
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, "experiments.json"), "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)

    f3 = lambda s: "n/a" if s is None else (f"{s['mean']:.3f} [{s['p05']:.3f}, {s['p95']:.3f}]" if isinstance(s, dict) else f"{s:.3f}")
    pc = lambda d: ", ".join(f"{c} {'n/a' if v is None else format(v, '.3f')}" for c, v in d.items())
    print("CLEANING", json.dumps(log))
    print("SOURCE RULE", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}))
    print("SOURCE RULE (photo-level)", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info_photo.items()}))
    e1 = result["e1_cluster"]
    print(f"E1 transfer, {e1['reps']} random halves of {e1['fit_photos']} photos: report threshold {f3(e1['chosen']['t_high'])}, none certified in {e1['chosen']['no_report_threshold']} runs")
    for n, d in e1["datasets"].items():
        print(f"  {n:17s} false-report {f3(d['false_report_rate'])} within target {d['share_of_runs_within_alpha']:.2f} | photos with a false report {f3(d['false_alarm_images'])} | routed {f3(d['share_routed'])} | missed heads {f3(d['miss_rate'])} | ECE raw {f3(d['ece_raw'])} cal {f3(d['ece_calibrated'])}")
        print("                    by class: " + ", ".join(f"{c} {f3(v)}" for c, v in d["by_class_false_report"].items()))
    print("E2 containment sensitivity (source rule)")
    for n, rows in result["e2_containment"].items():
        for label, v in rows.items():
            print(f"  {n:17s} {label:22s} false-report {v['false_report_rate']:.3f} | {pc(v['by_class'])} | missed heads {v['miss_rate']:.3f}")
    print("E3 label budget on cleaned targets")
    for n, methods in result["e3_budget"].items():
        for method, v in methods.items():
            s = v["source_rule"][0]
            print(f"  {n} [{method}, target {v['alpha']}, risk {v['risk']}] source rule {s['false_report_rate_mean']:.3f}, routed {s['share_routed_mean']:.3f}")
            print("     " + " | ".join(f"n={r['n']}: {r['false_report_rate_mean']:.3f}, ok {r['false_report_within_alpha']:.2f}, silent {r['reports_nothing']:.2f}, routed {r['share_routed_mean']:.2f}" for r in v["local"]))
    print("E4 miss-rate control for bare heads (highest hide threshold per target; None = not achievable)")
    for n, v in result["e4_miss"].items():
        print(f"  {n:17s} undetected {v['undetected_share']:.3f} | " + ", ".join(f"{b}: {v[b]}" for b in ("0.1", "0.2", "0.3", "0.5")))
    print("E5 preprocessing (source rule)")
    for n, rows in result["e5_stretch"].items():
        for label, v in rows.items():
            print(f"  {n:5s} {label:10s} photos {v['n_photos']} false-report {v['false_report_rate']:.3f} | {pc(v['by_class'])} | AP50 {pc(v['ap50'])}")


if __name__ == "__main__":
    main()
