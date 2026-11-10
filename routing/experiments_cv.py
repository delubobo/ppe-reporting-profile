"""
Paper experiments with the five cross-validation models. Run after `python -m routing.folds`
for the source photos and every target.

  python -m routing.experiments_cv         results in outputs/results/experiments_cv.json

Source baseline: every source photo is scored by the one model that did not train on it, and
those out-of-fold detections are pooled. Calibration and the report threshold are fitted on
that pool (cross-fitting). Each target is scored by all five models, and every target number
is reported as the mean over models with the range across them.

A1  Detector accuracy: AP50 by class, out-of-fold on source and per model on each target.
A2  Reference: the published v1 model against the fold models on the photos v1 held out.
E1  Transfer. Fit the rule on a random half of the pooled source photos, score the other
    half and every target under each model. Repeated over random halves.
E1b One model at a time: rule fitted on the photos held out from model k only, applied to
    the targets scored by model k. No pooling across models.
E2  Sensitivity of the whole-head datasets to the containment matching parameters.
E3  Label budget on the cleaned targets, detection-level and photo-level tests.
E4  Whether any hide threshold can meet a miss-rate target for the violation class.
E6  Label budget on the source data itself: the same procedure with no dataset shift.
E7  Practice baselines on the same random halves as E1: a fixed raw-score threshold of 0.5, and
    the F1-optimal raw-score threshold chosen on the fitting half, each used as a single
    report-or-discard threshold, against the certified rule's report band.

Cleaning
  source   a photo is dropped when a near-duplicate of it sits in another fold, because the
           model that held it out trained on the duplicate
  targets  photos that near-duplicate any source photo are dropped; duplicates between
           targets are dropped from the lower-priority set (CHV > Hard Hat Workers > SHWD)
"""
import json
import os
import re
import sys

import numpy as np

from . import core, datasets
from .experiments import ALL_IDS, ALPHA, ALPHA_PHOTO, BETA, DELTA, MATCH, T_LOW_DEFAULT, TARGETS, class_rates, records, spread
from .folds import N_FOLDS, VARIANT, load_fold_cache
from .run_study import RESULT_DIR, near_duplicates

BUDGETS = (10, 25, 50, 100, 200, 400)
RESULT_FILE = os.path.join(RESULT_DIR, f"experiments_cv_{VARIANT}.json" if VARIANT else "experiments_cv.json")


def photo_group(path):
    """Coarse provenance of a source photo, read from its file name."""
    name = os.path.basename(path).split(".rf.")[0]
    if re.search(r"_mp4-\d+", name) or name.lower().startswith("youtube"):
        return "video frame"
    return "still"


def target_names():
    """The registered targets whose fold caches exist for the current detector size."""
    from .folds import FOLD_CACHE
    extra = [n for n in ("sh17",) if os.path.exists(os.path.join(FOLD_CACHE, f"{n}_{VARIANT + '_' if VARIANT else ''}f0.pkl"))]
    return tuple(TARGETS) + tuple(extra)


def clean_sets(bits=4):
    by_model = [load_fold_cache("css_v30", k) for k in range(N_FOLDS)]
    photos = by_model[0]
    fold = np.array([im["fold"] for im in photos])
    pairs = [(i, j) for i, j, _ in near_duplicates(photos, photos, bits) if i != j and fold[i] != fold[j]]
    leaked = {i for pair in pairs for i in pair}
    keep = [i for i in range(len(photos)) if i not in leaked]
    source = [by_model[fold[i]][i] for i in keep]                      # out-of-fold detections
    log = {"source_photos": len(photos), "source_dropped_duplicate_in_another_fold": len(leaked), "source_kept": len(source),
           "source_kept_by_fold": [int((fold[keep] == k).sum()) for k in range(N_FOLDS)],
           "source_video_frames_kept": int(sum(photo_group(im["path"]) == "video frame" for im in source))}
    targets, kept_ref = {}, {}
    for name in target_names():
        models = [load_fold_cache(name, k) for k in range(N_FOLDS)]
        images = models[0]
        drop = set()
        for other_name, other in [("source", photos)] + list(kept_ref.items()):
            hits = {j for _, j, _ in near_duplicates(other, images, bits)}
            log[f"{name}_dropped_vs_{other_name}"] = len(hits - drop)
            drop |= hits
        idx = [j for j in range(len(images)) if j not in drop]
        targets[name] = [[m[j] for j in idx] for m in models]
        kept_ref[name] = targets[name][0]
        log[f"{name}_kept"] = len(idx)
    return source, targets, log


def across(values):
    """Mean and range over the five models."""
    v = np.array([x for x in values if x is not None and x == x], dtype=float)
    return None if len(v) == 0 else {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def fit_rule(rec, method="cluster", alpha=ALPHA):
    cal = core.make_calibrator("platt").fit(rec.det_score, rec.det_tp)
    p = cal.predict(rec.det_score)
    t_high = core.select_t_high(rec, p, alpha, DELTA, method=method)
    t_low = core.select_t_low(rec, p, BETA)
    info = {"method": method, "alpha": alpha, "n_photos": rec.n_images, "platt_a": cal.a, "platt_b": cal.b, "t_high": t_high,
            "t_low_certified": t_low, "t_low_used": T_LOW_DEFAULT if t_low is None else t_low,
            "ece_raw": core.ece(rec.det_score, rec.det_tp)[0], "ece_calibrated": core.ece(p, rec.det_tp)[0]}
    return (cal, info["t_low_used"], t_high), info


def a1_accuracy(source, targets):
    out = {}
    rec = core.Records.from_images(source, floor=0.01, class_ids=ALL_IDS)
    folds = np.array([im["fold"] for im in source])
    out["source"] = {"n_photos": rec.n_images, "pooled": {core.CLASSES[i]: core.average_precision(rec, i) for i in ALL_IDS},
                     "n_gt": {core.CLASSES[i]: int((rec.gt_cls == i).sum()) for i in ALL_IDS}, "by_fold": {}}
    per_fold = [rec.subset(np.where(folds == k)[0]) for k in range(N_FOLDS)]
    for i in ALL_IDS:
        out["source"]["by_fold"][core.CLASSES[i]] = across([core.average_precision(r, i) for r in per_fold])
    for n, models in targets.items():
        ids = datasets.evaluable_ids(n)
        recs = [core.Records.from_images(ims, floor=0.01, class_ids=ids, rule=MATCH[n]) for ims in models]
        out[n] = {"n_photos": recs[0].n_images, "n_gt": {core.CLASSES[i]: int((recs[0].gt_cls == i).sum()) for i in ids},
                  "by_model": {core.CLASSES[i]: across([core.average_precision(r, i) for r in recs]) for i in ids}}
    return out


def a2_reference():
    """v1 was trained on the pre-augmented export and its checkpoint was picked on these photos."""
    v1 = load_fold_cache("css_v30", "v1")
    by_model = [load_fold_cache("css_v30", k) for k in range(N_FOLDS)]
    held = [i for i, im in enumerate(v1) if os.path.normpath(im["path"]).split(os.sep)[-2] in ("valid", "test")]
    ap = lambda ims: {core.CLASSES[i]: core.average_precision(core.Records.from_images(ims, floor=0.01, class_ids=ALL_IDS), i) for i in ALL_IDS}
    return {"n_photos": len(held), "v1": ap([v1[i] for i in held]), "fold_models_out_of_fold": ap([by_model[v1[i]["fold"]][i] for i in held])}


def a3_density(src, tgt, cal, thresholds=(0.57, 0.78)):
    """Reports per photo and the share of photos with at least one report: what a photo-level risk scales with."""
    def one(rec, t):
        rep = cal.predict(rec.det_score) >= t
        n = np.bincount(rec.det_img[rep], minlength=rec.n_images)
        return {"reports_per_photo": float(rep.sum() / rec.n_images), "photos_with_a_report": float((n > 0).mean()),
                "labeled_objects_per_photo": float(len(rec.gt_cls) / rec.n_images)}
    out = {}
    for t in thresholds:
        out[str(t)] = {"source": one(src, t)}
        for n, recs in tgt.items():
            rows = [one(r, t) for r in recs]
            out[str(t)][n] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


def e1_transfer(src, tgt, reps=50, method="cluster", seed=0, alpha=ALPHA, risk="false_report_rate"):
    """risk names the quantity the rule is certified for; "within target" is judged on it."""
    rng = np.random.default_rng(seed)
    keys = ("false_report_rate", "false_alarm_images", "share_routed", "miss_rate")
    rows = {n: {k: [] for k in keys} for n in ("source", *tgt)}
    per_model = {n: [[] for _ in range(N_FOLDS)] for n in tgt}
    per_class = {n: {} for n in ("source", *tgt)}
    ece = {n: {"raw": [], "calibrated": []} for n in ("source", *tgt)}
    chosen = {"t_high": [], "no_report_threshold": 0}
    for _ in range(reps):
        perm = rng.permutation(src.n_images)
        fit, held = src.subset(perm[: src.n_images // 2]), src.subset(perm[src.n_images // 2:])
        cal = core.make_calibrator("platt").fit(fit.det_score, fit.det_tp)
        th = core.select_t_high(fit, cal.predict(fit.det_score), alpha, DELTA, method=method)
        if th is None:
            chosen["no_report_threshold"] += 1
            continue
        chosen["t_high"].append(th)
        cases = [("source", None, held)] + [(n, k, rec) for n, recs in tgt.items() for k, rec in enumerate(recs)]
        for n, k, rec in cases:
            p = cal.predict(rec.det_score)
            m = core.routing_metrics(rec, p, T_LOW_DEFAULT, th)
            for key in keys:
                rows[n][key].append(m[key])
            if k is not None:
                per_model[n][k].append(m[risk])
            ece[n]["raw"].append(core.ece(rec.det_score, rec.det_tp)[0]); ece[n]["calibrated"].append(core.ece(p, rec.det_tp)[0])
            ids = ALL_IDS if n == "source" else datasets.evaluable_ids(n)
            for c, v in class_rates(rec, p, th, ids).items():
                per_class[n].setdefault(c, []).append(v["false_report_rate"])
    out = {"method": method, "reps": reps, "alpha": alpha, "risk": risk, "delta": DELTA, "fit_photos": src.n_images // 2,
           "chosen": {"t_high": spread(chosen["t_high"]), "no_report_threshold": chosen["no_report_threshold"]}, "datasets": {}}
    for n in rows:
        d = {k: spread(v) for k, v in rows[n].items()}
        d["by_class_false_report"] = {c: spread(v) for c, v in per_class[n].items()}
        d["ece_raw"], d["ece_calibrated"] = spread(ece[n]["raw"]), spread(ece[n]["calibrated"])
        d["share_of_runs_within_alpha"] = float(np.mean(np.array(rows[n][risk]) <= alpha))
        if n in per_model:
            d["false_report_by_model"] = [float(np.mean(v)) for v in per_model[n]]
            d["within_alpha_by_model"] = [float(np.mean(np.array(v) <= alpha)) for v in per_model[n]]
        out["datasets"][n] = d
    return out


def f1_threshold(rec, grid=None):
    """Raw-score threshold that maximizes F1 over all classes on rec (the common practice)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    n_gt = len(rec.gt_cls)
    best, best_t = -1.0, 0.5
    for t in grid:
        rep_ = rec.det_score >= t
        tp = int(rec.det_tp[rep_].sum()); n_rep = int(rep_.sum())
        prec = tp / n_rep if n_rep else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t


def e7_baselines(src, tgt, reps=50, seed=0):
    """Fixed 0.5 and F1-optimal raw-score thresholds against the certified rule, on E1's random halves."""
    rng = np.random.default_rng(seed)
    names = ("source", *tgt)
    keys = ("false_report_rate", "false_alarm_images", "miss_rate", "recall")
    rows = {m: {n: {k: [] for k in keys} for n in names} for m in ("fixed_0.5", "f1_optimal", "certified")}
    by_class = {m: {n: {} for n in names} for m in rows}
    chosen = {"f1_optimal": [], "certified": []}

    def outcome(rec, score_like, t):
        rep_ = score_like >= t
        n_rep = int(rep_.sum())
        tp = int(rec.det_tp[rep_].sum())
        m = core.routing_metrics(rec, score_like, t, t)       # hide = report threshold: a single threshold
        return {"false_report_rate": m["false_report_rate"], "false_alarm_images": m["false_alarm_images"], "miss_rate": m["miss_rate"],
                "recall": tp / max(1, len(rec.gt_cls))}

    for _ in range(reps):
        perm = rng.permutation(src.n_images)
        fit, held = src.subset(perm[: src.n_images // 2]), src.subset(perm[src.n_images // 2:])
        cal = core.make_calibrator("platt").fit(fit.det_score, fit.det_tp)
        th = core.select_t_high(fit, cal.predict(fit.det_score), ALPHA, DELTA, method="cluster")
        t_f1 = f1_threshold(fit)
        chosen["f1_optimal"].append(t_f1)
        if th is not None:
            chosen["certified"].append(th)
        cases = [("source", None, held)] + [(n, k, rec) for n, recs in tgt.items() for k, rec in enumerate(recs)]
        for n, k, rec in cases:
            ids = ALL_IDS if n == "source" else datasets.evaluable_ids(n)
            for method, score_like, t in (("fixed_0.5", rec.det_score, 0.5), ("f1_optimal", rec.det_score, t_f1), ("certified", cal.predict(rec.det_score), th)):
                if t is None:
                    continue
                o = outcome(rec, score_like, t)
                for key in keys:
                    rows[method][n][key].append(o[key])
                for c, v in class_rates(rec, score_like, t, ids).items():
                    by_class[method][n].setdefault(c, []).append(v["false_report_rate"])
    out = {"reps": reps, "alpha": ALPHA, "thresholds": {k: spread(v) for k, v in chosen.items()}, "methods": {}}
    for method in rows:
        out["methods"][method] = {}
        for n in names:
            d = {k: spread(v) for k, v in rows[method][n].items()}
            d["within_alpha"] = float(np.mean(np.array(rows[method][n]["false_report_rate"]) <= ALPHA)) if rows[method][n]["false_report_rate"] else None
            d["by_class_false_report"] = {c: spread(v) for c, v in by_class[method][n].items()}
            out["methods"][method][n] = d
    return out


def e1b_one_model(source, targets):
    """No pooling: each model's rule comes from its own held-out photos."""
    folds = np.array([im["fold"] for im in source])
    out = {"models": [], "targets": {n: [] for n in targets}}
    for k in range(N_FOLDS):
        rec = core.Records.from_images([im for im, f in zip(source, folds) if f == k], class_ids=ALL_IDS)
        (cal, tl, th), info = fit_rule(rec)
        out["models"].append({"n_photos": info["n_photos"], "t_high": th})
        for n, models in targets.items():
            r = records(n, models[k])
            out["targets"][n].append(None if th is None else core.routing_metrics(r, cal.predict(r.det_score), tl, th)["false_report_rate"])
    out["summary"] = {n: across(v) for n, v in out["targets"].items()}
    return out


def e2_containment(targets, rule):
    cal, tl, th = rule
    out = {}
    for n in ("hard_hat_workers", "shwd"):
        ids = datasets.evaluable_ids(n)
        out[n] = {}
        for label, match in (("iou 0.5", "iou"), ("inside 0.5, iou 0.1", ("contain", 0.5, 0.1)), ("inside 0.7, iou 0.2", ("contain", 0.7, 0.2)),
                             ("inside 0.7, iou 0.3", ("contain", 0.7, 0.3)), ("inside 0.9, iou 0.2", ("contain", 0.9, 0.2))):
            frr, miss, by_class = [], [], {}
            for ims in targets[n]:
                rec = records(n, ims, match)
                p = cal.predict(rec.det_score)
                m = core.routing_metrics(rec, p, tl, th)
                frr.append(m["false_report_rate"]); miss.append(m["miss_rate"])
                for c, v in class_rates(rec, p, th, ids).items():
                    by_class.setdefault(c, []).append(v["false_report_rate"])
            out[n][label] = {"false_report_rate": across(frr), "miss_rate": across(miss), "by_class": {c: across(v) for c, v in by_class.items()}}
    return out


def budget(recs, rules, reps, seed=1):
    """Label-budget curves for a list of record sets (one per model), pooled over models."""
    out = {}
    for method, alpha, risk in (("cluster", ALPHA, "false_report_rate"), ("image", ALPHA_PHOTO, "false_alarm_images")):
        rows, src_rate, src_routed, by_model = [], [], [], []
        for k, rec in enumerate(recs):
            res = core.budget_curve(None, rec, BUDGETS, reps=reps, alpha=alpha, beta=BETA, delta=DELTA, method=method,
                                    rule=rules[method], seed=seed + k, refit=())
            rows += res["rows"]
            s = core.summarize_budget(res, "source", risk)
            src_rate.append(s[0]["false_report_rate_mean"]); src_routed.append(s[0]["share_routed_mean"])
            by_model.append([r["false_report_rate_mean"] for r in core.summarize_budget(res, "target", risk)])
        pooled = {"alpha": alpha, "beta": BETA, "rows": rows}
        local = core.summarize_budget(pooled, "target", risk)
        for j, row in enumerate(local):
            row["false_report_rate_by_model"] = across([m[j] for m in by_model])
        out[method] = {"alpha": alpha, "risk": risk, "local": local, "source_rule": {"false_report_rate": across(src_rate), "share_routed": across(src_routed)}}
    return out


def e4_miss_feasibility(src, targets, rule):
    cal = rule[0]
    out = {}

    def one(rec):
        p = cal.predict(rec.det_score)
        return {"undetected_share": core.routing_metrics(rec, p, 0.0, None)["miss_rate"],
                "miss_at_hide_threshold": core.routing_metrics(rec, p, rule[1], rule[2])["miss_rate"],
                **{str(b): core.select_t_low(rec, p, b) for b in (0.1, 0.2, 0.3, 0.5)}}

    out["source"] = one(src)
    for n, models in targets.items():
        recs = [records(n, ims) for ims in models]
        if not np.isin(recs[0].gt_cls, core.VIOLATION_IDS).any():
            continue
        rows = [one(r) for r in recs]
        out[n] = {"undetected_share": across([r["undetected_share"] for r in rows]),
                  "miss_at_hide_threshold": across([r["miss_at_hide_threshold"] for r in rows]),
                  **{b: [r[b] for r in rows] for b in ("0.1", "0.2", "0.3", "0.5")}}
    return out


def main(reps_budget=60):
    source, targets, log = clean_sets()
    src = core.Records.from_images(source, class_ids=ALL_IDS)
    tgt = {n: [records(n, ims) for ims in models] for n, models in targets.items()}
    rule, info = fit_rule(src, "cluster", ALPHA)
    rule_photo, info_photo = fit_rule(src, "image", ALPHA_PHOTO)
    rules = {"cluster": rule, "image": rule_photo}
    result = {"cleaning": log, "source_rule": info, "source_rule_photo": info_photo,
              "a1_accuracy": a1_accuracy(source, targets),
              "a2_reference": a2_reference(),
              "a3_density": a3_density(src, tgt, rule[0]),
              "e1_cluster": e1_transfer(src, tgt),
              "e1_photo": e1_transfer(src, tgt, method="image", alpha=ALPHA_PHOTO, risk="false_alarm_images"),
              "e1b_one_model": e1b_one_model(source, targets),
              "e2_containment": e2_containment(targets, rule),
              "e3_budget": {n: budget(recs, rules, reps_budget) for n, recs in tgt.items()},
              "e4_miss": e4_miss_feasibility(src, targets, rule),
              "e6_source_budget": budget([src], rules, reps_budget * N_FOLDS),
              "e7_baselines": e7_baselines(src, tgt)}
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    report(result)


def report(result):
    f3 = lambda s: "n/a" if s is None else (f"{s['mean']:.3f} [{s['p05']:.3f}, {s['p95']:.3f}]" if "p05" in s else f"{s['mean']:.3f} ({s['min']:.3f} to {s['max']:.3f})")
    rnd = lambda d: json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
    print("CLEANING", json.dumps(result["cleaning"]))
    print("SOURCE RULE", rnd(result["source_rule"]))
    print("SOURCE RULE (photo-level)", rnd(result["source_rule_photo"]))
    print("A1 AP50")
    a = result["a1_accuracy"]
    print(f"  source ({a['source']['n_photos']} photos, out-of-fold, pooled): " + ", ".join(f"{c} {v:.3f}" for c, v in a["source"]["pooled"].items()))
    print("     by fold: " + ", ".join(f"{c} {f3(v)}" for c, v in a["source"]["by_fold"].items()) + " | labels " + json.dumps(a["source"]["n_gt"]))
    for n in TARGETS:
        print(f"  {n} ({a[n]['n_photos']} photos): " + ", ".join(f"{c} {f3(v)}" for c, v in a[n]["by_model"].items()) + " | labels " + json.dumps(a[n]["n_gt"]))
    r = result["a2_reference"]
    print(f"A2 AP50 on the {r['n_photos']} photos v1 held out: v1 " + ", ".join(f"{c} {v:.3f}" for c, v in r["v1"].items())
          + " | fold models, out-of-fold " + ", ".join(f"{c} {v:.3f}" for c, v in r["fold_models_out_of_fold"].items()))
    for t, rows in result.get("a3_density", {}).items():
        print(f"A3 at report threshold {t}: " + " | ".join(f"{n} {v['reports_per_photo']:.2f} reports per photo, {v['photos_with_a_report']:.2f} of photos with a report, {v['labeled_objects_per_photo']:.2f} labeled objects per photo" for n, v in rows.items()))
    for key in ("e1_cluster", "e1_photo"):
      if key not in result:
        continue
      e1 = result[key]
      print(f"E1 transfer [{e1['method']} test, target {e1['alpha']} on {e1.get('risk', 'false_report_rate')}], {e1['reps']} random halves of {e1['fit_photos']} photos: report threshold {f3(e1['chosen']['t_high'])}, none certified in {e1['chosen']['no_report_threshold']} runs")
      for n, d in e1["datasets"].items():
        print(f"  {n:17s} false-report {f3(d['false_report_rate'])} within target {d['share_of_runs_within_alpha']:.2f} | photos with a false report {f3(d['false_alarm_images'])} | routed {f3(d['share_routed'])} | missed heads {f3(d['miss_rate'])} | ECE raw {f3(d['ece_raw'])} cal {f3(d['ece_calibrated'])}")
        print("                    by class: " + ", ".join(f"{c} {f3(v)}" for c, v in d["by_class_false_report"].items()))
        if "false_report_by_model" in d:
            print("                    certified risk by model: " + ", ".join(f"{v:.3f}" for v in d["false_report_by_model"]) + " | within target: " + ", ".join(f"{v:.2f}" for v in d["within_alpha_by_model"]))
    b = result["e1b_one_model"]
    print("E1b one model at a time: thresholds " + ", ".join(f"{m['t_high']} (n={m['n_photos']})" for m in b["models"]))
    for n, v in b["targets"].items():
        print(f"  {n:17s} " + ", ".join("none" if x is None else f"{x:.3f}" for x in v) + f" | {f3(b['summary'][n])}")
    print("E2 containment sensitivity (source rule, mean and range over models)")
    for n, rows in result["e2_containment"].items():
        for label, v in rows.items():
            print(f"  {n:17s} {label:22s} false-report {f3(v['false_report_rate'])} | " + ", ".join(f"{c} {f3(x)}" for c, x in v["by_class"].items()) + f" | missed heads {f3(v['miss_rate'])}")

    def budget_lines(name, methods):
        for method, v in methods.items():
            s = v["source_rule"]
            print(f"  {name} [{method}, target {v['alpha']}, risk {v['risk']}] source rule {f3(s['false_report_rate'])}, routed {f3(s['share_routed'])}")
            print("     " + " | ".join(f"n={r['n']}: {r['false_report_rate_mean']:.3f}, ok {r['false_report_within_alpha']:.2f}, silent {r['reports_nothing']:.2f}, routed {r['share_routed_mean']:.2f}" for r in v["local"]))

    print("E3 label budget on cleaned targets (pooled over models)")
    for n, methods in result["e3_budget"].items():
        budget_lines(n, methods)
    print("E4 miss-rate control for bare heads (highest hide threshold per target; None = not achievable)")
    for n, v in result["e4_miss"].items():
        und = v["undetected_share"]
        print(f"  {n:17s} undetected {und if isinstance(und, float) else f3(und)} | missed at hide threshold {v['miss_at_hide_threshold'] if isinstance(v['miss_at_hide_threshold'], float) else f3(v['miss_at_hide_threshold'])} | " + ", ".join(f"{b}: {v[b]}" for b in ("0.1", "0.2", "0.3", "0.5")))
    print("E6 label budget on the source data (no shift)")
    budget_lines("source", result["e6_source_budget"])
    if "e7_baselines" in result:
        e7 = result["e7_baselines"]
        print(f"E7 practice baselines on the E1 halves: F1-optimal raw threshold {f3(e7['thresholds']['f1_optimal'])}, certified calibrated threshold {f3(e7['thresholds']['certified'])}")
        for method, sets in e7["methods"].items():
            for n, d in sets.items():
                print(f"  {method:10s} {n:17s} false-report {f3(d['false_report_rate'])} within target {d['within_alpha']:.2f} | photos with a false report {f3(d['false_alarm_images'])} | recall {f3(d['recall'])} | missed heads {f3(d['miss_rate'])}")
                print("                               by class: " + ", ".join(f"{c} {f3(v)}" for c, v in d["by_class_false_report"].items()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        with open(RESULT_FILE, encoding="utf8") as f:
            report(json.load(f))
    elif len(sys.argv) > 1 and sys.argv[1] == "e7":   # add the practice baselines to an existing result file
        source, targets, _ = clean_sets()
        src = core.Records.from_images(source, class_ids=ALL_IDS)
        tgt = {n: [records(n, ims) for ims in models] for n, models in targets.items()}
        with open(RESULT_FILE, encoding="utf8") as f:
            result = json.load(f)
        result["e7_baselines"] = e7_baselines(src, tgt)
        with open(RESULT_FILE, "w", encoding="utf8") as f:
            json.dump(result, f, indent=2)
        report(result)
    elif len(sys.argv) > 1 and sys.argv[1] == "e1photo":   # add the photo-level transfer run to an existing result file
        source, targets, _ = clean_sets()
        src = core.Records.from_images(source, class_ids=ALL_IDS)
        tgt = {n: [records(n, ims) for ims in models] for n, models in targets.items()}
        with open(RESULT_FILE, encoding="utf8") as f:
            result = json.load(f)
        result["e1_photo"] = e1_transfer(src, tgt, method="image", alpha=ALPHA_PHOTO, risk="false_alarm_images")
        result["a3_density"] = a3_density(src, tgt, fit_rule(src)[0][0])
        with open(RESULT_FILE, "w", encoding="utf8") as f:
            json.dump(result, f, indent=2)
        report(result)
    else:
        main()
