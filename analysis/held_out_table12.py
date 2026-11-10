"""Held-out version of J1 Table 12. Design: HELD_OUT_TABLE12_DESIGN_2026-11-07.md, written first.

Run from the study repository root, with PYTHONDONTWRITEBYTECODE=1 and the repository on PYTHONPATH:
  PPE_VARIANT=rtdetr_r18 python held_out_table12.py reproduce|heldout
Writes only beside this script. Reads the saved hide_budget JSON and cached predictions; writes nothing to the repository.
"""
import hashlib
import json
import os
import sys

import numpy as np

from routing import core, datasets
from routing.experiments import ALL_IDS, MATCH
from routing.experiments_cv import clean_sets, fit_rule
from routing.folds import VARIANT
from routing.run_study import near_duplicates

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "HELD_OUT_TABLE12_DESIGN_2026-11-07.md")
NAME, BETA, N_SPLITS, SEED0 = "hard_hat_workers", 0.10, 200, 20261107
KEYS = ("share_routed", "routed_per_image", "miss_rate", "miss_rate_per_image", "false_report_rate")


def load():
    source, targets, log = clean_sets()
    rec = core.Records.from_images(source, class_ids=ALL_IDS)
    (cal, _, t_high), _ = fit_rule(rec)
    models = targets[NAME]
    recs = [core.Records.from_images(ims, class_ids=datasets.evaluable_ids(NAME), rule=MATCH[NAME]) for ims in models]
    return cal, t_high, models, recs, log


def reproduce():
    cal, t_high, _, recs, log = load()
    with open(os.path.join("outputs", "results", f"hide_budget_{VARIANT}.json"), encoding="utf8") as f:
        saved = json.load(f)
    rows, ok = [], abs(saved["t_high"] - t_high) < 1e-9
    for k, r in enumerate(recs):
        p = cal.predict(r.det_score)
        t = core.select_t_low(r, p, BETA)
        m = core.routing_metrics(r, p, t, t_high)
        s = saved["datasets"][NAME]["models"][k]
        diffs = {x: abs(m[x] - s[x]) for x in KEYS}
        diffs["t_low"] = abs(t - s["t_low"])
        ok &= max(diffs.values()) < 1e-9
        rows.append({"model": k, "t_low": t, **{x: m[x] for x in KEYS}, "max_abs_diff_to_saved": max(diffs.values())})
    out = {"variant": VARIANT, "t_high": t_high, "gate_passed": bool(ok), "n_photos": recs[0].n_images, "cleaning": log, "models": rows,
           "mean": {x: float(np.mean([r[x] for r in rows])) for x in KEYS},
           "t_low_range": [min(r["t_low"] for r in rows), max(r["t_low"] for r in rows)]}
    path = os.path.join(HERE, f"reproduce_{VARIANT}.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print(VARIANT, "gate_passed", ok, "t_high", t_high, "n", recs[0].n_images, "t_low", out["t_low_range"],
          {k: round(v, 4) for k, v in out["mean"].items()}, "max diff", max(r["max_abs_diff_to_saved"] for r in rows))
    return ok


def components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    return np.array([find(i) for i in range(n)])


def split_photos(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[: n // 2]), np.sort(order[n // 2:])


def split_groups(group, seed):
    """Equal-as-possible halves that never divide a near-duplicate group: shuffled groups fill the selection half to n//2."""
    n = len(group)
    labels = np.random.default_rng(seed).permutation(np.unique(group))
    sel, count = set(), 0
    sizes = np.bincount(np.unique(group, return_inverse=True)[1])
    index = {g: i for i, g in enumerate(np.unique(group))}
    for g in labels:
        s = sizes[index[g]]
        if count + s <= n // 2:
            sel.add(g)
            count += s
    mask = np.isin(group, list(sel))
    return np.where(mask)[0], np.where(~mask)[0]


def run(recs, cal, t_high, splitter):
    rows = []
    for k, r in enumerate(recs):
        for s in range(N_SPLITS):
            a, b = splitter(SEED0 + s)
            ra, rb = r.subset(a), r.subset(b)
            t = core.select_t_low(ra, cal.predict(ra.det_score), BETA)
            pb = cal.predict(rb.det_score)
            m = core.routing_metrics(rb, pb, t if t is not None else 0.05, t_high)
            rows.append({"model": k, "split": s, "t_low": t, "selected": t is not None, "n_select": ra.n_images, "n_eval": rb.n_images,
                         **{x: m[x] for x in KEYS}})
    return rows


def summary(rows):
    sel = [r for r in rows if r["selected"]]
    arr = lambda x, rs: np.array([r[x] for r in rs], dtype=float)
    within = lambda rs: float(np.mean(arr("miss_rate_per_image", rs) <= BETA)) if rs else None
    miss = arr("miss_rate_per_image", sel)
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r["selected"] and r["miss_rate_per_image"] <= BETA)
    tl = arr("t_low", sel)
    vals, counts = np.unique(tl, return_counts=True)
    out = {"pairs": len(rows), "no_cut_pairs": len(rows) - len(sel),
           "mean_miss_selected": float(miss.mean()), "p05_miss": float(np.percentile(miss, 5)), "p95_miss": float(np.percentile(miss, 95)),
           "share_within_all_pairs_nocut_fail": float(np.mean([r["selected"] and r["miss_rate_per_image"] <= BETA for r in rows])),
           "share_within_selected_pairs": within(sel),
           "share_splits_all_models_within": float(np.mean([all(v) for v in by_split.values()])),
           "t_low_min": float(tl.min()), "t_low_max": float(tl.max()), "t_low_mode": float(vals[counts.argmax()]),
           "mean_miss_with_nocut_at_floor": float(arr("miss_rate_per_image", rows).mean()),
           "mean_referrals_with_nocut_at_floor": float(arr("routed_per_image", rows).mean())}
    for x in ("routed_per_image", "share_routed", "false_report_rate"):
        v = arr(x, sel)
        out[x] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return out


def reading(result):
    """The reading table from the design, applied mechanically to one analysis."""
    size = result["size"]["mean_miss_selected"]
    models = [m["mean_miss_selected"] for m in result["by_model"]]
    if size > 0.105:
        label = "clear miss"
    elif size <= 0.10 and max(models) <= 0.10:
        label = "supported"
    else:
        label = "marginal"
    return {"label": label, "edge_of_feasibility": result["size"]["no_cut_pairs"] / result["size"]["pairs"] > 0.05}


def analyse(rows):
    res = {"size": summary(rows), "by_model": [{"model": k, **summary([r for r in rows if r["model"] == k])} for k in sorted({r["model"] for r in rows})]}
    res["reading"] = reading(res)
    return res


def heldout():
    cal, t_high, models, recs, log = load()
    n = recs[0].n_images
    pairs = [(i, j) for i, j, _ in near_duplicates(models[0], models[0], 4) if i < j]
    group = components(n, pairs)
    with open(DESIGN, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    photo = run(recs, cal, t_high, lambda s: split_photos(n, s))
    grouped = run(recs, cal, t_high, lambda s: split_groups(group, s))
    out = {"variant": VARIANT, "design_sha256": sha, "t_high": t_high, "n_photos": n, "n_splits": N_SPLITS, "seed0": SEED0,
           "within_dataset_near_duplicate_pairs": len(pairs), "near_duplicate_groups": int(len(np.unique(group))),
           "primary_photo_splits": analyse(photo), "sensitivity_near_duplicates_together": analyse(grouped)}
    with open(os.path.join(HERE, f"heldout_{VARIANT}.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(HERE, f"heldout_{VARIANT}_pairs.json"), "w", encoding="utf8") as f:
        json.dump({"primary": photo, "sensitivity": grouped}, f)
    print(VARIANT, "done", json.dumps({k: out[k] for k in ("n_photos", "within_dataset_near_duplicate_pairs", "near_duplicate_groups")}))


if __name__ == "__main__":
    if sys.argv[1] == "reproduce":
        sys.exit(0 if reproduce() else 1)
    heldout()
