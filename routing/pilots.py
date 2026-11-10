"""
Feasibility pilots for the journal paper, on the five-fold detections. Nothing here is a
paper result yet; the point is to learn whether each idea is worth building properly.

  python -m routing.pilots queue      P1  review queue that certifies itself
  python -m routing.pilots person     P2  person-anchored control of missed bare heads

P1  Photos arrive one at a time and a reviewer labels every detection. After each photo a
    sequential test (a betting e-process, valid at every photo by Ville's inequality) may
    certify a lower report threshold. Fixed-sequence over the threshold grid, several start
    points with a Bonferroni split, exactly the family structure of core.select_t_high.
    Compared with the fixed-sample test of the conference paper applied at a set of
    checkpoints, both used once (valid) and re-used at every checkpoint (peeking).
      photo risk       share of photos with at least one false report (target 0.20)
      detection risk   false reports among reports (target 0.10); the per-photo loss
                       FP - alpha * N must be bounded, so a cap on automatic reports per photo
                       is declared in advance and a photo above the cap goes to review whole

P2  A bare head the detector never proposes cannot be recovered by any threshold. A detected
    person can stand in for it: when no hard hat is found in the head zone of a detected
    person, the person is sent to review. Measured on labeled heads: how many bare heads
    are still missed, and how many helmeted workers are sent to review by mistake.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from . import core, datasets
from .experiments import ALL_IDS, ALPHA, ALPHA_PHOTO, DELTA, MATCH, T_LOW_DEFAULT, records
from .experiments_cv import across, clean_sets, fit_rule
from .folds import N_FOLDS, VARIANT
from .run_study import RESULT_DIR

GRID = np.round(np.arange(0.99, 0.495, -0.01), 2)
STARTS = (0.98, 0.95, 0.90, 0.80, 0.70)
CHECKPOINTS = (10, 25, 50, 100, 200, 400, 800)
REPORT_CAP = 10
HELMET, HEAD, VEST, PERSON = (core.CLASSES.index(c) for c in ("helmet", "head", "vest", "person"))


# ---------------------------------------------------------------- P1
def per_photo_counts(rec, p):
    """Reports and false reports per photo at every grid threshold: arrays (photos, thresholds)."""
    n = np.zeros((rec.n_images, len(GRID)))
    fp = np.zeros_like(n)
    for j, t in enumerate(GRID):
        rep = p >= t
        n[:, j] = np.bincount(rec.det_img[rep], minlength=rec.n_images)
        fp[:, j] = np.bincount(rec.det_img[rep & ~rec.det_tp], minlength=rec.n_images)
    return n, fp


def chains(certified_at):
    """certified_at: (thresholds,) first photo at which each threshold's own test passes (inf if never).
    Returns the photo at which each threshold becomes usable under fixed-sequence testing from each start."""
    usable = np.full(len(GRID), np.inf)
    for s in STARTS:
        start = int(np.argmax(GRID <= s + 1e-9))
        reached = 0.0
        for j in range(start, len(GRID)):
            reached = max(reached, certified_at[j])
            if not np.isfinite(reached):
                break
            usable[j] = min(usable[j], reached)
    return usable


def threshold_at(usable, n):
    """Lowest usable threshold index after n photos, or -1."""
    ok = np.where(usable <= n)[0]
    return int(ok.max()) if len(ok) else -1


def eprocess_times(x, alpha, level, bound):
    """x: (photos, thresholds) per-photo loss minus nothing; H0: mean loss > 0 where loss = x.
    Betting e-process K_n = mean over bets of prod(1 - lam * x_i / bound); first n with K_n >= 1/level."""
    lams = np.linspace(0.1, 0.9, 5)                 # bets as a share of the largest safe bet
    logk = np.zeros((len(lams),) + x.shape)
    for a, lam in enumerate(lams):
        logk[a] = np.cumsum(np.log1p(-lam * x / bound), axis=0)
    k = np.exp(logk).mean(axis=0)
    crossed = k >= 1.0 / level
    first = np.where(crossed.any(axis=0), crossed.argmax(axis=0) + 1, np.inf)
    return first


def fixed_sample_pass(x_cum, sq_cum, z_cum, n, alpha, level, risk):
    """Which thresholds pass the conference paper's fixed-sample test on the first n photos."""
    if risk == "photo":
        return stats.binom.cdf(z_cum[n - 1], n, alpha) <= level
    m = x_cum[n - 1] / n
    var = np.maximum(sq_cum[n - 1] / n - m ** 2, 0) * n / max(n - 1, 1)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        pval = np.where(sd > 0, stats.t.cdf(m / (sd / np.sqrt(n)), n - 1), np.where(m < 0, 0.0, 1.0))
    any_reports = z_cum[n - 1] > 0  # at least one photo with a report, as in core.false_report_pvalue
    return (pval <= level) & any_reports


def queue_runs(rec, p, risk, runs=100, n_max=800, seed=0, cap=REPORT_CAP):
    alpha = ALPHA_PHOTO if risk == "photo" else ALPHA
    level = DELTA / len(STARTS)
    n_i, fp_i = per_photo_counts(rec, p)
    routable = n_i[:, -1].sum()                      # everything at or above the hide threshold 0.50
    if risk == "detection":                          # crowded photos are reviewed whole, at every threshold
        over = n_i > cap
        n_i, fp_i = np.where(over, 0.0, n_i), np.where(over, 0.0, fp_i)
    if risk == "photo":
        loss, bound = (fp_i > 0).astype(float) - alpha, 1.0 - alpha       # H0: mean > 0
        truth = (fp_i > 0).mean(axis=0)
    else:
        loss, bound = fp_i - alpha * n_i, (1.0 - alpha) * cap
        truth = fp_i.sum(axis=0) / np.maximum(n_i.sum(axis=0), 1)
    share_reported = n_i.sum(axis=0) / max(routable, 1)
    bad = truth > alpha
    n_max = min(n_max, rec.n_images)
    cps = [c for c in CHECKPOINTS if c <= n_max]
    rng = np.random.default_rng(seed)
    out = {"seq": {c: [] for c in cps}, "once": {c: [] for c in cps}, "peek": {c: [] for c in cps},
           "seq_ever_bad": [], "peek_ever_bad": [], "seq_first_unlock": []}
    for _ in range(runs):
        order = rng.permutation(rec.n_images)[:n_max]
        x = loss[order]
        usable = chains(eprocess_times(x, alpha, level, bound))
        x_cum, sq_cum = np.cumsum(x, axis=0), np.cumsum(x ** 2, axis=0)
        z_cum = np.cumsum(n_i[order] > 0, axis=0) if risk != "photo" else np.cumsum(fp_i[order] > 0, axis=0)
        peek_best, peek_bad = -1, False
        for c in cps:
            j = threshold_at(usable, c)
            out["seq"][c].append(j)
            passed = fixed_sample_pass(x_cum, sq_cum, z_cum, c, alpha, level, risk)
            once = threshold_at(chains(np.where(passed, 0.0, np.inf)), 0)
            out["once"][c].append(once)
            peek_best = max(peek_best, once)
            peek_bad |= peek_best >= 0 and bool(bad[peek_best])
            out["peek"][c].append(peek_best)
        final = threshold_at(usable, n_max)
        ever = np.where(usable <= n_max)[0]
        out["seq_ever_bad"].append(bool(bad[ever].any()) if len(ever) else False)
        out["peek_ever_bad"].append(peek_bad)
        first = usable[np.isfinite(usable)]
        out["seq_first_unlock"].append(float(first.min()) if len(first) else np.inf)
    summary = {"alpha": alpha, "n_photos": rec.n_images, "cap": float(cap), "rows": []}
    for c in cps:
        row = {"n": c}
        for name in ("seq", "once", "peek"):
            js = np.array(out[name][c])
            has = js >= 0
            row[name] = {"reports": float(has.mean()),
                         "threshold_median": float(np.median(GRID[js[has]])) if has.any() else None,
                         "risk_mean": float(np.mean(np.where(has, truth[np.maximum(js, 0)], 0.0))),
                         "over_target": float(np.mean(has & bad[np.maximum(js, 0)])),
                         "auto_share": float(np.mean(np.where(has, share_reported[np.maximum(js, 0)], 0.0)))}
        summary["rows"].append(row)
    unlock = np.array(out["seq_first_unlock"])
    summary["seq_first_unlock_median"] = float(np.median(unlock)) if np.isfinite(np.median(unlock)) else None
    summary["seq_ever_over_target"] = float(np.mean(out["seq_ever_bad"]))
    summary["peek_ever_over_target"] = float(np.mean(out["peek_ever_bad"]))
    return summary


def pilot_queue(runs=100):
    source, targets, _ = clean_sets()
    src = core.Records.from_images(source, class_ids=ALL_IDS)
    (cal, _, _), _ = fit_rule(src)
    sets = {"source": [src], **{n: [records(n, ims) for ims in models] for n, models in targets.items()}}
    result = {}
    for name, recs in sets.items():
        result[name] = {}
        for risk in ("photo", "detection"):
            per_model = [queue_runs(r, cal.predict(r.det_score), risk, runs=runs if len(recs) > 1 else runs * N_FOLDS, seed=k) for k, r in enumerate(recs)]
            result[name][risk] = per_model
    with open(os.path.join(RESULT_DIR, f"pilot_queue_{VARIANT}.json" if VARIANT else "pilot_queue.json"), "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    report_queue(result)


def report_queue(result):
    avg = lambda vals: float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else float("nan")
    for name, risks in result.items():
        for risk, models in risks.items():
            m0 = models[0]
            unlock = [m["seq_first_unlock_median"] for m in models]
            print(f"P1 {name} [{risk} risk, target {m0['alpha']}, {m0['n_photos']} photos, cap {m0['cap']:.0f}] "
                  f"first unlock after {avg(unlock):.0f} photos (median) | ever over target: sequential {avg([m['seq_ever_over_target'] for m in models]):.2f}, "
                  f"fixed test re-used at every checkpoint {avg([m['peek_ever_over_target'] for m in models]):.2f}")
            for i, row in enumerate(m0["rows"]):
                cells = []
                for key, label in (("seq", "sequential"), ("once", "fixed, used once"), ("peek", "fixed, re-used")):
                    g = lambda field: avg([m["rows"][i][key][field] for m in models if i < len(m["rows"])])
                    cells.append(f"{label}: reports {g('reports'):.2f}, thr {g('threshold_median'):.2f}, risk {g('risk_mean'):.3f}, over {g('over_target'):.2f}, auto {g('auto_share'):.2f}")
                print(f"     n={row['n']:<4d} " + " | ".join(cells))


# ---------------------------------------------------------------- P2
def centers(boxes):
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def head_zones(person_boxes, top=0.35, widen=0.10):
    w = person_boxes[:, 2] - person_boxes[:, 0]
    h = person_boxes[:, 3] - person_boxes[:, 1]
    return np.stack([person_boxes[:, 0] - widen * w, person_boxes[:, 1] - 0.05 * h, person_boxes[:, 2] + widen * w, person_boxes[:, 1] + top * h], axis=1)


def in_zone(points, zones):
    """(points, zones) boolean."""
    return ((points[:, None, 0] >= zones[None, :, 0]) & (points[:, None, 0] <= zones[None, :, 2])
            & (points[:, None, 1] >= zones[None, :, 1]) & (points[:, None, 1] <= zones[None, :, 3]))


def person_pass(images, cal, match, class_ids, taus, person_floor, t_low=T_LOW_DEFAULT):
    """Outcome for every labeled bare head and helmet, under the detection rule and the person-anchored rule."""
    tot = {"heads": 0, "missed_det": 0, "undetected": 0, "anchored": 0, "helmets": 0, "helmets_anchored": 0, "photos": 0, "persons": 0,
           "routed_det": 0, **{f"missed_{t}": 0 for t in taus}, **{f"helmet_routed_{t}": 0 for t in taus}, **{f"flagged_{t}": 0 for t in taus}}
    for im in images:
        s = np.asarray(im["det_scores"], dtype=float); c = np.asarray(im["det_cls"], dtype=int)
        b = np.asarray(im["det_boxes"], dtype=float).reshape(-1, 4)
        keep = s >= 0.05
        s, c, b = s[keep], c[keep], b[keep]
        p = cal.predict(s)
        gb = np.asarray(im["gt_boxes"], dtype=float).reshape(-1, 4); gc = np.asarray(im["gt_cls"], dtype=int)
        gkeep = np.isin(gc, class_ids)
        scored = np.isin(c, class_ids)
        _, g = core.match_image(b[scored], s[scored], c[scored], gb[gkeep], gc[gkeep], 0.5, match)
        ps = p[scored]
        pg = np.array([ps[j] if j >= 0 else -1.0 for j in g], dtype=float)
        gcl = gc[gkeep]
        heads, helmets = gb[gkeep][gcl == HEAD], gb[gkeep][gcl == HELMET]
        p_head = pg[gcl == HEAD]
        tot["photos"] += 1
        tot["heads"] += len(heads); tot["helmets"] += len(helmets)
        tot["undetected"] += int((p_head < 0).sum())
        caught_det = p_head >= t_low
        tot["missed_det"] += int((~caught_det).sum())
        tot["routed_det"] += int(((p >= t_low) & scored).sum())
        persons = b[(c == PERSON) & (p >= person_floor)]
        tot["persons"] += len(persons)
        if len(persons) == 0:
            for t in taus:
                tot[f"missed_{t}"] += int((~caught_det).sum())
            continue
        zones = head_zones(persons)
        zc = centers(zones)

        def owner(boxes):
            """Index of the detected person whose head zone holds each box center (nearest zone center), -1 if none."""
            if len(boxes) == 0:
                return np.zeros(0, dtype=int)
            pts = centers(boxes)
            inside = in_zone(pts, zones)
            dist = np.linalg.norm(pts[:, None, :] - zc[None, :, :], axis=2)
            dist[~inside] = np.inf
            return np.where(inside.any(axis=1), dist.argmin(axis=1), -1)

        det_h = c == HELMET
        own_det = owner(b[det_h])
        own_head, own_helm = owner(heads), owner(helmets)
        tot["anchored"] += int((own_head >= 0).sum()); tot["helmets_anchored"] += int((own_helm >= 0).sum())
        for t in taus:
            evidence = np.zeros(len(persons))
            for o, score in zip(own_det, p[det_h]):
                if o >= 0:
                    evidence[o] = max(evidence[o], score)
            flagged = evidence < t
            tot[f"flagged_{t}"] += int(flagged.sum())
            caught = caught_det | ((own_head >= 0) & flagged[np.maximum(own_head, 0)])
            tot[f"missed_{t}"] += int((~caught).sum())
            tot[f"helmet_routed_{t}"] += int(((own_helm >= 0) & flagged[np.maximum(own_helm, 0)]).sum())
    return tot


def pilot_person(taus=(0.3, 0.5, 0.7, 0.9), floors=(0.3, 0.5)):
    source, targets, _ = clean_sets()
    src = core.Records.from_images(source, class_ids=ALL_IDS)
    (cal, _, _), _ = fit_rule(src)
    sets = {"source": ([source], "iou", ALL_IDS), **{n: (models, MATCH[n], datasets.evaluable_ids(n)) for n, models in targets.items()}}
    result = {}
    for name, (models, match, ids) in sets.items():
        result[name] = {}
        for floor in floors:
            rows = [person_pass(ims, cal, match, ids, taus, floor) for ims in models]
            r = {"n_heads": rows[0]["heads"], "n_helmets": rows[0]["helmets"], "n_photos": rows[0]["photos"]}
            div = lambda a, b_: [x[a] / max(x[b_], 1) for x in rows]
            r["miss_detection_rule"] = across(div("missed_det", "heads"))
            r["undetected_any_confidence"] = across(div("undetected", "heads"))
            r["bare_heads_inside_a_detected_person"] = across(div("anchored", "heads"))
            r["helmets_inside_a_detected_person"] = across(div("helmets_anchored", "helmets"))
            r["routed_detections_per_photo"] = across(div("routed_det", "photos"))
            r["persons_per_photo"] = across(div("persons", "photos"))
            for t in taus:
                r[f"tau_{t}"] = {"miss": across(div(f"missed_{t}", "heads")), "helmeted_sent_to_review": across(div(f"helmet_routed_{t}", "helmets")),
                                 "flagged_persons_per_photo": across(div(f"flagged_{t}", "photos"))}
            result[name][str(floor)] = r
    with open(os.path.join(RESULT_DIR, f"pilot_person_{VARIANT}.json" if VARIANT else "pilot_person.json"), "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    f3 = lambda s: "n/a" if s is None else f"{s['mean']:.3f} ({s['min']:.3f} to {s['max']:.3f})"
    for name, floors_ in result.items():
        for floor, r in floors_.items():
            print(f"P2 {name} [person floor {floor}] bare heads {r['n_heads']}, helmets {r['n_helmets']}, photos {r['n_photos']}")
            print(f"     detection rule: missed {f3(r['miss_detection_rule'])}, never detected {f3(r['undetected_any_confidence'])}, routed detections per photo {f3(r['routed_detections_per_photo'])}")
            print(f"     anchoring: bare heads inside a detected person {f3(r['bare_heads_inside_a_detected_person'])}, helmets {f3(r['helmets_inside_a_detected_person'])}, persons per photo {f3(r['persons_per_photo'])}")
            for t in taus:
                v = r[f"tau_{t}"]
                print(f"     clear a person only with hard-hat evidence >= {t}: missed {f3(v['miss'])} | helmeted workers sent to review {f3(v['helmeted_sent_to_review'])} | flagged persons per photo {f3(v['flagged_persons_per_photo'])}")


# ---------------------------------------------------------------- P3
def pilot_class(reps=50, budgets=(25, 50, 100, 200, 400, 800), budget_reps=40, seed=0):
    """P3  Certify each class on its own instead of the pooled false-report rate.

    Part 1, transfer: per-class report thresholds are certified on a random half of the source
    photos (same test and targets as the pooled rule) and applied unchanged to the other half
    and to every target. Part 2, label budget: the report threshold of one class is reselected
    on n local photos. "auto" is the share of that class's visible detections (at or above the
    hide threshold) that are reported without review.
    """
    source, targets, _ = clean_sets()
    names = ("source", *targets)
    ids_of = {"source": ALL_IDS, **{n: datasets.evaluable_ids(n) for n in targets}}

    def class_records(name, images, c):
        return core.Records.from_images(images, class_ids=(c,), rule="iou" if name == "source" else MATCH[name])

    src_all = core.Records.from_images(source, class_ids=ALL_IDS)
    src_c = {c: class_records("source", source, c) for c in ALL_IDS}
    tgt_c = {n: {c: [class_records(n, ims, c) for ims in models] for c in ids_of[n]} for n, models in targets.items()}
    rng = np.random.default_rng(seed)
    rows = {n: {c: {"frr": [], "auto": []} for c in ids_of[n]} for n in names}
    chosen = {c: [] for c in ALL_IDS}

    def outcome(rec, p, th):
        m = core.routing_metrics(rec, p, T_LOW_DEFAULT, th)
        return m["false_report_rate"], m["n_report"] / max(1, m["n_report"] + m["n_route"])

    for _ in range(reps):
        perm = rng.permutation(src_all.n_images)
        half = src_all.n_images // 2
        fit_all = src_all.subset(perm[:half])
        cal = core.make_calibrator("platt").fit(fit_all.det_score, fit_all.det_tp)
        for c in ALL_IDS:
            fit, held = src_c[c].subset(perm[:half]), src_c[c].subset(perm[half:])
            th = core.select_t_high(fit, cal.predict(fit.det_score), ALPHA, DELTA)
            chosen[c].append(th)
            cases = [("source", held)] + [(n, r) for n in targets if c in ids_of[n] for r in tgt_c[n][c]]
            for n, rec in cases:
                frr, auto = outcome(rec, cal.predict(rec.det_score), th)
                rows[n][c]["frr"].append(frr if th is not None else 0.0); rows[n][c]["auto"].append(auto)
    transfer = {"reps": reps, "alpha": ALPHA, "thresholds": {}, "datasets": {}}
    for c in ALL_IDS:
        got = [t for t in chosen[c] if t is not None]
        transfer["thresholds"][core.CLASSES[c]] = {"none_certified": float(np.mean([t is None for t in chosen[c]])),
                                                   "median": float(np.median(got)) if got else None}
    for n in names:
        transfer["datasets"][n] = {core.CLASSES[c]: {"false_report_rate": float(np.mean(v["frr"])), "within_target": float(np.mean(np.array(v["frr"]) <= ALPHA)),
                                                     "auto_share": float(np.mean(v["auto"]))} for c, v in rows[n].items()}

    (cal, _, _), _ = fit_rule(src_all)
    budget = {}
    for n in names:
        budget[n] = {}
        for c in ids_of[n]:
            recs = [src_c[c]] if n == "source" else tgt_c[n][c]
            th_src = core.select_t_high(src_c[c], cal.predict(src_c[c].det_score), ALPHA, DELTA)
            all_rows, src_frr = [], []
            for k, rec in enumerate(recs):
                res = core.budget_curve(None, rec, budgets, reps=budget_reps * (N_FOLDS if n == "source" else 1), alpha=ALPHA, beta=0.10, delta=DELTA,
                                        method="cluster", rule=(cal, T_LOW_DEFAULT, th_src), seed=seed + k, refit=())
                all_rows += res["rows"]
                src_frr.append(outcome(rec, cal.predict(rec.det_score), th_src)[0] if th_src is not None else None)
            out = []
            for b in sorted({r["n"] for r in all_rows}):
                rs = [r for r in all_rows if r["n"] == b]
                frr = np.array([r["target_false_report_rate"] for r in rs])
                out.append({"n": b, "false_report_rate": float(frr.mean()), "within_target": float((frr <= ALPHA).mean()),
                            "silent": float(np.mean([r["target_t_high"] is None for r in rs])),
                            "auto_share": float(np.mean([r["target_n_report"] / max(1, r["target_n_report"] + r["target_n_route"]) for r in rs]))})
            budget[n][core.CLASSES[c]] = {"source_threshold": th_src, "source_rule_false_report_rate": across(src_frr), "local": out}
    result = {"transfer": transfer, "budget": budget}
    with open(os.path.join(RESULT_DIR, f"pilot_class_{VARIANT}.json" if VARIANT else "pilot_class.json"), "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    print("P3 per-class certificates, certified on a source half (median threshold, share of halves where none could be certified)")
    print("   " + " | ".join(f"{c}: {('none' if v['median'] is None else format(v['median'], '.2f'))}, none in {v['none_certified']:.2f}" for c, v in transfer["thresholds"].items()))
    for n, classes in transfer["datasets"].items():
        print(f"   {n:17s} " + " | ".join(f"{c}: {v['false_report_rate']:.3f}, within {v['within_target']:.2f}, auto {v['auto_share']:.2f}" for c, v in classes.items()))
    print("P3 label budget per class (threshold reselected on n local photos)")
    for n, classes in budget.items():
        for c, v in classes.items():
            s0 = v["source_rule_false_report_rate"]
            head = f"   {n:17s} {c:7s} source class rule {'none' if v['source_threshold'] is None else format(v['source_threshold'], '.2f')}" + ("" if s0 is None else f" gives {s0['mean']:.3f}")
            print(head + " || " + " | ".join(f"n={r['n']}: {r['false_report_rate']:.3f}, ok {r['within_target']:.2f}, silent {r['silent']:.2f}, auto {r['auto_share']:.2f}" for r in v["local"]))


if __name__ == "__main__":
    {"queue": pilot_queue, "person": pilot_person, "class": pilot_class}[sys.argv[1]]()
