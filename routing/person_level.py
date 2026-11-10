"""
ISARC 2027 analysis: a certified bound on missed bare heads through person-anchored routing.

  python -m routing.person_level            (PPE_VARIANT selects the detector size, as elsewhere)

Rule. A detected person (confidence at or above a floor) is cleared when a hard hat is detected
in its head zone with calibrated confidence at or above tau; otherwise the person is sent to
review. A labeled bare head counts as caught when a bare-head detection at or above the hide
threshold matched it (the detection-level path) or when it lies in the head zone of a person
sent to review (the person-level path). The loss of a photo is the share of its bare heads
that are caught by neither path. Raising tau sends more persons to review, so the loss is
monotone non-increasing in tau, and conformal risk control picks the lowest tau whose
corrected empirical risk meets the target. The price is the share of helmeted workers whose
person is sent to review, and the number of flagged persons per photo.

Experiments
  P1  Transfer: tau certified on the source photos for miss targets of 0.10, 0.20 and 0.30,
      applied unchanged to each target dataset under each model.
  P2  Local certification: tau certified on n local photos that contain a bare head, evaluated
      on the remaining photos; repeated draws.
  Both are reported next to the detection-level hide threshold (core.select_t_low), which is
  the rule without person anchoring.
"""
import json
import os
import sys

import numpy as np

from . import core, datasets
from .experiments import ALL_IDS, MATCH, T_LOW_DEFAULT
from .experiments_cv import across, clean_sets, fit_rule
from .folds import N_FOLDS, VARIANT
from .pilots import HEAD, HELMET, PERSON, centers, head_zones, in_zone
from .run_study import RESULT_DIR

TAUS = np.round(np.arange(0.05, 1.0001, 0.05), 2)      # 1.0 means every detected person goes to review
BETAS = (0.10, 0.20, 0.30)
PERSON_FLOOR = 0.3
BUDGETS = (25, 50, 100, 200, 400)


def photo_table(images, cal, match, class_ids, person_floor=PERSON_FLOOR, t_low=T_LOW_DEFAULT, top=0.35, widen=0.10):
    """Per photo: labeled bare heads, those missed at each tau, labeled helmets, helmets whose
    person is flagged at each tau, flagged persons at each tau, and the detection-level miss
    at each candidate hide threshold (for the baseline)."""
    rows = {"heads": [], "missed": [], "helmets": [], "helmet_routed": [], "flagged": [], "persons": [], "missed_det_grid": []}
    det_grid = np.round(np.arange(0.05, 0.905, 0.01), 2)
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
        caught_det = p_head >= t_low
        rows["heads"].append(len(heads)); rows["helmets"].append(len(helmets))
        rows["missed_det_grid"].append((p_head[:, None] < det_grid[None, :]).sum(axis=0))
        persons = b[(c == PERSON) & (p >= person_floor)]
        rows["persons"].append(len(persons))
        if len(persons) == 0:
            rows["missed"].append(np.full(len(TAUS), int((~caught_det).sum())))
            rows["helmet_routed"].append(np.zeros(len(TAUS), dtype=int)); rows["flagged"].append(np.zeros(len(TAUS), dtype=int))
            continue
        zones = head_zones(persons, top=top, widen=widen)
        zc = centers(zones)

        def owner(boxes):
            if len(boxes) == 0:
                return np.zeros(0, dtype=int)
            pts = centers(boxes)
            inside = in_zone(pts, zones)
            dist = np.linalg.norm(pts[:, None, :] - zc[None, :, :], axis=2)
            dist[~inside] = np.inf
            return np.where(inside.any(axis=1), dist.argmin(axis=1), -1)

        det_h = c == HELMET
        own_det = owner(b[det_h]); own_head, own_helm = owner(heads), owner(helmets)
        evidence = np.zeros(len(persons))
        for o, score in zip(own_det, p[det_h]):
            if o >= 0:
                evidence[o] = max(evidence[o], score)
        flagged = evidence[None, :] < TAUS[:, None]                         # (taus, persons)
        anchored_head = own_head >= 0
        caught = caught_det[None, :] | (anchored_head[None, :] & flagged[:, np.maximum(own_head, 0)])
        rows["missed"].append((~caught).sum(axis=1))
        anchored_helm = own_helm >= 0
        rows["helmet_routed"].append((anchored_helm[None, :] & flagged[:, np.maximum(own_helm, 0)]).sum(axis=1))
        rows["flagged"].append(flagged.sum(axis=1))
    return {k: np.array(v) for k, v in rows.items()}


def certify_tau(table, idx, beta):
    """Lowest tau whose corrected empirical photo-level miss risk is at or below beta, on photos idx that hold a bare head."""
    heads = table["heads"][idx]
    use = heads > 0
    n = int(use.sum())
    if n == 0:
        return None
    risk = (table["missed"][idx][use] / heads[use, None]).mean(axis=0)   # (taus,)
    ok = np.where(n / (n + 1) * risk + 1 / (n + 1) <= beta)[0]
    return float(TAUS[ok[0]]) if len(ok) else None


def evaluate(table, idx, tau):
    """Outcomes on photos idx at tau (None means no person-level routing: detection path only)."""
    j = int(np.argmin(np.abs(TAUS - tau))) if tau is not None else None
    heads = table["heads"][idx]; use = heads > 0
    missed = table["missed"][idx][:, j] if j is not None else table["missed"][idx][:, 0] * 0 + (table["heads"][idx] - (table["heads"][idx] - table["missed_det_grid"][idx][:, np.argmin(np.abs(np.round(np.arange(0.05, 0.905, 0.01), 2) - T_LOW_DEFAULT))]))
    out = {"miss_rate": float(missed.sum() / max(1, heads.sum())),
           "miss_rate_per_photo": float((missed[use] / heads[use]).mean()) if use.any() else float("nan")}
    if j is not None:
        out["helmeted_sent_to_review"] = float(table["helmet_routed"][idx][:, j].sum() / max(1, table["helmets"][idx].sum()))
        out["flagged_persons_per_photo"] = float(table["flagged"][idx][:, j].mean())
        out["share_of_persons_flagged"] = float(table["flagged"][idx][:, j].sum() / max(1, table["persons"][idx].sum()))
    return out


def detection_baseline(table, idx, beta):
    """The hide threshold the detection-level rule would certify for the same target, and its miss rate (None if not certifiable)."""
    grid = np.round(np.arange(0.05, 0.905, 0.01), 2)
    heads = table["heads"][idx]; use = heads > 0
    n = int(use.sum())
    if n == 0:
        return None, None
    risk = (table["missed_det_grid"][idx][use] / heads[use, None]).mean(axis=0)
    ok = np.where(n / (n + 1) * risk + 1 / (n + 1) <= beta)[0]
    if not len(ok):
        return None, None
    j = ok[-1]                                                               # highest hide threshold that meets the target
    return float(grid[j]), float(table["missed_det_grid"][idx][:, j].sum() / max(1, heads.sum()))


def main(reps=100, seed=0):
    source, targets, _ = clean_sets()
    src_rec = core.Records.from_images(source, class_ids=ALL_IDS)
    (cal, _, _), _ = fit_rule(src_rec)
    tables = {"source": [photo_table(source, cal, "iou", ALL_IDS)]}
    for n, models in targets.items():
        if n == "chv":
            continue                                                          # no bare-head labels
        tables[n] = [photo_table(ims, cal, MATCH[n], datasets.evaluable_ids(n)) for ims in models]
    result = {"variant": VARIANT or "nano", "person_floor": PERSON_FLOOR, "taus": TAUS.tolist(), "transfer": {}, "local": {}, "ceiling": {}, "curves": {}}

    # tau curves for the figure: miss rate, helmeted workers reviewed, flagged persons per photo, mean over models
    for n, tabs in tables.items():
        miss = np.mean([t["missed"].sum(axis=0) / max(1, t["heads"].sum()) for t in tabs], axis=0)
        helm = np.mean([t["helmet_routed"].sum(axis=0) / max(1, t["helmets"].sum()) for t in tabs], axis=0)
        flagged = np.mean([t["flagged"].mean(axis=0) for t in tabs], axis=0)
        result["curves"][n] = {"miss_rate": miss.tolist(), "helmeted_sent_to_review": helm.tolist(), "flagged_persons_per_photo": flagged.tolist()}

    # anchoring ceiling: every detected person sent to review (tau = 1.0)
    for n, tabs in tables.items():
        all_idx = [np.arange(len(t["heads"])) for t in tabs]
        result["ceiling"][n] = {"miss_rate": across([evaluate(t, i, 1.0)["miss_rate"] for t, i in zip(tabs, all_idx)]),
                                "miss_rate_per_photo": across([evaluate(t, i, 1.0)["miss_rate_per_photo"] for t, i in zip(tabs, all_idx)]),
                                "detection_rule_miss_rate": across([evaluate(t, i, None)["miss_rate"] for t, i in zip(tabs, all_idx)]),
                                "detection_rule_miss_rate_per_photo": across([evaluate(t, i, None)["miss_rate_per_photo"] for t, i in zip(tabs, all_idx)])}

    # P1 transfer
    src_t = tables["source"][0]; src_all = np.arange(len(src_t["heads"]))
    for beta in BETAS:
        tau = certify_tau(src_t, src_all, beta)
        det_t, det_miss = detection_baseline(src_t, src_all, beta)
        row = {"tau": tau, "source": evaluate(src_t, src_all, tau) if tau is not None else None,
               "detection_hide_threshold": det_t, "detection_source_miss": det_miss, "targets": {}}
        for n, tabs in tables.items():
            if n == "source":
                continue
            per = [evaluate(t, np.arange(len(t["heads"])), tau) for t in tabs] if tau is not None else []
            row["targets"][n] = None if tau is None else {
                "miss_rate": across([x["miss_rate"] for x in per]), "miss_rate_per_photo": across([x["miss_rate_per_photo"] for x in per]),
                "within_target": float(np.mean([x["miss_rate_per_photo"] <= beta for x in per])),
                "helmeted_sent_to_review": across([x["helmeted_sent_to_review"] for x in per]),
                "flagged_persons_per_photo": across([x["flagged_persons_per_photo"] for x in per])}
        result["transfer"][str(beta)] = row

    # P2 local certification
    rng = np.random.default_rng(seed)
    for n, tabs in tables.items():
        result["local"][n] = {}
        for beta in BETAS:
            rows = []
            for b in BUDGETS:
                recs = []
                for t in tabs:
                    with_heads = np.where(t["heads"] > 0)[0]
                    if b >= len(with_heads):
                        continue
                    for _ in range(reps if len(tabs) > 1 else reps * N_FOLDS):
                        perm = rng.permutation(with_heads)
                        cal_idx, test_idx = perm[:b], np.setdiff1d(np.arange(len(t["heads"])), perm[:b])
                        tau = certify_tau(t, cal_idx, beta)
                        det_t, _ = detection_baseline(t, cal_idx, beta)
                        e = evaluate(t, test_idx, tau) if tau is not None else None
                        recs.append({"tau": tau, "det_t": det_t, "e": e})
                if not recs:
                    continue
                got = [r for r in recs if r["tau"] is not None]
                rows.append({"n": b, "certified": float(len(got) / len(recs)), "detection_rule_certified": float(np.mean([r["det_t"] is not None for r in recs])),
                             "tau_median": float(np.median([r["tau"] for r in got])) if got else None,
                             "miss_rate_per_photo": float(np.mean([r["e"]["miss_rate_per_photo"] for r in got])) if got else None,
                             "within_target": float(np.mean([r["e"]["miss_rate_per_photo"] <= beta for r in got])) if got else None,
                             "miss_rate": float(np.mean([r["e"]["miss_rate"] for r in got])) if got else None,
                             "helmeted_sent_to_review": float(np.mean([r["e"]["helmeted_sent_to_review"] for r in got])) if got else None,
                             "flagged_persons_per_photo": float(np.mean([r["e"]["flagged_persons_per_photo"] for r in got])) if got else None})
            result["local"][n][str(beta)] = rows

    name = f"isarc_person_{VARIANT}.json" if VARIANT else "isarc_person.json"
    with open(os.path.join(RESULT_DIR, name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    report(result)


def report(r):
    f3 = lambda s: "n/a" if s is None else f"{s['mean']:.3f} ({s['min']:.3f} to {s['max']:.3f})"
    print(f"PERSON-LEVEL MISS CONTROL [{r['variant']}, person floor {r['person_floor']}]")
    print("Ceiling (every detected person reviewed) against the detection rule at hide threshold 0.5, miss rate of bare heads:")
    for n, v in r["ceiling"].items():
        print(f"  {n:17s} detection rule {f3(v['detection_rule_miss_rate'])} | all persons reviewed {f3(v['miss_rate'])}"
              + (f" | per photo: detection rule {f3(v['detection_rule_miss_rate_per_photo'])}, all persons reviewed {f3(v['miss_rate_per_photo'])}" if 'miss_rate_per_photo' in v else ""))
    print("P1 transfer: tau certified on source photos, applied unchanged")
    for beta, row in r["transfer"].items():
        if row["tau"] is None:
            print(f"  target {beta}: not certifiable on source (detection rule: {'not certifiable' if row['detection_hide_threshold'] is None else row['detection_hide_threshold']})")
            continue
        s = row["source"]
        print(f"  target {beta}: tau {row['tau']:.2f} | source miss {s['miss_rate']:.3f} (per photo {s['miss_rate_per_photo']:.3f}), helmeted reviewed {s['helmeted_sent_to_review']:.3f}, flagged per photo {s['flagged_persons_per_photo']:.2f}"
              f" | detection rule: {'not certifiable' if row['detection_hide_threshold'] is None else f'hide {row[chr(100)+chr(101)+chr(116)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(104)+chr(105)+chr(100)+chr(101)+chr(95)+chr(116)+chr(104)+chr(114)+chr(101)+chr(115)+chr(104)+chr(111)+chr(108)+chr(100)]:.2f}, miss {row[chr(100)+chr(101)+chr(116)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(115)+chr(111)+chr(117)+chr(114)+chr(99)+chr(101)+chr(95)+chr(109)+chr(105)+chr(115)+chr(115)]:.3f}'}")
        for n, v in row["targets"].items():
            print(f"     {n:17s} miss {f3(v['miss_rate'])} | per photo {f3(v['miss_rate_per_photo'])}, within target {v['within_target']:.2f} | helmeted reviewed {f3(v['helmeted_sent_to_review'])} | flagged per photo {f3(v['flagged_persons_per_photo'])}")
    print("P2 local certification on n photos that hold a bare head (mean over draws and models)")
    for n, betas in r["local"].items():
        for beta, rows in betas.items():
            cells = []
            for x in rows:
                if x["tau_median"] is None:
                    cells.append(f"n={x['n']}: certified {x['certified']:.2f} (detection rule {x['detection_rule_certified']:.2f})")
                else:
                    cells.append(f"n={x['n']}: certified {x['certified']:.2f} (detection rule {x['detection_rule_certified']:.2f}), tau {x['tau_median']:.2f}, miss {x['miss_rate']:.3f}, per photo {x['miss_rate_per_photo']:.3f}, within {x['within_target']:.2f}, helmeted reviewed {x['helmeted_sent_to_review']:.3f}, flagged/photo {x['flagged_persons_per_photo']:.2f}")
            print(f"  {n:17s} target {beta}: " + " | ".join(cells))


def sensitivity(tau=0.10):
    """Head-zone geometry on Hard Hat Workers: ceiling and the outcome at a fixed tau, mean over models."""
    source, targets, _ = clean_sets()
    src_rec = core.Records.from_images(source, class_ids=ALL_IDS)
    (cal, _, _), _ = fit_rule(src_rec)
    rows = []
    for top, widen in ((0.35, 0.10), (0.30, 0.10), (0.40, 0.10), (0.35, 0.0), (0.35, 0.20), (0.50, 0.10)):
        tabs = [photo_table(ims, cal, MATCH["hard_hat_workers"], datasets.evaluable_ids("hard_hat_workers"), top=top, widen=widen) for ims in targets["hard_hat_workers"]]
        full = [np.arange(len(t["heads"])) for t in tabs]
        ceil = [evaluate(t, i, 1.0)["miss_rate"] for t, i in zip(tabs, full)]
        at = [evaluate(t, i, tau) for t, i in zip(tabs, full)]
        rows.append({"top": top, "widen": widen, "ceiling_miss": across(ceil), "miss_at_tau": across([x["miss_rate"] for x in at]),
                     "helmeted_sent_to_review": across([x["helmeted_sent_to_review"] for x in at])})
        r = rows[-1]
        print(f"  top {top:.2f} widen {widen:.2f}: ceiling miss {r['ceiling_miss']['mean']:.3f} | at tau {tau}: miss {r['miss_at_tau']['mean']:.3f}, helmeted reviewed {r['helmeted_sent_to_review']['mean']:.3f}", flush=True)
    name = f"isarc_zone_sensitivity_{VARIANT}.json" if VARIANT else "isarc_zone_sensitivity.json"
    with open(os.path.join(RESULT_DIR, name), "w", encoding="utf8") as f:
        json.dump({"variant": VARIANT or "nano", "tau": tau, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sensitivity":
        print(f"HEAD-ZONE SENSITIVITY on Hard Hat Workers [{VARIANT or 'nano'}]")
        sensitivity()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        name = f"isarc_person_{VARIANT}.json" if VARIANT else "isarc_person.json"
        with open(os.path.join(RESULT_DIR, name), encoding="utf8") as f:
            report(json.load(f))
    else:
        main()
