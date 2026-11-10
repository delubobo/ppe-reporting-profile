"""
Markdown tables for Journal 1 (the audit), built from outputs/results/experiments_cv*.json so
that every number in the draft is traceable to a run on disk.

  python -m routing.tables [sizes...]      default: nano s m (l is added when its file exists)
"""
import json
import os
import sys

from .run_study import RESULT_DIR

LABEL = {"nano": "Nano", "s": "S", "m": "M", "l": "L", "rtdetr_r18": "RT-DETR r18", "rtdetr_r50": "RT-DETR r50",
         "rtdetr_r18nms": "RT-DETR r18, NMS", "rtdetr_r50nms": "RT-DETR r50, NMS"}
SETS = (("source", "Source held-out"), ("chv", "CHV"), ("hard_hat_workers", "Hard Hat Workers"), ("shwd", "SHWD"), ("sh17", "SH17"))


def load(size):
    name = "experiments_cv.json" if size == "nano" else f"experiments_cv_{size}.json"
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf8") as f:
        return json.load(f)


def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}"


def row(cells):
    return "| " + " | ".join(cells) + " |"


def table(header, rows):
    return "\n".join([row(header), row(["---"] * len(header))] + [row(r) for r in rows]) + "\n"


def main(sizes):
    results = {s: load(s) for s in sizes}
    results = {s: r for s, r in results.items() if r}
    sizes = list(results)
    out = []

    # A. detector accuracy
    out.append("**Table A. Detector accuracy, AP50 by class (source out-of-fold; targets mean over five models).**\n")
    rows = []
    for key, name in SETS:
        for size in sizes:
            if key not in results[size]["a1_accuracy"]:
                continue
            a = results[size]["a1_accuracy"][key]
            vals = a["pooled"] if key == "source" else {c: v["mean"] for c, v in a["by_model"].items()}
            rows.append([name if size == sizes[0] else "", LABEL[size]] + [f"{vals[c]:.2f}" if c in vals else "not labeled" for c in ("helmet", "head", "vest", "person")])
    out.append(table(["Dataset", "Size", "Hard hat", "Bare head", "Vest", "Person"], rows))

    # B. aggregate certificate
    out.append("**Table B. Rule certified on source halves for at most 10% false reports over all classes, applied unchanged. Mean (5th to 95th percentile) over 50 halves and five models; share of runs within target.**\n")
    rows = []
    for key, name in SETS:
        for size in sizes:
            if key not in results[size]["e1_cluster"]["datasets"]:
                continue
            d = results[size]["e1_cluster"]["datasets"][key]
            f = d["false_report_rate"]
            rows.append([name if size == sizes[0] else "", LABEL[size], f"{pct(f['mean'])} ({pct(f['p05'])} to {pct(f['p95'])})", pct(d["share_of_runs_within_alpha"], 0),
                         pct(d["share_routed"]["mean"]), pct(d["miss_rate"]["mean"]) if d["miss_rate"] else "not labeled"])
    out.append(table(["Dataset", "Size", "False-report rate (%)", "Runs within target (%)", "Routed share (%)", "Bare heads missed (%)"], rows))

    # C. by class
    out.append("**Table C. False-report rate by class under the same rule (%).**\n")
    rows = []
    for key, name in SETS:
        for size in sizes:
            if key not in results[size]["e1_cluster"]["datasets"]:
                continue
            bc = results[size]["e1_cluster"]["datasets"][key]["by_class_false_report"]
            rows.append([name if size == sizes[0] else "", LABEL[size]] + [pct(bc[c]["mean"]) if c in bc and bc[c] else "not labeled" for c in ("helmet", "head", "vest", "person")])
    out.append(table(["Dataset", "Size", "Hard hat", "Bare head", "Vest", "Person"], rows))

    # D. photo level
    out.append("**Table D. Rule certified on source halves so that at most 20% of photos carry a false report (exact test), applied unchanged.**\n")
    rows = []
    for key, name in SETS:
        for size in sizes:
            r = results[size]
            if "e1_photo" not in r:
                continue
            if key not in r["e1_photo"]["datasets"]:
                continue
            d = r["e1_photo"]["datasets"][key]
            f = d["false_alarm_images"]
            dens = r.get("a3_density", {}).get("0.57", {}).get(key, {})
            rows.append([name if size == sizes[0] else "", LABEL[size], f"{pct(f['mean'])} ({pct(f['p05'])} to {pct(f['p95'])})", pct(d["share_of_runs_within_alpha"], 0),
                         f"{dens.get('reports_per_photo', float('nan')):.1f}", pct(dens.get("photos_with_a_report"), 0)])
    out.append(table(["Dataset", "Size", "Photos with a false report (%)", "Runs within target (%)", "Reports per photo", "Photos with any report (%)"], rows))

    # E. misses
    out.append("**Table E. Missed bare heads under the source rule (hide threshold 0.50), the share with no detection at any confidence, and the lowest miss target that conformal risk control can certify.**\n")
    rows = []
    for key, name in (("source", "Source held-out"), ("hard_hat_workers", "Hard Hat Workers"), ("shwd", "SHWD"), ("sh17", "SH17")):
        for size in sizes:
            if key not in results[size]["e4_miss"]:
                continue
            e = results[size]["e4_miss"][key]
            miss = e["miss_at_hide_threshold"]; und = e["undetected_share"]
            miss = miss["mean"] if isinstance(miss, dict) else miss
            und = und["mean"] if isinstance(und, dict) else und
            lowest = "none"
            for b in ("0.1", "0.2", "0.3", "0.5"):
                v = e[b]
                ok = (v is not None) if not isinstance(v, list) else all(x is not None for x in v)
                if ok:
                    lowest = f"{int(float(b) * 100)}%"
                    break
            rows.append([name if size == sizes[0] else "", LABEL[size], pct(miss), pct(und), lowest])
    out.append(table(["Dataset", "Size", "Missed (%)", "Never detected (%)", "Lowest certifiable miss target (all five models)"], rows))

    # F. practice baselines
    out.append("**Table F. Practice baselines on the same halves: a fixed raw-score threshold of 0.5 and the F1-optimal raw-score threshold, each as a single report-or-discard threshold, against the certified rule's report band. False-report rate (%) and share of runs within the 10% target.**\n")
    rows = []
    for key, name in SETS:
        for size in sizes:
            r = results[size]
            if "e7_baselines" not in r:
                continue
            cells = [name if size == sizes[0] else "", LABEL[size]]
            for method in ("fixed_0.5", "f1_optimal", "certified"):
                if key not in r["e7_baselines"]["methods"][method]:
                    cells.append("n/a"); continue
                d = r["e7_baselines"]["methods"][method][key]
                cells.append(f"{pct(d['false_report_rate']['mean'])} / {pct(d['within_alpha'], 0)}")
            rows.append(cells)
    out.append(table(["Dataset", "Size", "Fixed 0.5", "F1-optimal", "Certified"], rows))

    text = "\n".join(out)
    path = os.path.join("paper", "journal1_tables.md")
    with open(path, "w", encoding="utf8") as f:
        f.write(f"# Journal 1 tables (generated by `python -m routing.tables`; sizes: {', '.join(LABEL[s] for s in sizes)})\n\n" + text)
    print(text)
    print("written to", path)


if __name__ == "__main__":
    main(sys.argv[1:] or ["nano", "s", "m", "l"])
