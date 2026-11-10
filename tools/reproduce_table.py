"""Reproduce the Source held-out and Hard Hat Workers rows of manuscript Table 5 from the released files.

  python tools/reproduce_table.py

Table 5 reports the source-half aggregate reporting rule at alpha = 0.10, applied unchanged, for
each YOLOX size: false-report rate (mean, 5th to 95th percentile), share of runs within target and
routed share. The values come from the e1_cluster block of results/experiments_cv[_<size>].json,
which routing.experiments_cv.e1_transfer writes. This script rebuilds that block from the
released detections only:

  1. load detections/css_v30 and detections/hard_hat_workers for the size;
  2. apply data/duplicates: drop every source photo listed in css_v30_vs_css_v30.csv, and every
     Hard Hat Workers photo listed as file_b in any list;
  3. take each source photo's detections from the model that held it out (out-of-fold);
  4. build records with routing.core.Records.from_images and routing.experiments.records;
  5. fit the full-source rule with routing.experiments_cv.fit_rule (checked against source_rule
     in the saved JSON) and run routing.experiments_cv.e1_transfer with its defaults.

The CHV, SHWD and SH17 rows need detections that are not released in this version. The script
prints the recomputed table, the saved JSON cells and the published cells, and exits with code 1
if any cell differs at the printed precision.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from routing import core  # noqa: E402
from routing.experiments import ALL_IDS, ALPHA, records  # noqa: E402
from routing.experiments_cv import e1_transfer, fit_rule  # noqa: E402

SIZES = ("nano", "s", "m", "l")
LABEL = {"nano": "Nano", "s": "S", "m": "M", "l": "L"}
RESULT = {"nano": "experiments_cv.json", "s": "experiments_cv_s.json", "m": "experiments_cv_m.json", "l": "experiments_cv_l.json"}
ROWS = (("source", "Source held-out"), ("hard_hat_workers", "Hard Hat Workers"))
# Table 5 of the submitted manuscript: false-report rate (%), runs within target (%), routed share (%).
PUBLISHED = {
    "source": {"nano": ("7.2 (4.5 to 10.0)", "94", "19"), "s": ("7.4 (5.3 to 10.0)", "94", "16"),
               "m": ("7.4 (4.4 to 11.1)", "90", "13"), "l": ("7.8 (4.9 to 10.4)", "86", "13")},
    "hard_hat_workers": {"nano": ("6.3 (3.5 to 8.5)", "100", "23"), "s": ("5.3 (3.9 to 6.5)", "100", "15"),
                         "m": ("6.3 (4.0 to 8.8)", "100", "11"), "l": ("6.0 (4.4 to 7.4)", "100", "9")},
}


def unpack(path):
    """Photo dicts in the layout routing.core.Records.from_images reads."""
    with np.load(path) as z:
        z = {k: z[k] for k in z.files}
    go, do = z["gt_offsets"], z["det_offsets"]
    rows = []
    for i, name in enumerate(z["file"]):
        im = {"file": str(name), "gt_boxes": z["gt_boxes"][go[i]:go[i + 1]], "gt_cls": z["gt_cls"][go[i]:go[i + 1]],
              "det_boxes": z["det_boxes"][do[i]:do[i + 1]], "det_scores": z["det_scores"][do[i]:do[i + 1]],
              "det_cls": z["det_cls"][do[i]:do[i + 1]]}
        if "fold" in z:
            im["fold"] = int(z["fold"][i])
        rows.append(im)
    return rows


def exclusions():
    source, hhw = set(), set()
    for path in sorted(glob.glob(os.path.join(PKG, "data", "duplicates", "*.csv"))):
        with open(path, newline="", encoding="utf8") as f:
            for r in csv.DictReader(f):
                if r["dataset_a"] == "css_v30" and r["dataset_b"] == "css_v30":
                    source |= {r["file_a"], r["file_b"]}
                elif r["dataset_b"] == "hard_hat_workers":
                    hhw.add(r["file_b"])
    return source, hhw


def cells(d):
    f = d["false_report_rate"]
    return (f"{100 * f['mean']:.1f} ({100 * f['p05']:.1f} to {100 * f['p95']:.1f})",
            f"{100 * d['share_of_runs_within_alpha']:.0f}", f"{100 * d['share_routed']['mean']:.0f}")


def main():
    drop_src, drop_hhw = exclusions()
    with open(os.path.join(PKG, "data", "duplicates", "summary.json"), encoding="utf8") as f:
        cleaning = json.load(f)["cleaning"]
    print(f"exclusion lists: {len(drop_src)} source photos, {len(drop_hhw)} Hard Hat Workers photos")
    recomputed, saved, failures, max_diff = {}, {}, [], 0.0
    for size in SIZES:
        by_model = [unpack(os.path.join(PKG, "detections", "css_v30", f"yolox_{size}_fold{k}.npz")) for k in range(5)]
        photos = by_model[0]
        source = [by_model[im["fold"]][i] for i, im in enumerate(photos) if im["file"] not in drop_src]
        models = [[im for im in unpack(os.path.join(PKG, "detections", "hard_hat_workers", f"yolox_{size}_fold{k}.npz"))
                   if im["file"] not in drop_hhw] for k in range(5)]
        for name, n, key in (("source", len(source), "source_kept"), ("hard_hat_workers", len(models[0]), "hard_hat_workers_kept")):
            if n != cleaning[key]:
                failures.append(f"{size}: {name} photos after cleaning {n}, saved {cleaning[key]}")
        src = core.Records.from_images(source, class_ids=ALL_IDS)
        tgt = {"hard_hat_workers": [records("hard_hat_workers", ims) for ims in models]}
        with open(os.path.join(PKG, "results", RESULT[size]), encoding="utf8") as f:
            result = json.load(f)
        _, info = fit_rule(src, "cluster", ALPHA)
        ref = result["source_rule"]
        for k in ("n_photos", "t_high", "t_low_used"):
            if info[k] != ref[k]:
                failures.append(f"{size}: fit_rule {k} {info[k]} against saved {ref[k]}")
        for k in ("platt_a", "platt_b"):
            if f"{info[k]:.6f}" != f"{ref[k]:.6f}":
                failures.append(f"{size}: fit_rule {k} {info[k]:.6f} against saved {ref[k]:.6f}")
        print(f"{LABEL[size]}: {len(source)} source photos, {len(models[0])} Hard Hat Workers photos; full-source rule "
              f"t_high {info['t_high']} (saved {ref['t_high']}), Platt a {info['platt_a']:.6f} b {info['platt_b']:.6f}", flush=True)
        e1 = e1_transfer(src, tgt)
        for name, _ in ROWS:
            mine, theirs = e1["datasets"][name], result["e1_cluster"]["datasets"][name]
            recomputed[(name, size)], saved[(name, size)] = cells(mine), cells(theirs)
            for a, b in ((mine["false_report_rate"], theirs["false_report_rate"]), (mine["share_routed"], theirs["share_routed"])):
                max_diff = max(max_diff, *(abs(a[q] - b[q]) for q in ("mean", "p05", "p95")))
            max_diff = max(max_diff, abs(mine["share_of_runs_within_alpha"] - theirs["share_of_runs_within_alpha"]))
    header = "| Dataset | Size | False-report rate (%) | Runs within target (%) | Routed share (%) |\n|---|---|---|---|---|"
    for title, table in (("Recomputed from the released files", recomputed), ("Saved results (results/experiments_cv*.json)", saved),
                         ("Published, manuscript Table 5", {(n, s): PUBLISHED[n][s] for n, _ in ROWS for s in SIZES})):
        print(f"\n{title}\n{header}")
        for name, label in ROWS:
            parts = [" / ".join(table[(name, s)][c] for s in SIZES) for c in range(3)]
            print(f"| {label} | Nano / S / M / L | {parts[0]} | {parts[1]} | {parts[2]} |")
    n_cells = 0
    for name, label in ROWS:
        for size in SIZES:
            for c, col in enumerate(("false-report rate", "runs within target", "routed share")):
                n_cells += 1
                values = (recomputed[(name, size)][c], saved[(name, size)][c], PUBLISHED[name][size][c])
                if len(set(values)) != 1:
                    failures.append(f"{label} {LABEL[size]} {col}: recomputed {values[0]}, saved {values[1]}, published {values[2]}")
    print(f"\nlargest absolute difference between recomputed and saved values (fractions): {max_diff:.3g}")
    print(f"cells compared: {n_cells}; rows not covered by the released detections: CHV, SHWD, SH17")
    if failures:
        print("MISMATCH")
        print("\n".join(failures))
        sys.exit(1)
    print("ALL CELLS MATCH")


if __name__ == "__main__":
    main()
