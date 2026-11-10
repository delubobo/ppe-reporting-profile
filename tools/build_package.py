"""Build the release package from the study repository. Rerunnable.

  python tools/build_package.py --study <study repository> --analysis <folder holding held_out_table12.py>

Run with the study's own Python environment and PYTHONDONTWRITEBYTECODE=1, so that importing
from the study repository leaves no bytecode in it. The study repository is only read.

Steps
  1. routing/   the import closure of the analysis entry points, copied unchanged except for
                the documented docstring edits in EDITS
  2. tests/     __init__.py and test_core.py
  3. analysis/  held_out_table12.py, report_pool_composition.py and their JSON outputs
  4. data/      class_map.json and matching_settings.json written from the code; folds.json,
                environment.json and yolox_source.json copied
  5. results/   outputs/results/*.json, with any absolute local path replaced by its base name
  6. detections/, data/manifests/  by tools/export_detections.py
  7. data/duplicates/              by tools/export_duplicate_lists.py

A generated file that would change on a rerun is first moved to archive/<path>_<date> inside
the package folder (archive/ is not part of the release).
"""
import argparse
import datetime
import inspect
import json
import os
import re
import shutil
import subprocess
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = datetime.date.today().isoformat()

ROUTING = ("__init__.py", "core.py", "datasets.py", "experiments.py", "experiments_cv.py", "folds.py", "run_study.py",
           "detect.py", "pilots.py", "tables.py", "hide_budget.py", "person_level.py")
ENTRY_POINTS = ("routing/experiments_cv.py", "routing/tables.py", "routing/hide_budget.py", "routing/person_level.py",
                "tests/test_core.py")
ANALYSIS = ("held_out_table12.py", "report_pool_composition.py", "heldout_rtdetr_r18.json", "heldout_rtdetr_r18_pairs.json",
            "heldout_rtdetr_r50.json", "heldout_rtdetr_r50_pairs.json", "report_pool_composition_nano.json",
            "report_pool_composition_s.json", "report_pool_composition_m.json", "report_pool_composition_l.json",
            "reproduce_rtdetr_r18.json", "reproduce_rtdetr_r50.json")
# Docstring and comment edits only; no code is changed. Each (file, old, new) must match once.
EDITS = (
    ("routing/folds.py", "the 700 source photos (data/css_v30)",
     "the 700 source photos (data/css_v30; 656 remain after duplicate cleaning)"),
    ("routing/folds.py", '"""Ground truth for the 700 photos, from the held-out label file of the fold each belongs to."""',
     '"""Ground truth for the 700 source photos (656 after duplicate cleaning), from the held-out label file of the fold each belongs to."""'),
)
ABS_PATH_JSON = re.compile(r'(?:[A-Za-z]:(?:\\\\|/)|/' + 'Users' + r'/)[^"]*?([^"\\/]+)"')


def put(rel, data, log):
    """Write bytes to PKG/rel; an existing different file is archived first."""
    dst = os.path.join(PKG, rel)
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == data:
                log.append(f"unchanged  {rel}")
                return
        arc = os.path.join(PKG, "archive", f"{rel}_{TODAY}")
        n = 1
        while os.path.exists(arc):
            n += 1
            arc = os.path.join(PKG, "archive", f"{rel}_{TODAY}_{n}")
        os.makedirs(os.path.dirname(arc), exist_ok=True)
        shutil.move(dst, arc)
        log.append(f"archived   {rel} -> {os.path.relpath(arc, PKG)}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    log.append(f"written    {rel} ({len(data):,} bytes)")


def read(path):
    with open(path, "rb") as f:
        return f.read()


def imports_of(text):
    """Module names a file imports from the routing package."""
    names = set()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"from \.(\w*) import (.+)", line) or re.match(r"from routing\.?(\w*) import (.+)", line)
        if m:
            if m.group(1):
                names.add(m.group(1))
            else:
                names |= {x.strip() for x in m.group(2).split(",")}
    return names


def closure(study):
    todo = [os.path.join(study, p) for p in ENTRY_POINTS]
    seen = {os.path.basename(p)[:-3] for p in ENTRY_POINTS if p.startswith("routing/")}
    while todo:
        text = read(todo.pop()).decode("utf8")
        for name in imports_of(text):
            if name not in seen and os.path.exists(os.path.join(study, "routing", name + ".py")):
                seen.add(name)
                todo.append(os.path.join(study, "routing", name + ".py"))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--analysis", required=True)
    args = ap.parse_args()
    study, analysis = os.path.abspath(args.study), os.path.abspath(args.analysis)
    log = []

    # 1. routing closure, verified from the imports of every entry point
    found = closure(study)
    for p in ANALYSIS[:2]:
        for name in imports_of(read(os.path.join(analysis, p)).decode("utf8")):
            if os.path.exists(os.path.join(study, "routing", name + ".py")):
                found.add(name)
    expected = {f[:-3] for f in ROUTING if f != "__init__.py"}
    print("import closure found:", sorted(found))
    if found != expected:
        raise SystemExit(f"closure differs from ROUTING: missing {sorted(found - expected)}, extra {sorted(expected - found)}")
    edits_done = []
    for f in ROUTING:
        text = read(os.path.join(study, "routing", f)).decode("utf8")
        for rel, old, new in EDITS:
            if rel == f"routing/{f}":
                if text.count(old) != 1:
                    raise SystemExit(f"edit target not found exactly once in {rel}: {old}")
                text = text.replace(old, new)
                edits_done.append((rel, old, new))
        put(f"routing/{f}", text.encode("utf8"), log)

    # 2. tests
    for f in ("__init__.py", "test_core.py"):
        put(f"tests/{f}", read(os.path.join(study, "tests", f)), log)

    # 3. analysis scripts and outputs
    for f in ANALYSIS:
        put(f"analysis/{f}", read(os.path.join(analysis, f)), log)

    # 4. class map and matching settings, read from the code itself
    sys.path.insert(0, study)
    cwd = os.getcwd()
    os.chdir(study)
    try:
        from routing import core, datasets, experiments, experiments_cv, run_study
    finally:
        os.chdir(cwd)
    names = {"css_v30": "css_v27", "chv": "chv", "hard_hat_workers": "hard_hat_workers", "shwd": "shwd", "sh17": "sh17"}
    class_map = {"unified_classes": list(core.CLASSES),
                 "note": ("The source photos are the 700 photos of the Construction Site Safety export v30 listed in folds.json; "
                          "they use the class mapping of registry entry css_v27 in routing/datasets.py. Classes a dataset does not "
                          "label are never scored on it (evaluable classes). Licences as read on the dataset pages are in LICENSE-DATA.md."),
                 "datasets": {}}
    for name, key in names.items():
        spec = datasets.REGISTRY[key]
        entry = {"registry_entry": key, "role": spec["role"], "title": spec["title"], "license_registry": spec["license"],
                 "format": spec["format"]}
        if "names" in spec:
            entry["source_label_names"] = list(spec["names"])
        entry["mapping_to_unified"] = spec["mapping"]
        entry["evaluable_classes"] = list(spec["evaluable"])
        entry["evaluable_ids"] = list(datasets.evaluable_ids(key))
        entry["download"] = spec["download"]
        class_map["datasets"][name] = entry
    put("data/class_map.json", (json.dumps(class_map, indent=2) + "\n").encode("utf8"), log)

    src = inspect.getsource(core.match_image)
    m = re.search(r'inside, min_iou = \(([\d.]+), ([\d.]+)\) if rule_d == "contain"', src)
    if not m:
        raise SystemExit("containment parameters not found in core.match_image")
    sig_match = inspect.signature(core.match_image).parameters
    sig_rec = inspect.signature(core.Records.from_images).parameters
    sig_clean = inspect.signature(experiments_cv.clean_sets).parameters
    match = {}
    for n, rule in experiments.MATCH.items():
        match[n] = {core.CLASSES[k]: v for k, v in rule.items()} if isinstance(rule, dict) else rule
    if "sh17" in match:
        match["sh17_note"] = "classes not listed use IoU"
    with open(os.path.join(study, "models", "folds", "folds.json"), encoding="utf8") as f:
        folds = json.load(f)
    settings = {
        "ALPHA": experiments.ALPHA, "BETA": experiments.BETA, "DELTA": experiments.DELTA,
        "ALPHA_PHOTO": experiments.ALPHA_PHOTO, "T_LOW_DEFAULT": experiments.T_LOW_DEFAULT,
        "MATCH": match,
        "iou_threshold": sig_match["iou_thr"].default,
        "containment": {"min_share_of_detection_inside_ground_truth": float(m.group(1)), "min_iou": float(m.group(2)),
                        "source": "routing/core.py match_image, rule 'contain'"},
        "records_score_floor": sig_rec["floor"].default,
        "ap_score_floor": 0.01,
        "calibrator": "platt",
        "report_threshold_method": {"aggregate": "cluster", "photo_level": "image"},
        "fold_rule": folds["rule"], "n_folds": folds["n_folds"],
        "duplicate_exclusion": {"hash": "64-bit difference hash (dhash) of a 9 x 8 grey thumbnail, routing/run_study.py",
                                "max_differing_bits": sig_clean["bits"].default,
                                "order": ["source against itself across folds",
                                          "each target in order " + ", ".join(experiments.TARGETS) + ", then sh17: "
                                          "against the source, then against every previously kept target"]},
    }
    put("data/matching_settings.json", (json.dumps(settings, indent=2) + "\n").encode("utf8"), log)
    for f in ("folds.json", "environment.json", "yolox_source.json"):
        put(f"data/{f}", read(os.path.join(study, "models", "folds", f)), log)

    # 5. saved results, JSON only, absolute local paths reduced to base names
    res_dir = os.path.join(study, run_study.RESULT_DIR)
    scrubbed = {}
    for f in sorted(os.listdir(res_dir)):
        if not f.endswith(".json"):
            continue
        text = read(os.path.join(res_dir, f)).decode("utf8")
        new, n = ABS_PATH_JSON.subn(lambda mm: mm.group(1) + '"', text)
        if n:
            json.loads(new)
            scrubbed[f] = n
        put(f"results/{f}", new.encode("utf8"), log)

    # 6 and 7. detections, manifests and duplicate lists
    here = os.path.dirname(os.path.abspath(__file__))
    for script in ("export_detections.py", "export_duplicate_lists.py"):
        print(f"--- {script}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(here, script), "--study", study], check=False)
        if r.returncode:
            raise SystemExit(f"{script} failed with exit code {r.returncode}")

    print("\n".join(log))
    print("docstring edits:", json.dumps(edits_done, indent=1))
    print("results with local paths reduced to base names:", json.dumps(scrubbed))


if __name__ == "__main__":
    main()
