"""Class composition of the automatic-report pool on Hard Hat Workers, YOLOX, for the R12 introduction.

Question: do helmet reports dominate the report pool, so that the aggregate false-report rate is
low while bare-head alerts are less reliable? Uses the full-source rule (fit_rule on all source
photos) and, as a check, the 5th and 95th percentile of the source-half report cuts saved in
experiments_cv*.json (e1_cluster.chosen.t_high). Hide threshold does not affect reports.
Run from the study repository root with PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=.:
  PPE_VARIANT=<''|s|m|l> python report_pool_composition.py
Writes only beside this script.
"""
import json
import os

import numpy as np

from routing import core, datasets
from routing.experiments import ALL_IDS, MATCH
from routing.experiments_cv import RESULT_FILE, clean_sets, fit_rule
from routing.folds import VARIANT

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "hard_hat_workers"
HELMET, HEAD = core.CLASSES.index("helmet"), core.CLASSES.index("head")

source, targets, _ = clean_sets()
cal, _, t_full = fit_rule(core.Records.from_images(source, class_ids=ALL_IDS))[0]
with open(RESULT_FILE, encoding="utf8") as f:
    chosen = json.load(f)["e1_cluster"]["chosen"]["t_high"]
cuts = {"full_source": t_full, "source_half_p05": chosen["p05"], "source_half_p95": chosen["p95"]}
recs = [core.Records.from_images(ims, class_ids=datasets.evaluable_ids(NAME), rule=MATCH[NAME]) for ims in targets[NAME]]
out = {"variant": VARIANT or "nano", "cuts": cuts, "rows": {}}
for label, t in cuts.items():
    rows = []
    for r in recs:
        rep = cal.predict(r.det_score) >= t
        n = int(rep.sum())
        h = int((rep & (r.det_cls == HELMET)).sum())
        b = int((rep & (r.det_cls == HEAD)).sum())
        fh = int((rep & (r.det_cls == HELMET) & ~r.det_tp).sum())
        fb = int((rep & (r.det_cls == HEAD) & ~r.det_tp).sum())
        rows.append({"reports": n, "helmet_share": h / n, "head_share": b / n,
                     "helmet_false_rate": fh / max(1, h), "head_false_rate": fb / max(1, b),
                     "aggregate_false_rate": (fh + fb) / n, "false_reports_from_heads_share": fb / max(1, fh + fb)})
    out["rows"][label] = {"models": rows, **{k: [min(x[k] for x in rows), max(x[k] for x in rows)] for k in rows[0]}}
with open(os.path.join(HERE, f"report_pool_composition_{VARIANT or 'nano'}.json"), "w", encoding="utf8") as f:
    json.dump(out, f, indent=2)
for label, s in out["rows"].items():
    print(VARIANT or "nano", label, round(cuts[label], 3), "helmet share", [round(v, 3) for v in s["helmet_share"]],
          "head false", [round(v, 3) for v in s["head_false_rate"]], "agg", [round(v, 3) for v in s["aggregate_false_rate"]])
