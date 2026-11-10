"""The price of a certified miss target: routed share and routed detections per photo when the hide
threshold is lowered to the value conformal risk control certified (E4), under the source rule's
report threshold. Usage: PPE_VARIANT=rtdetr_r50 python -m routing.hide_budget [target]
Writes outputs/results/hide_budget_<variant>.json and prints one line per model."""
import json
import os
import sys

import numpy as np

from . import core, datasets
from .experiments import ALL_IDS, MATCH
from .experiments_cv import RESULT_FILE, across, clean_sets, fit_rule
from .folds import VARIANT
from .run_study import RESULT_DIR


def main(target="0.1"):
    with open(RESULT_FILE, encoding="utf8") as f:
        e4 = json.load(f)["e4_miss"]
    source, targets, _ = clean_sets()
    rec = core.Records.from_images(source, class_ids=ALL_IDS)
    (cal, _, t_high), _ = fit_rule(rec)
    out = {"variant": VARIANT or "nano", "target": target, "t_high": t_high, "datasets": {}}
    for name, models in targets.items():
        if name not in e4:
            continue
        thresholds = e4[name].get(target)
        if thresholds is None:
            continue
        if not isinstance(thresholds, list):
            thresholds = [thresholds] * len(models)
        rows = []
        for ims, t_low in zip(models, thresholds):
            if t_low is None:
                continue
            r = core.Records.from_images(ims, class_ids=datasets.evaluable_ids(name), rule=MATCH[name])
            m = core.routing_metrics(r, cal.predict(r.det_score), t_low, t_high)
            rows.append({"t_low": t_low, "share_routed": m["share_routed"], "routed_per_image": m["routed_per_image"],
                         "miss_rate": m["miss_rate"], "miss_rate_per_image": m["miss_rate_per_image"], "false_report_rate": m["false_report_rate"]})
        if rows:
            out["datasets"][name] = {"models": rows, **{k: across([x[k] for x in rows]) for k in rows[0]}}
            s = out["datasets"][name]
            print(f"{VARIANT or 'nano'} {name} miss target {target}: hide {s['t_low']['min']:.2f} to {s['t_low']['max']:.2f} | routed share {s['share_routed']['mean']:.3f} "
                  f"({s['share_routed']['min']:.3f} to {s['share_routed']['max']:.3f}) | routed per photo {s['routed_per_image']['mean']:.2f} "
                  f"({s['routed_per_image']['min']:.2f} to {s['routed_per_image']['max']:.2f}) | miss per photo {s['miss_rate_per_image']['mean']:.3f} | {len(rows)} models", flush=True)
    with open(os.path.join(RESULT_DIR, f"hide_budget_{VARIANT or 'nano'}.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0.1")
