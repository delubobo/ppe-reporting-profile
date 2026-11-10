"""Synthetic checks of the statistical core. Run: python -m tests.test_core"""
import numpy as np

from routing import core


def synthetic(n_images, rng, a=1.3, b=0.5, shift=0.0, miss=0.15):
    """Detections whose chance of being right is sigmoid(a * logit(s) + b - shift)."""
    images = []
    for _ in range(n_images):
        k = rng.poisson(6) + 1
        s = np.clip(rng.beta(2, 2, k), 0.06, 0.99)
        cls = rng.integers(0, 4, k)
        right = rng.random(k) < core._sigmoid(a * core._logit(s) + b - shift)
        boxes = np.zeros((k, 4)); gt_boxes = []; gt_cls = []
        for j in range(k):
            x = 100.0 * j
            boxes[j] = [x, 0, x + 50, 50]
            if right[j]:
                gt_boxes.append([x, 0, x + 50, 50]); gt_cls.append(cls[j])
        for j in range(rng.binomial(3, miss)):  # violations the detector never saw
            gt_boxes.append([5000.0 + 100 * j, 0, 5050.0 + 100 * j, 50]); gt_cls.append(1)
        images.append({"det_boxes": boxes, "det_scores": s, "det_cls": cls,
                       "gt_boxes": np.array(gt_boxes).reshape(-1, 4), "gt_cls": np.array(gt_cls, dtype=int)})
    return core.Records.from_images(images)


def test_matching():
    tp, g = core.match_image(np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]]), np.array([0.9, 0.8, 0.7]),
                             np.array([0, 0, 0]), np.array([[0, 0, 10, 10]]), np.array([0]))
    assert tp.tolist() == [True, False, False] and g.tolist() == [0]


def test_subset_keeps_links():
    rec = synthetic(60, np.random.default_rng(1))
    sub = rec.subset(np.arange(10, 30))
    linked = sub.gt_det >= 0
    assert sub.n_images == 20 and sub.det_tp.sum() == linked.sum()
    assert (sub.det_img[sub.gt_det[linked]] == sub.gt_img[linked]).all()
    assert sub.det_tp[sub.gt_det[linked]].all()


def test_calibration_recovers_map():
    rec = synthetic(1500, np.random.default_rng(2))
    cal = core.make_calibrator("platt").fit(rec.det_score, rec.det_tp)
    assert abs(cal.a - 1.3) < 0.15 and abs(cal.b - 0.5) < 0.15, (cal.a, cal.b)
    before = core.ece(rec.det_score, rec.det_tp)[0]
    for name in ("platt", "isotonic", "platt_per_class"):
        after = core.ece(core.make_calibrator(name).fit(rec.det_score, rec.det_tp, rec.det_cls).predict(rec.det_score, rec.det_cls), rec.det_tp)[0]
        assert after < before / 2, (name, before, after)


def test_threshold_control():
    """Selected thresholds should meet their targets on fresh data at about the stated rate."""
    rng = np.random.default_rng(3)
    alpha, beta, delta = 0.10, 0.25, 0.10
    big = synthetic(6000, rng, miss=0.05)
    cal = core.make_calibrator("platt").fit(big.det_score, big.det_tp)
    p_big = cal.predict(big.det_score)
    broke_alpha, miss_risk, no_high, no_low = 0, [], 0, 0
    trials = 150
    for _ in range(trials):
        small = synthetic(200, rng, miss=0.05)
        p = cal.predict(small.det_score)
        tl, th = core.select_thresholds(small, p, alpha, beta, delta)
        m = core.routing_metrics(big, p_big, tl, th)
        no_high += th is None
        no_low += tl is None
        broke_alpha += m["false_report_rate"] > alpha + 0.005
        miss_risk.append(m["miss_rate_per_image"])
    assert no_high < trials * 0.2 and no_low < trials * 0.2, (no_high, no_low)
    assert broke_alpha / trials <= delta + 0.05, broke_alpha / trials
    assert np.mean(miss_risk) <= beta + 0.01, np.mean(miss_risk)
    # An unreachable target is reported as such instead of returning a threshold.
    hard = synthetic(300, rng, miss=0.6)
    assert core.select_t_low(hard, cal.predict(hard.det_score), beta=0.05) is None


def test_budget_curve_detects_shift():
    rng = np.random.default_rng(4)
    source, target = synthetic(800, rng), synthetic(800, rng, shift=1.2)
    res = core.budget_curve(source, target, budgets=[25, 100, 300], reps=40, seed=5)
    src = {r["n"]: r for r in core.summarize_budget(res, "source")}
    tgt = {r["n"]: r for r in core.summarize_budget(res, "target")}
    assert src[300]["false_report_rate_mean"] > 0.10          # the source rule breaks under shift
    assert tgt[300]["false_report_rate_mean"] <= 0.105        # local labels restore it
    assert tgt[300]["false_report_within_alpha"] >= 0.8
    mean = lambda n, key: np.mean([r[key] for r in res["rows"] if r["n"] == n])
    assert mean(300, "ece_refit_platt") < mean(300, "ece_source")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
