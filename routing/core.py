"""
Routing study core: matching, calibration, routing metrics, threshold selection with
statistical control, and the label-budget resampling experiment.

Pure numpy / scipy / scikit-learn, no images, so every piece can be unit tested.

Vocabulary
  detection score s   objectness * class score after NMS (what the detector emits)
  confidence p        calibrated score shown to a person
  report              p >= t_high: logged as a finding without review
  route               t_low <= p < t_high: sent to a person
  hidden              p < t_low
  false-report rate   share of reported detections that are wrong (1 - precision of the report band)
  miss rate           share of true violations with no detection at or above t_low
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

CLASSES = ("helmet", "head", "vest", "person")
VIOLATION_IDS = (CLASSES.index("head"),)  # a bare head is a missing hard hat


# ---------------------------------------------------------------- matching
def box_iou(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_inside(box, boxes):
    """Share of box that lies inside each of boxes."""
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / max(1e-12, (box[2] - box[0]) * (box[3] - box[1]))


def match_image(det_boxes, det_scores, det_cls, gt_boxes, gt_cls, iou_thr=0.5, rule="iou"):
    """Greedy matching in descending score order within a class (the COCO convention).

    Because higher scores are matched first, dropping low-score detections never changes
    the outcome for the ones that remain, so true-positive flags hold for every threshold.
    Returns tp (bool per detection) and gt_det (index of the matching detection per
    ground truth, -1 if none).

    rule="iou"      IoU at or above iou_thr.
    rule="contain"  for datasets that box the whole head where the detector boxes only the
                    hat: the detection must lie at least 70 percent inside the ground truth
                    box and overlap it with IoU at or above 0.2.
    rule={class_id: rule}  one of the rules above per class (SH17: whole-head labels for bare
                    heads, hat and body boxes for the rest); classes not listed use IoU.
    rule="symmetric" IoU at or above iou_thr, or containment in either direction (the smaller
                    box at least 70 percent inside the larger, IoU at or above 0.2); a post-hoc
                    check used by the multi-source study, where detectors trained on whole-head
                    labels return boxes larger than hat-only labels.
    """
    nd, ng = len(det_scores), len(gt_cls)
    tp = np.zeros(nd, dtype=bool)
    gt_det = np.full(ng, -1, dtype=np.int64)
    if nd == 0 or ng == 0:
        return tp, gt_det
    per_class = isinstance(rule, dict)
    for d in np.argsort(-np.asarray(det_scores), kind="stable"):
        cand = np.where((gt_cls == det_cls[d]) & (gt_det < 0))[0]
        if cand.size == 0:
            continue
        ious = box_iou(det_boxes[d], gt_boxes[cand])
        rule_d = rule.get(int(det_cls[d]), "iou") if per_class else rule
        if rule_d == "symmetric":
            # post-hoc check for the multi-source study: a hit by IoU, or by containment in either
            # direction (the smaller box at least 70 percent inside the larger, IoU at or above 0.2)
            inside_gt = box_inside(det_boxes[d], gt_boxes[cand])
            gt_inside = np.array([box_inside(g, det_boxes[d:d + 1])[0] for g in gt_boxes[cand]])
            ok = (ious >= iou_thr) | ((ious >= 0.2) & ((inside_gt >= 0.7) | (gt_inside >= 0.7)))
            ious = np.where(ok, ious, -1.0)
            j = int(np.argmax(ious))
            hit = ious[j] >= 0
        elif rule_d != "iou":
            inside, min_iou = (0.7, 0.2) if rule_d == "contain" else rule_d[1:]
            ok = (ious >= min_iou) & (box_inside(det_boxes[d], gt_boxes[cand]) >= inside)
            ious = np.where(ok, ious, -1.0)
            j = int(np.argmax(ious))
            hit = ious[j] >= 0
        else:
            j = int(np.argmax(ious))
            hit = ious[j] >= iou_thr
        if hit:
            tp[d] = True
            gt_det[cand[j]] = d
    return tp, gt_det


@dataclass
class Records:
    """Flat arrays for one dataset: one row per detection and one per ground truth."""
    n_images: int
    det_img: np.ndarray
    det_cls: np.ndarray
    det_score: np.ndarray
    det_tp: np.ndarray
    gt_img: np.ndarray
    gt_cls: np.ndarray
    gt_det: np.ndarray  # index into the det arrays, -1 if the ground truth was never detected

    @staticmethod
    def from_images(images, floor=0.05, iou_thr=0.5, class_ids=None, rule="iou"):
        """images: iterable of dicts with det_boxes, det_scores, det_cls, gt_boxes, gt_cls.

        floor drops detections below the raw score the web page also ignores.
        class_ids limits scoring to the classes a dataset actually labels.
        """
        di, dc, ds, dt, gi, gc, gd = [], [], [], [], [], [], []
        offset = 0
        n = 0
        for i, im in enumerate(images):
            n += 1
            s = np.asarray(im["det_scores"], dtype=np.float64)
            c = np.asarray(im["det_cls"], dtype=np.int64)
            b = np.asarray(im["det_boxes"], dtype=np.float64).reshape(-1, 4)
            gb = np.asarray(im["gt_boxes"], dtype=np.float64).reshape(-1, 4)
            gcl = np.asarray(im["gt_cls"], dtype=np.int64)
            keep = s >= floor
            gkeep = np.ones(len(gcl), dtype=bool)
            if class_ids is not None:
                keep &= np.isin(c, class_ids)
                gkeep = np.isin(gcl, class_ids)
            s, c, b, gb, gcl = s[keep], c[keep], b[keep], gb[gkeep], gcl[gkeep]
            tp, g = match_image(b, s, c, gb, gcl, iou_thr, rule)
            di.append(np.full(len(s), i)); dc.append(c); ds.append(s); dt.append(tp)
            gi.append(np.full(len(gcl), i)); gc.append(gcl); gd.append(np.where(g >= 0, g + offset, -1))
            offset += len(s)
        cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)
        return Records(n, cat(di, np.int64), cat(dc, np.int64), cat(ds, np.float64), cat(dt, bool),
                       cat(gi, np.int64), cat(gc, np.int64), cat(gd, np.int64))

    def subset(self, image_ids):
        """Records for a subset of images, re-indexed from zero."""
        image_ids = np.asarray(image_ids, dtype=np.int64)
        new_index = np.full(self.n_images, -1, dtype=np.int64)
        new_index[image_ids] = np.arange(len(image_ids))
        dmask = new_index[self.det_img] >= 0
        gmask = new_index[self.gt_img] >= 0
        dmap = np.cumsum(dmask) - 1
        g = self.gt_det[gmask]
        return Records(len(image_ids), new_index[self.det_img[dmask]], self.det_cls[dmask], self.det_score[dmask],
                       self.det_tp[dmask], new_index[self.gt_img[gmask]], self.gt_cls[gmask],
                       np.where(g >= 0, dmap[np.maximum(g, 0)], -1))


# ---------------------------------------------------------------- calibration
def _logit(s):
    s = np.clip(np.asarray(s, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Identity:
    name = "none"

    def fit(self, s, y, cls=None):
        return self

    def predict(self, s, cls=None):
        return np.asarray(s, dtype=np.float64)


class Platt:
    """p = sigmoid(a * logit(s) + b). With fit_bias=False this is temperature scaling."""

    def __init__(self, fit_bias=True):
        self.fit_bias = fit_bias
        self.name = "platt" if fit_bias else "temperature"
        self.a, self.b = 1.0, 0.0

    def fit(self, s, y, cls=None):
        y = np.asarray(y, dtype=np.float64)
        if len(y) < 10 or y.min() == y.max():
            return self  # too little evidence to move off the identity map
        x = _logit(s)

        def nll(w):
            z = w[0] * x + (w[1] if self.fit_bias else 0.0)
            return np.mean(np.logaddexp(0.0, z) - y * z)

        res = minimize(nll, [1.0, 0.0] if self.fit_bias else [1.0], method="BFGS")
        self.a = float(res.x[0])
        self.b = float(res.x[1]) if self.fit_bias else 0.0
        return self

    def predict(self, s, cls=None):
        return _sigmoid(self.a * _logit(s) + self.b)


class Isotonic:
    name = "isotonic"

    def __init__(self):
        self.model = None

    def fit(self, s, y, cls=None):
        y = np.asarray(y, dtype=np.float64)
        if len(y) < 10 or y.min() == y.max():
            return self
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(s, dtype=np.float64), y)
        return self

    def predict(self, s, cls=None):
        s = np.asarray(s, dtype=np.float64)
        return s if self.model is None else self.model.predict(s)


class PerClass:
    """One calibrator per class, falling back to a pooled one where a class has few detections."""

    def __init__(self, factory, min_n=50):
        self.factory, self.min_n = factory, min_n
        self.name = factory().name + "_per_class"
        self.pooled, self.by_class = None, {}

    def fit(self, s, y, cls):
        s, y, cls = np.asarray(s), np.asarray(y), np.asarray(cls)
        self.pooled = self.factory().fit(s, y)
        self.by_class = {}
        for c in np.unique(cls):
            m = cls == c
            if m.sum() >= self.min_n:
                self.by_class[int(c)] = self.factory().fit(s[m], y[m])
        return self

    def predict(self, s, cls):
        s, cls = np.asarray(s, dtype=np.float64), np.asarray(cls)
        out = self.pooled.predict(s)
        for c, model in self.by_class.items():
            m = cls == c
            if m.any():
                out[m] = model.predict(s[m])
        return out


def make_calibrator(name):
    table = {
        "none": Identity,
        "temperature": lambda: Platt(fit_bias=False),
        "platt": Platt,
        "isotonic": Isotonic,
        "platt_per_class": lambda: PerClass(Platt),
        "isotonic_per_class": lambda: PerClass(Isotonic),
    }
    return table[name]()


def ece(p, y, bins=10):
    """Detection expected calibration error over equal-width bins (lo, hi]."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return float("nan"), []
    idx = np.clip(np.ceil(p * bins).astype(int) - 1, 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            rows.append({"lo": b / bins, "hi": (b + 1) / bins, "count": 0, "confidence": None, "accuracy": None})
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        total += m.mean() * abs(conf - acc)
        rows.append({"lo": b / bins, "hi": (b + 1) / bins, "count": int(m.sum()), "confidence": conf, "accuracy": acc})
    return float(total), rows


# ---------------------------------------------------------------- routing metrics
def routing_metrics(rec, p, t_low, t_high, violation_ids=VIOLATION_IDS):
    """Outcome of the three-band rule on one set of records.

    t_high=None means nothing is reported without review; t_low=None means nothing is hidden.
    """
    th = np.inf if t_high is None else t_high
    tl = 0.0 if t_low is None else min(t_low, th)
    report = p >= th
    route = (p >= tl) & ~report
    n_rep, n_route = int(report.sum()), int(route.sum())
    out = {
        "t_low": None if t_low is None else float(tl),
        "t_high": None if t_high is None else float(th),
        "n_report": n_rep,
        "n_route": n_route,
        "false_report_rate": float(1 - rec.det_tp[report].mean()) if n_rep else 0.0,
        "route_precision": float(rec.det_tp[route].mean()) if n_route else float("nan"),
        "false_alarm_images": float((np.bincount(rec.det_img[report & ~rec.det_tp], minlength=rec.n_images) > 0).mean()) if rec.n_images else float("nan"),
        "share_routed": n_route / max(1, n_rep + n_route),
        "routed_per_image": n_route / max(1, rec.n_images),
    }
    g = np.isin(rec.gt_cls, violation_ids)
    n_viol = int(g.sum())
    if n_viol:
        d = rec.gt_det[g]
        pg = np.where(d >= 0, p[np.maximum(d, 0)], -np.inf)
        auto = pg >= th
        review = (pg >= tl) & ~auto
        missed = ~(auto | review)
        _, inv = np.unique(rec.gt_img[g], return_inverse=True)
        per_image = np.bincount(inv, weights=missed.astype(np.float64)) / np.bincount(inv)
        # miss_rate pools every violation; miss_rate_per_image averages the share missed per
        # photo, which is the quantity select_t_low controls.
        out.update({"n_violations": n_viol, "violation_auto": float(auto.mean()), "violation_review": float(review.mean()),
                    "miss_rate": float(missed.mean()), "miss_rate_per_image": float(per_image.mean())})
    else:
        out.update({"n_violations": 0, "violation_auto": float("nan"), "violation_review": float("nan"),
                    "miss_rate": float("nan"), "miss_rate_per_image": float("nan")})
    return out


# ---------------------------------------------------------------- threshold selection
def false_report_pvalue(rec, p, t, alpha, method="cluster"):
    """P-value for H0: the false-report rate at threshold t is above alpha.

    cluster   images are the sampling unit. The rate is a ratio of per-image counts, so test
              mean(FP_i - alpha * N_i) < 0 with a one-sided t test. Asymptotic, but it does
              not pretend that detections inside one photo are independent.
    binomial  exact binomial on pooled detections. Assumes independent detections; kept as
              the naive baseline.
    image     a different risk: the share of photos that contain at least one false report.
              One Bernoulli outcome per photo, so the exact binomial test is valid at any
              sample size. Photos with nothing reported count as clean.
    """
    rep = p >= t
    n_i = np.bincount(rec.det_img[rep], minlength=rec.n_images).astype(np.float64)
    fp_i = np.bincount(rec.det_img[rep & ~rec.det_tp], minlength=rec.n_images).astype(np.float64)
    if n_i.sum() == 0:
        return 1.0
    if method == "binomial":
        return float(stats.binom.cdf(fp_i.sum(), n_i.sum(), alpha))
    if method == "image":
        return float(stats.binom.cdf((fp_i > 0).sum(), len(fp_i), alpha))
    loss = fp_i - alpha * n_i
    m, sd, n = loss.mean(), loss.std(ddof=1), len(loss)
    if sd == 0:
        return 0.0 if m < 0 else 1.0
    return float(stats.t.cdf(m / (sd / np.sqrt(n)), n - 1))


def select_t_high(rec, p, alpha=0.10, delta=0.10, grid=None, method="cluster", starts=(0.98, 0.95, 0.90, 0.80, 0.70)):
    """Lowest report threshold whose false-report rate is certified at or below alpha.

    Learn-then-Test with fixed-sequence testing from several start points (Bonferroni across
    starts), so the family-wise error is at most delta. The start points are fixed in advance
    and sit mostly at the top of the range, because that is where a low false-report rate is
    plausible and the very top holds too few detections to certify anything. Returns None
    when no threshold can be certified, which means nothing should be reported without review.
    """
    grid = np.round(np.arange(0.99, 0.495, -0.01), 2) if grid is None else np.asarray(grid)
    level = delta / len(starts)
    valid = []
    for s in starts:
        start = int(np.argmax(grid <= s + 1e-9))
        for t in grid[start:]:
            if false_report_pvalue(rec, p, t, alpha, method) <= level:
                valid.append(float(t))
            else:
                break
    return min(valid) if valid else None


def select_t_low(rec, p, beta=0.10, grid=None, violation_ids=VIOLATION_IDS):
    """Highest hide threshold that keeps the expected miss rate of true violations at or below beta.

    Conformal risk control. The loss per image is the share of its true violations with no
    detection at or above the threshold; it is monotone in the threshold and bounded by 1.
    Returns None when even the lowest threshold misses too many, which means the detector
    cannot meet the target on this data whatever the rule.
    """
    grid = np.round(np.arange(0.05, 0.905, 0.01), 2) if grid is None else np.asarray(grid)
    g = np.isin(rec.gt_cls, violation_ids)
    if not g.any():
        return None
    d = rec.gt_det[g]
    pg = np.where(d >= 0, p[np.maximum(d, 0)], -np.inf)
    imgs, inv = np.unique(rec.gt_img[g], return_inverse=True)
    per_img_total = np.bincount(inv).astype(np.float64)
    n = len(imgs)
    best = None
    for t in np.sort(grid):
        risk = (np.bincount(inv, weights=(pg < t).astype(np.float64)) / per_img_total).mean()
        if n / (n + 1) * risk + 1 / (n + 1) <= beta:
            best = float(t)
        else:
            break
    return best


def select_thresholds(rec, p, alpha=0.10, beta=0.10, delta=0.10, method="cluster"):
    return select_t_low(rec, p, beta), select_t_high(rec, p, alpha, delta, method=method)


# ---------------------------------------------------------------- label budget
def average_precision(rec, class_id):
    """AP at the matching IoU for one class, 101-point interpolation (the COCO convention)."""
    d = rec.det_cls == class_id
    n_gt = int((rec.gt_cls == class_id).sum())
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-rec.det_score[d], kind="stable")
    tp = rec.det_tp[d][order].astype(np.float64)
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # precision envelope
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0).mean())


def budget_curve(source, target, budgets, reps=200, alpha=0.10, beta=0.10, delta=0.10,
                 calibrator="platt", refit=("platt", "isotonic"), method="cluster", seed=0, rule=None):
    """How many labeled target images restore the routing rule after a shift.

    For each budget n and repetition: draw n target images as the local labeled set, keep
    the rest as the test set, then compare on the test set
      source    calibrator and thresholds from the source data, unchanged
      target    source calibrator, thresholds reselected on the n local images
    and record the calibration error of the source calibrator against one refitted on the
    n local images. The calibrator is monotone, so reselecting thresholds on calibrated
    confidence is equivalent to reselecting on raw scores and keeps the selection valid.
    """
    rng = np.random.default_rng(seed)
    if rule is not None:  # a rule fixed elsewhere, such as the one shipped with the model: (calibrator, t_low, t_high)
        cal, tl_src, th_src = rule
    else:
        cal = make_calibrator(calibrator).fit(source.det_score, source.det_tp, source.det_cls)
        tl_src, th_src = select_thresholds(source, cal.predict(source.det_score, source.det_cls), alpha, beta, delta, method)
    rows = []
    for n in budgets:
        if n >= target.n_images:
            continue
        for r in range(reps):
            perm = rng.permutation(target.n_images)
            local, test = target.subset(perm[:n]), target.subset(perm[n:])
            p_local, p_test = cal.predict(local.det_score, local.det_cls), cal.predict(test.det_score, test.det_cls)
            row = {"n": int(n), "rep": r}
            for key, value in routing_metrics(test, p_test, tl_src, th_src).items():
                row[f"source_{key}"] = value
            tl, th = select_thresholds(local, p_local, alpha, beta, delta, method)
            if tl is None:  # no miss-rate target can be certified locally: keep the source hide threshold
                tl = tl_src
            for key, value in routing_metrics(test, p_test, tl, th).items():
                row[f"target_{key}"] = value
            row["ece_source"] = ece(p_test, test.det_tp)[0]
            for name in refit:
                local_cal = make_calibrator(name).fit(local.det_score, local.det_tp, local.det_cls)
                row[f"ece_refit_{name}"] = ece(local_cal.predict(test.det_score, test.det_cls), test.det_tp)[0]
            rows.append(row)
    return {"source_thresholds": {"t_low": tl_src, "t_high": th_src}, "alpha": alpha, "beta": beta, "delta": delta, "rows": rows}


def summarize_budget(result, prefix="target", risk="false_report_rate"):
    """Per budget: mean outcome and the share of repetitions that met each target."""
    alpha, beta = result["alpha"], result["beta"]
    out = []
    for n in sorted({r["n"] for r in result["rows"]}):
        rows = [r for r in result["rows"] if r["n"] == n]
        frr = np.array([r[f"{prefix}_{risk}"] for r in rows])
        miss = np.array([r[f"{prefix}_miss_rate_per_image"] for r in rows])
        out.append({
            "n": n,
            "false_report_rate_mean": float(frr.mean()),
            "false_report_within_alpha": float((frr <= alpha).mean()),
            "miss_rate_mean": float(np.nanmean(miss)) if np.isfinite(miss).any() else float("nan"),
            "miss_within_beta": float((miss <= beta).mean()),
            "share_routed_mean": float(np.mean([r[f"{prefix}_share_routed"] for r in rows])),
            "reports_nothing": float(np.mean([r[f"{prefix}_t_high"] is None for r in rows])),
        })
    return out
