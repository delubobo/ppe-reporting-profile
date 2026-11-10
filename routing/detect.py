"""
Run the exported detector over images and return detections in the unified classes.

Pre and post processing are copied from ml/site-photo-check/calibrate.py in the site
repository, which is the source of record and matches what the browser page does:
letterbox to 416 with gray 114 padding, BGR, no normalization, grid decode over strides
8, 16, 32, per-class NMS at IoU 0.45.

Usage:
  python -m routing.detect --model models/site-photo-check-v1 --images some/folder --out detections.npz
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from .core import CLASSES

MODEL_TO_UNIFIED = {"Hardhat": "helmet", "NO-Hardhat": "head", "Safety Vest": "vest", "Person": "person"}
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preproc(img, input_size, swap=(2, 0, 1)):
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(swap), dtype=np.float32), r


def decode(outputs, img_size):
    grids, strides_all = [], []
    for stride in (8, 16, 32):
        h, w = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides_all.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    strides_all = np.concatenate(strides_all, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * strides_all
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides_all
    return outputs


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


class Detector:
    def __init__(self, model_dir, size=416, conf_thr=0.01, nms_thr=0.45, threads=None):
        with open(os.path.join(model_dir, "classes.json"), encoding="utf8") as f:
            self.model_classes = json.load(f)["classes"]
        options = ort.SessionOptions()
        if threads:  # several detectors in one process: idle threads that keep spinning starve the others
            options.intra_op_num_threads = threads
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(os.path.join(model_dir, "yolox_nano_ppe.onnx"), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size, self.conf_thr, self.nms_thr = (size, size), conf_thr, nms_thr
        # model class index -> unified class index, for the classes the study scores
        self.class_map = {i: CLASSES.index(MODEL_TO_UNIFIED[name]) for i, name in enumerate(self.model_classes) if name in MODEL_TO_UNIFIED}

    def __call__(self, img_bgr):
        """Boxes (x1, y1, x2, y2 in original pixels), raw scores, unified class ids."""
        x, r = preproc(img_bgr, self.size)
        pred = decode(self.session.run(None, {self.input_name: x[None]})[0], self.size)[0]
        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2; xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2; xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        out_b, out_s, out_c = [], [], []
        for c, unified in self.class_map.items():
            s = scores[:, c]
            m = s > self.conf_thr
            if not m.any():
                continue
            b, sc = xyxy[m], s[m]
            keep = nms(b, sc, self.nms_thr)
            out_b.append(b[keep] / r); out_s.append(sc[keep]); out_c.append(np.full(len(keep), unified))
        if not out_b:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        return np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_c).astype(np.int64)


def run(detector, paths, log_every=200):
    """Detections for a list of image paths, as parallel lists (None where an image is unreadable)."""
    results = []
    for n, path in enumerate(paths):
        img = cv2.imread(path)
        results.append(None if img is None else detector(img))
        if log_every and n % log_every == 0:
            print(f"  {n}/{len(paths)} images")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder with yolox_nano_ppe.onnx and classes.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(os.path.join(args.images, f) for f in os.listdir(args.images) if f.lower().endswith(IMAGE_EXT))
    results = run(Detector(args.model), paths)
    keep = [(p, r) for p, r in zip(paths, results) if r is not None]
    np.savez_compressed(args.out, paths=np.array([p for p, _ in keep]),
                        counts=np.array([len(r[1]) for _, r in keep]),
                        boxes=np.concatenate([r[0] for _, r in keep]) if keep else np.zeros((0, 4)),
                        scores=np.concatenate([r[1] for _, r in keep]) if keep else np.zeros(0),
                        cls=np.concatenate([r[2] for _, r in keep]) if keep else np.zeros(0, dtype=np.int64))
    print(f"{len(keep)} images, {sum(len(r[1]) for _, r in keep)} detections -> {args.out}")


if __name__ == "__main__":
    main()
