"""
Dataset registry and loaders. Every dataset is reduced to the unified classes in core.CLASSES
and to the subset of those classes it actually labels ("evaluable"), so a detector is never
scored on a class the annotators did not mark.

License and source notes were checked on 3 October 2026 from each dataset's own page. Items
marked CONFIRM must be checked against the downloaded files before any number is reported.
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from .core import CLASSES

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

REGISTRY = {
    "css_v27": {
        "role": "source",
        "title": "Construction Site Safety v27, Roboflow Universe Projects",
        "license": "CC BY 4.0 (CONFIRM on the dataset page; image provenance not stated)",
        "format": "coco",
        "mapping": {"Hardhat": "helmet", "NO-Hardhat": "head", "Safety Vest": "vest", "Person": "person"},
        "evaluable": ("helmet", "head", "vest", "person"),
        "download": "Roboflow export, API key entered by the owner (same as training).",
    },
    "hard_hat_workers": {
        "role": "target",
        "title": "Hard Hat Workers, Northeastern University China",
        "license": "CC0 1.0 public domain",
        "format": "voc",
        "mapping": {"helmet": "helmet", "head": "head", "person": "person"},
        # 'person' is labeled only occasionally (616 boxes against 19,852 helmets), so it is not scored.
        "evaluable": ("helmet", "head"),
        "download": "Harvard Dataverse doi:10.7910/DVN/7CBGOS or public.roboflow.com/object-detection/hard-hat-workers",
    },
    "shwd": {
        "role": "target",
        "title": "Safety Helmet Wearing Dataset (SHWD)",
        "license": "MIT (repository level)",
        "format": "voc",
        # In SHWD the label 'person' marks a head without a helmet, not a whole person.
        "mapping": {"hat": "helmet", "person": "head"},
        "evaluable": ("helmet", "head"),
        "download": "Google Drive link in github.com/njvisionpower/Safety-Helmet-Wearing-Dataset",
    },
    "chv": {
        "role": "target",
        "title": "Color Helmet and Vest (CHV), Wang et al. 2021, Sensors",
        "license": "No formal license; README grants free use with citation",
        "format": "yolo",
        # Names and order from annotations/README.md inside the download.
        "names": ("person", "vest", "blue helmet", "red helmet", "white helmet", "yellow helmet"),
        "mapping": {"person": "person", "vest": "vest", "blue helmet": "helmet", "red helmet": "helmet",
                    "white helmet": "helmet", "yellow helmet": "helmet"},
        "evaluable": ("helmet", "vest", "person"),
        "download": "Google Drive link in github.com/ZijianWang-ZW/PPE_detection",
    },
    "sh17": {
        "role": "target",
        "title": "SH17, Ahmad and Rahimi 2024, Journal of Safety Science and Resilience",
        # Licence read on github.com/ahmadmughees/SH17dataset on 6 October 2026: CC BY-NC-SA 4.0,
        # research use; images are Pexels stock photos. Evaluation in a paper is permitted; the
        # images and labels are not redistributed and no public model is trained on them.
        "license": "CC BY-NC-SA 4.0 (evaluation only; never used for training the public model)",
        "format": "yolo",
        # Class order from sh17.yaml in the repository.
        "names": ("person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot", "tool", "glasses", "gloves",
                  "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest"),
        # "head" in SH17 marks every head, helmeted or not. A head box that overlaps a helmet box is
        # dropped after loading (see load), so the unified "head" class is the bare head, as elsewhere.
        "mapping": {"person": "person", "safety-vest": "vest", "helmet": "helmet", "head": "head"},
        "evaluable": ("helmet", "head", "vest", "person"),
        "download": "Kaggle mugheesahmad/sh17-dataset-for-ppe-detection (images and labels), or the Pexels URL list in the repository",
        "box_convention": "CONFIRM on the downloaded labels: whether the helmet box covers the hat alone or the whole head",
    },
}


def _ids(mapping, names):
    """Source label -> unified class index, for the labels this dataset maps."""
    return {src: CLASSES.index(dst) for src, dst in mapping.items() if names is None or src in names}


def evaluable_ids(name):
    key = "css_v27" if name.startswith("css_v27") else name
    return tuple(CLASSES.index(c) for c in REGISTRY[key]["evaluable"])


def load(name, root):
    """Ground truth for a registered dataset, from its extracted download at root."""
    spec = REGISTRY["css_v27" if name.startswith("css_v27") else name]
    if name == "shwd":
        base = os.path.join(root, "VOC2028")
        images = load_voc(os.path.join(base, "Annotations"), os.path.join(base, "JPEGImages"), spec["mapping"])
        # 4,340 of the 7,581 images (PartA_*, PartB_*) come from SCUT-HEAD and show classrooms,
        # not sites. Only the 3,241 web-collected images are used.
        return [im for im in images if not os.path.basename(im["path"]).startswith("Part")]
    if name == "hard_hat_workers":
        images = []
        for split in ("Train", "Test"):
            base = os.path.join(root, "Hardhat", split)
            images += load_voc(os.path.join(base, "Annotation"), os.path.join(base, "JPEGImage"), spec["mapping"])
        return images
    if name.startswith("css_v27"):
        # Roboflow COCO export: one folder per split, each with its own annotation file.
        # Use names css_v27_train, css_v27_valid, css_v27_test to pick the split.
        split = name.split("_")[-1]
        folder = os.path.join(root, split)
        return load_coco(os.path.join(folder, "_annotations.coco.json"), folder, REGISTRY["css_v27"]["mapping"])
    if name == "chv":
        base = os.path.join(root, "CHV_dataset")
        return load_yolo(os.path.join(base, "images"), os.path.join(base, "annotations"), spec["names"], spec["mapping"])
    if name == "sh17":
        # Kaggle layout: images/ and labels/ side by side (CONFIRM after download; adjust the two folder names if not).
        img_dir = next(d for d in (os.path.join(root, "images"), os.path.join(root, "sh17", "images")) if os.path.isdir(d))
        label_dir = os.path.join(os.path.dirname(img_dir), "labels")
        images = load_yolo(img_dir, label_dir, spec["names"], spec["mapping"])
        return strip_helmeted_heads(images)
    raise KeyError(name)


def strip_helmeted_heads(images, min_overlap=0.3):
    """Keep only bare heads in the unified head class: a head box is dropped when a helmet box
    covers at least min_overlap of it or has its center inside it."""
    head, helmet = CLASSES.index("head"), CLASSES.index("helmet")
    for im in images:
        gb, gc = im["gt_boxes"], im["gt_cls"]
        if not len(gb) or not (gc == head).any() or not (gc == helmet).any():
            continue
        heads, hats = gb[gc == head], gb[gc == helmet]
        hx = (hats[:, 0] + hats[:, 2]) / 2; hy = (hats[:, 1] + hats[:, 3]) / 2
        keep = np.ones(len(gb), dtype=bool)
        for i, (idx, box) in enumerate(zip(np.where(gc == head)[0], heads)):
            ix1 = np.maximum(box[0], hats[:, 0]); iy1 = np.maximum(box[1], hats[:, 1])
            ix2 = np.minimum(box[2], hats[:, 2]); iy2 = np.minimum(box[3], hats[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
            inside = (hx >= box[0]) & (hx <= box[2]) & (hy >= box[1]) & (hy <= box[3])
            if ((inter / area) >= min_overlap).any() or inside.any():
                keep[idx] = False
        im["gt_boxes"], im["gt_cls"] = gb[keep], gc[keep]
    return images


def load_voc(ann_dir, img_dir, mapping):
    """Pascal VOC XML, one file per image."""
    table = _ids(mapping, None)
    images = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.lower().endswith(".xml"):
            continue
        root = ET.parse(os.path.join(ann_dir, fname)).getroot()
        img_name = root.findtext("filename") or (os.path.splitext(fname)[0] + ".jpg")
        path = os.path.join(img_dir, img_name)
        if not os.path.exists(path):  # some sets store a stale filename; fall back to the annotation stem
            stem = os.path.splitext(fname)[0]
            path = next((os.path.join(img_dir, stem + e) for e in IMAGE_EXT if os.path.exists(os.path.join(img_dir, stem + e))), None)
            if path is None:
                continue
        boxes, cls = [], []
        for obj in root.iter("object"):
            label = (obj.findtext("name") or "").strip()
            if label not in table:
                continue
            bb = obj.find("bndbox")
            boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            cls.append(table[label])
        images.append({"path": path, "gt_boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4), "gt_cls": np.array(cls, dtype=np.int64)})
    return images


def load_coco(json_path, img_dir, mapping):
    """COCO instances JSON (the Roboflow export)."""
    with open(json_path, encoding="utf8") as f:
        data = json.load(f)
    names = {c["id"]: c["name"] for c in data["categories"]}
    table = _ids(mapping, None)
    by_image = {}
    for a in data["annotations"]:
        label = names.get(a["category_id"])
        if label in table:
            x, y, w, h = a["bbox"]
            by_image.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], table[label]))
    images = []
    for im in data["images"]:
        items = by_image.get(im["id"], [])
        images.append({"path": os.path.join(img_dir, im["file_name"]),
                       "gt_boxes": np.array([b for b, _ in items], dtype=np.float64).reshape(-1, 4),
                       "gt_cls": np.array([c for _, c in items], dtype=np.int64)})
    return images


def load_yolo(img_dir, label_dir, names, mapping):
    """YOLO txt labels (class cx cy w h, normalized). names is the dataset's own class list in index order."""
    import cv2

    table = {i: CLASSES.index(mapping[n]) for i, n in enumerate(names) if n in mapping}
    images = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMAGE_EXT):
            continue
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, cls = [], []
        label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, encoding="utf8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5 or int(parts[0]) not in table:
                        continue
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                    cls.append(table[int(parts[0])])
        images.append({"path": path, "gt_boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4), "gt_cls": np.array(cls, dtype=np.int64)})
    return images
