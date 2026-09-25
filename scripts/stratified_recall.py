DESCRIPTION = "Lesion-level recall stratified by ground-truth box size.  A prediction counts as a hit at IoU >= 0.5. Size bins follow common object-detection conventions, measured at the network input scale."

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

BINS = (("small<32", 0, 32), ("medium32-96", 32, 96), ("large>=96", 96, float("inf")))


def load_gt(label_path, width, height, scale):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = (float(v) for v in parts)
            boxes.append([
                (xc - w / 2) * width, (yc - h / 2) * height,
                (xc + w / 2) * width, (yc + h / 2) * height,
                max(w, h) * scale,
            ])
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", required=True, help="dataset root holding images/<split> and labels/<split>")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    from ultralytics import YOLO

    splits = args.splits.split(",")
    files = []
    for split in splits:
        image_dir = os.path.join(args.data_root, "images", split)
        files += [(split, n) for n in sorted(os.listdir(image_dir)) if n.lower().endswith(".png")]

    model = YOLO(args.weights)
    hits = {name: 0 for name, _, _ in BINS}
    totals = {name: 0 for name, _, _ in BINS}
    false_positives = 0

    for start in range(0, len(files), args.batch):
        chunk = files[start:start + args.batch]
        paths = [os.path.join(args.data_root, "images", s, n) for s, n in chunk]
        predictions = model.predict(paths, conf=args.conf, imgsz=args.imgsz, verbose=False)
        for (split, name), result in zip(chunk, predictions):
            height, width = result.orig_shape
            label_path = os.path.join(args.data_root, "labels", split,
                                      os.path.splitext(name)[0] + ".txt")
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
            false_positives += len(boxes)
            used = set()
            for gt in load_gt(label_path, width, height, args.imgsz):
                for bin_name, low, high in BINS:
                    if low <= gt[4] < high:
                        totals[bin_name] += 1
                        best_index, best_iou = -1, 0.0
                        for index, box in enumerate(boxes):
                            if index in used:
                                continue
                            value = iou(gt[:4], box)
                            if value > best_iou:
                                best_index, best_iou = index, value
                        if best_iou >= args.iou:
                            hits[bin_name] += 1
                            used.add(best_index)
                        break

    print(f"images: {len(files)}  false positives per image: {false_positives / max(len(files), 1):.2f}")
    for bin_name, _, _ in BINS:
        total = totals[bin_name]
        recall = hits[bin_name] / total if total else float("nan")
        print(f"  {bin_name:14s} recall {recall:.3f}  ({hits[bin_name]}/{total})")


if __name__ == "__main__":
    main()
