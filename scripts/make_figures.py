DESCRIPTION = ("Regenerate the training-curve figure and the qualitative detection "
               "figure from trained runs.")

import argparse
import csv
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_run(run_dir):
    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [cell.strip() for cell in rows[0]]
    def column(key):
        return [float(row[header.index(key)]) for row in rows[1:]]
    return {
        "epoch": column("epoch"),
        "mAP50": column("metrics/mAP50(B)"),
        "mAP50-95": column("metrics/mAP50-95(B)"),
    }


def training_curves(run_a, run_b, label_a, label_b, out):
    a, b = load_run(run_a), load_run(run_b)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    panels = [("mAP50", "mAP@0.5"), ("mAP50-95", "mAP@0.5:0.95")]
    for axis, (key, title) in zip(axes, panels):
        axis.plot(a["epoch"], a[key], label=label_a, lw=1.4, color="#4C72B0")
        axis.plot(b["epoch"], b[key], label=label_b, lw=1.4, color="#DD8452")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def draw(image, boxes, colour, confidences=None, scale=4):
    canvas = image.resize((image.width * scale, image.height * scale), Image.NEAREST).convert("RGB")
    pen = ImageDraw.Draw(canvas)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (value * scale for value in box)
        pen.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        if confidences is not None:
            pen.text((x1 + 3, max(0, y1 - 16)), f"{confidences[index]:.2f}", fill=colour)
    return canvas


def detection_grid(weights_a, weights_b, label_a, label_b, data_root, out, limit=5):
    from ultralytics import YOLO

    model_a, model_b = YOLO(weights_a), YOLO(weights_b)
    image_dir = os.path.join(data_root, "images", "test")
    label_dir = os.path.join(data_root, "labels", "test")
    names = sorted(n for n in os.listdir(image_dir) if n.lower().endswith(".png"))

    chosen = []
    for name in names:
        label_path = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            chosen.append(name)
        if len(chosen) >= limit:
            break

    rows = [[], [], []]
    for name in chosen:
        image = Image.open(os.path.join(image_dir, name))
        width, height = image.size
        with open(os.path.join(label_dir, os.path.splitext(name)[0] + ".txt"), encoding="utf-8") as handle:
            ground_truth = []
            for line in handle:
                parts = line.split()
                if len(parts) == 5:
                    _, xc, yc, w, h = (float(v) for v in parts)
                    ground_truth.append([(xc - w / 2) * width, (yc - h / 2) * height,
                                         (xc + w / 2) * width, (yc + h / 2) * height])
        path = os.path.join(image_dir, name)
        result_a = model_a.predict(path, conf=0.25, imgsz=640, verbose=False)[0]
        result_b = model_b.predict(path, conf=0.25, imgsz=640, verbose=False)[0]
        rows[0].append(draw(image, ground_truth, (0, 200, 0)))
        rows[1].append(draw(image, result_a.boxes.xyxy.cpu().numpy(), (60, 100, 220),
                            result_a.boxes.conf.cpu().numpy()))
        rows[2].append(draw(image, result_b.boxes.xyxy.cpu().numpy(), (230, 90, 40),
                            result_b.boxes.conf.cpu().numpy()))

    tile, gap, margin = image.height * 4, 8, 210
    canvas = Image.new("RGB", (len(chosen) * tile + (len(chosen) + 1) * gap + margin, 3 * tile + 4 * gap + 28),
                       (255, 255, 255))
    pen = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for row_index, (row, label) in enumerate(zip(rows, ("GT", label_a, label_b))):
        top = 28 + row_index * (tile + gap)
        pen.text((10, top + tile // 2 - 10), label, fill=(0, 0, 0), font=font)
        for column_index, image in enumerate(row):
            canvas.paste(image, (margin + column_index * (tile + gap), top))
    canvas.save(out)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--run-a", required=True, help="baseline run directory")
    parser.add_argument("--run-b", required=True, help="improved run directory")
    parser.add_argument("--label-a", default="YOLOv8n")
    parser.add_argument("--label-b", default="YOLOv8-CBAM-ASFF")
    parser.add_argument("--data-root", default=None, help="dataset root, required for the detection grid")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    training_curves(args.run_a, args.run_b, args.label_a, args.label_b,
                    os.path.join(args.out_dir, "fig_curves.png"))

    if args.data_root:
        detection_grid(os.path.join(args.run_a, "weights", "best.pt"),
                       os.path.join(args.run_b, "weights", "best.pt"),
                       args.label_a, args.label_b, args.data_root,
                       os.path.join(args.out_dir, "fig_detections.png"))


if __name__ == "__main__":
    main()
