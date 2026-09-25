DESCRIPTION = "Evaluate a trained checkpoint on the held-out test split."

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    metrics = YOLO(args.weights).val(
        data=args.data,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=0,
        device=args.device,
        plots=False,
    )
    f1 = float(np.mean(metrics.box.f1)) if metrics.box.f1 is not None else float("nan")
    print(f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}  F1={f1:.4f}  "
          f"mAP@0.5={metrics.box.map50:.4f}  mAP@0.5:0.95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
