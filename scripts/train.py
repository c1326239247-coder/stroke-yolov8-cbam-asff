DESCRIPTION = "Train one YOLOv8 variant under the fixed protocol used in the paper."

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

VARIANTS = {
    "yolov8": "yolov8.yaml",
    "cbam": "yolov8-C2f_CBAM-backbone.yaml",
    "asff": "yolov8-Detect_ASFF.yaml",
    "cbam-asff": "yolov8-Detect_ASFF+cbam.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--data", required=True, help="dataset yaml")
    parser.add_argument("--variant", default="cbam-asff", choices=sorted(VARIANTS))
    parser.add_argument("--weights", default="yolov8n.pt", help="pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    model_yaml = os.path.join(REPO, "ultralytics", "cfg", "models", "v8", VARIANTS[args.variant])
    model = YOLO(model_yaml, task="detect")
    if os.path.exists(args.weights):
        model = model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=100,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        seed=0,
        deterministic=True,
        pretrained=True,
        workers=args.workers,
        val=True,
        plots=True,
        amp=True,
        name=args.name or args.variant.replace("-", "_"),
    )


if __name__ == "__main__":
    main()
