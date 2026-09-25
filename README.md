# Usage

Environment: Python 3.11, PyTorch 2.11 (CUDA 12.8), timm 0.6.13.
Install dependencies with `pip install -r requirements.txt`.

## 1. Split the dataset (patient level)

```bash
python scripts/split_dataset_patient_level.py --src /path/to/raw/slices --dst /path/to/output
```

All slices of a patient are assigned to exactly one subset. The resulting
assignment is recorded in `data/patient_split_manifest.csv`.

## 2. Train

```bash
python scripts/train.py --data configs/dataset.yaml --variant cbam-asff
```

`--variant` selects `yolov8`, `cbam`, `asff` or `cbam-asff`; model definition
files live in `ultralytics/cfg/models/v8/`.

## 3. Evaluate on the held-out test set

```bash
python scripts/evaluate_test_set.py --weights runs/detect/best.pt
python scripts/stratified_recall.py --weights runs/detect/best.pt
```

## 4. Figures

```bash
python scripts/make_figures.py --run-a runs/detect/yolov8n --run-b runs/detect/cbam_asff
```
