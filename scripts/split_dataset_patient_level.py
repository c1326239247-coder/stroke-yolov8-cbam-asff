DESCRIPTION = "Rebuild a YOLO dataset split so that all slices of a patient stay together.  Adjacent 2D slices of one MRI study are highly similar, so splitting them independently leaks information between the training and evaluation sets. This script groups slices by patient identifier and assigns whole patients to train/val/test, targeting the requested image-count ratio."

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

PATIENT_RE = re.compile(r"(sub-[A-Za-z0-9]+case\d+)")


def collect_patients(src):
    images = defaultdict(list)
    for split in ("train", "val", "test"):
        image_dir = os.path.join(src, "images", split)
        if not os.path.isdir(image_dir):
            continue
        for name in os.listdir(image_dir):
            match = PATIENT_RE.match(name)
            if match:
                images[match.group(1)].append((split, name))
    if not images:
        raise SystemExit(f"no images found under {src}/images/<split>")
    return images


def assign(images, ratios):
    total = sum(len(v) for v in images.values())
    counts = {k: 0 for k in ratios}
    assignment = {}
    for patient, entries in sorted(images.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        target = min(ratios, key=lambda k: counts[k] / (ratios[k] * total))
        assignment[patient] = target
        counts[target] += len(entries)
    return assignment, counts, total


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--src", required=True, help="dataset root with images/<split> and labels/<split>")
    parser.add_argument("--dst", required=True, help="output dataset root")
    parser.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test image-count ratios")
    parser.add_argument("--manifest", default="patient_split_manifest.csv")
    args = parser.parse_args()

    ratios = dict(zip(("train", "val", "test"), (float(v) for v in args.ratios.split(","))))
    images = collect_patients(args.src)
    assignment, counts, total = assign(images, ratios)

    for split in ratios:
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)

    seen = {}
    for patient, entries in images.items():
        split = assignment[patient]
        for source_split, name in entries:
            stem = os.path.splitext(name)[0]
            shutil.copy2(os.path.join(args.src, "images", source_split, name),
                         os.path.join(args.dst, "images", split, name))
            label = stem + ".txt"
            shutil.copy2(os.path.join(args.src, "labels", source_split, label),
                         os.path.join(args.dst, "labels", split, label))
        assert patient not in seen or seen[patient] == split, f"patient {patient} in two splits"
        seen[patient] = split

    with open(args.manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "n_slices", "split"])
        for patient, entries in sorted(images.items()):
            writer.writerow([patient, len(entries), assignment[patient]])

    print(f"patients: {len(images)}  images: {total}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s} {counts[split]:5d} images  ({counts[split] / total:.1%})")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
