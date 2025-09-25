#!/usr/bin/env python
import argparse, hashlib, io, json, os, random, sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

# Optional imports guarded for datasets
from torchvision import datasets
from torchvision.datasets import ImageFolder

SEEDS_FILE = Path("repro/seeds.json")
OUT_DIR = Path("artifacts/reports/calibration_indices")

def load_seeds() -> int:
    if not SEEDS_FILE.exists():
        raise SystemExit("Missing repro/seeds.json; create via Step 2.")
    data = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
    return int(data.get("global", 42))

def sha1_bytes(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()

def cifar10_ids(ds) -> List[str]:
    # Stable ID from image pixel bytes (avoid PIL re-encode variance)
    ids = []
    for i in range(len(ds)):
        img, label = ds[i]
        arr = np.array(img)  # HWC, uint8
        raw = arr.tobytes() + bytes([label])
        ids.append(sha1_bytes(raw))
    return ids

def tiny_imagenet_ids(root: Path) -> Tuple[List[str], List[int], List[str]]:
    # Use ImageFolder to get (path,label); ID from file bytes on disk
    ds = ImageFolder(str(root / "train"))
    class_to_idx = ds.class_to_idx
    samples = ds.samples  # list of (path, label)
    ids, labels, paths = [], [], []
    for path, y in samples:
        with open(path, "rb") as f:
            raw = f.read()
        ids.append(sha1_bytes(raw))
        labels.append(y)
        paths.append(path)
    return ids, labels, [ds.classes[y] for y in labels]

def stratified_sample(indices_by_class: Dict[int, List[int]], size: int, per_class_min: int, rng: random.Random) -> List[int]:
    # Phase 1: guarantee per-class minimum
    selected = []
    leftovers = []
    for c, idxs in indices_by_class.items():
        idxs = idxs[:]  # copy
        rng.shuffle(idxs)
        need = min(per_class_min, len(idxs))
        selected.extend(idxs[:need])
        leftovers.extend(idxs[need:])
    # Phase 2: fill remaining quota proportionally from leftovers
    remaining = max(0, size - len(selected))
    rng.shuffle(leftovers)
    selected.extend(leftovers[:remaining])
    return sorted(selected)

def build_for_cifar10(root: Path, size: int, per_class_min: int, seed: int) -> Dict:
    train = datasets.CIFAR10(root=str(root), train=True, download=False)
    ids = cifar10_ids(train)
    labels = [train[i][1] for i in range(len(train))]
    classes = train.classes

    rng = random.Random(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(y, []).append(i)

    selected = stratified_sample(by_cls, size, per_class_min, rng)
    label_counts = {}
    for i in selected:
        label_counts[labels[i]] = label_counts.get(labels[i], 0) + 1

    return {
        "dataset": "cifar10",
        "split": "train",
        "root": str(root),
        "size": size,
        "per_class_min": per_class_min,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "class_names": {i: cls for i, cls in enumerate(classes)},
        "indices": selected,
        "ids": [ids[i] for i in selected],
        "labels": [labels[i] for i in selected],
        "label_counts": label_counts
    }

def build_for_tinyimagenet(root: Path, size: int, per_class_min: int, seed: int) -> Dict:
    ids, labels, class_names_list = tiny_imagenet_ids(root)
    classes = {i: name for i, name in enumerate(sorted(set(class_names_list)))}

    rng = random.Random(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(y, []).append(i)

    selected = stratified_sample(by_cls, size, per_class_min, rng)
    label_counts = {}
    for i in selected:
        label_counts[labels[i]] = label_counts.get(labels[i], 0) + 1

    return {
        "dataset": "tiny-imagenet",
        "split": "train",
        "root": str(root),
        "size": size,
        "per_class_min": per_class_min,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "class_names": classes,
        "indices": selected,
        "ids": [ids[i] for i in selected],
        "labels": [labels[i] for i in selected],
        "label_counts": label_counts
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","tiny-imagenet"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--per-class-min", type=int, required=True)
    args = ap.parse_args()

    seed = load_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    if args.dataset == "cifar10":
        payload = build_for_cifar10(root, args.size, args.per_class_min, seed)
        out = OUT_DIR / "cifar10.json"
    else:
        payload = build_for_tinyimagenet(root, args.size, args.per_class_min, seed)
        out = OUT_DIR / "tiny-imagenet.json"

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calib] wrote {out} (n={len(payload['indices'])})")

if __name__ == "__main__":
    main()
