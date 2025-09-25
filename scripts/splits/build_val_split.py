#!/usr/bin/env python
import argparse, json, random
from pathlib import Path
from datetime import datetime
from torchvision import datasets
from torchvision.datasets import ImageFolder

def stratified_indices(labels, frac, seed):
    by_c = {}
    for i, y in enumerate(labels):
        by_c.setdefault(y, []).append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for c, idxs in by_c.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return sorted(train_idx), sorted(val_idx)

def cifar10_split(root, frac, seed):
    ds = datasets.CIFAR10(root=str(root), train=True, download=False)
    labels = [ds[i][1] for i in range(len(ds))]
    tr, va = stratified_indices(labels, frac, seed)
    return {
        "dataset": "cifar10",
        "root": str(root),
        "frac": frac,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_idx": tr, "val_idx": va,
        "n_train": len(tr), "n_val": len(va),
        "num_classes": 10
    }

def tiny_imagenet_split(root, frac, seed):
    ds = ImageFolder(str(Path(root)/"train"))
    labels = [y for _, y in ds.samples]
    tr, va = stratified_indices(labels, frac, seed)
    return {
        "dataset": "tiny-imagenet",
        "root": str(root),
        "frac": frac,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_idx": tr, "val_idx": va,
        "n_train": len(tr), "n_val": len(va),
        "num_classes": len(ds.classes)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","tiny-imagenet"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--frac", type=float, default=0.1)  # 10%
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dataset == "cifar10":
        payload = cifar10_split(args.root, args.frac, args.seed)
    else:
        payload = tiny_imagenet_split(args.root, args.frac, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[split] wrote {args.out}: train={payload['n_train']} val={payload['n_val']}")

if __name__ == "__main__":
    main()
