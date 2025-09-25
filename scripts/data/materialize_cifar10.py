#!/usr/bin/env python
import argparse, json, os
from pathlib import Path
from torchvision import datasets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root, e.g., C:/data/cifar10")
    ap.add_argument("--download", action="store_true", help="download if missing")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    train = datasets.CIFAR10(root=str(root), train=True, download=args.download)
    test  = datasets.CIFAR10(root=str(root), train=False, download=args.download)

    meta = {
        "dataset":"cifar10",
        "root": str(root),
        "train_len": len(train),
        "test_len": len(test),
        "classes": train.classes
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
