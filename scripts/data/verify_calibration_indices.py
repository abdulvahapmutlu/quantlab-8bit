#!/usr/bin/env python
import argparse, json
from pathlib import Path
from hashlib import sha1
import numpy as np
from torchvision import datasets
from torchvision.datasets import ImageFolder

def h(x: bytes) -> str: return sha1(x).hexdigest()

def verify_cifar10(root: Path, payload: dict) -> int:
    ds = datasets.CIFAR10(root=str(root), train=True, download=False)
    ok = 0
    for idx, saved_id in zip(payload["indices"], payload["ids"]):
        img, y = ds[idx]
        raw = np.array(img).tobytes() + bytes([y])
        if h(raw) == saved_id: ok += 1
    return ok

def verify_tiny(root: Path, payload: dict) -> int:
    ds = ImageFolder(str(root/"train"))
    ok = 0
    for i, saved_id in zip(payload["indices"], payload["ids"]):
        path, y = ds.samples[i]
        with open(path, "rb") as f:
            raw = f.read()
        if h(raw) == saved_id: ok += 1
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="artifacts/.../cifar10.json or tiny-imagenet.json")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    root = Path(args.root)
    if payload["dataset"] == "cifar10":
        ok = verify_cifar10(root, payload)
    else:
        ok = verify_tiny(root, payload)

    n = len(payload["indices"])
    print(f"[verify] matched {ok}/{n} samples")
    raise SystemExit(0 if ok == n else 2)

if __name__ == "__main__":
    main()
