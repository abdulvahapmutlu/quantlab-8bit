#!/usr/bin/env python
import json, sys, os

REQUIRED_INT_KEYS = ["global", "python_random", "numpy", "torch", "torch_cuda"]
REQUIRED_BOOL_KEYS = [("torch_backends", "cudnn_deterministic"), ("torch_backends", "cudnn_benchmark")]
REQUIRED_DL_KEYS = [("dataloader", "workers"), ("dataloader", "generator_seed")]

def fail(msg: str) -> None:
    sys.stderr.write(f"[seeds-freshness] {msg}\n")
    sys.exit(1)

def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_seeds.py <path-to-seeds.json>")
    path = sys.argv[1]
    if not os.path.exists(path):
        fail(f"missing file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        fail(f"failed to read JSON: {e}")

    for k in REQUIRED_INT_KEYS:
        if k not in data:
            fail(f"missing key: {k}")
        if not isinstance(data[k], int):
            fail(f"key {k} must be int, got {type(data[k]).__name__}")

    for pkey, ckey in REQUIRED_BOOL_KEYS:
        if pkey not in data or ckey not in data[pkey]:
            fail(f"missing key: {pkey}.{ckey}")
        if not isinstance(data[pkey][ckey], bool):
            fail(f"key {pkey}.{ckey} must be bool")

    for pkey, ckey in REQUIRED_DL_KEYS:
        if pkey not in data or ckey not in data[pkey]:
            fail(f"missing key: {pkey}.{ckey}")
        if ckey == "workers" and not isinstance(data[pkey][ckey], int):
            fail(f"key {pkey}.{ckey} must be int")
        if ckey == "generator_seed" and not isinstance(data[pkey][ckey], int):
            fail(f"key {pkey}.{ckey} must be int")

    # Optional guidance: warn (non-fatal) if workers != 0 for strict determinism
    if data["dataloader"]["workers"] != 0:
        sys.stderr.write("[seeds-freshness] warning: dataloader.workers != 0 may reduce determinism on some OSes.\n")

    print("[seeds-freshness] OK")

if __name__ == "__main__":
    main()
