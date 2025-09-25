from pathlib import Path
import sys, json

SEEDS = Path("repro/seeds.json")
if not SEEDS.exists():
    print("[repro] seeds.json missing (ok for now)"); sys.exit(0)
try:
    data = json.loads(SEEDS.read_text(encoding="utf-8"))
    if "global" in data:
        print("[repro] seeds.json OK"); sys.exit(0)
    print("[repro] seeds.json missing 'global' key"); sys.exit(1)
except Exception as e:
    print("[repro] seeds.json unreadable:", e); sys.exit(1)
