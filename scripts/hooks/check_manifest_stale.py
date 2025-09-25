import json, sys
from pathlib import Path

MANIFEST = Path("repro/manifest.json")
if not MANIFEST.exists():
    print("[repro] manifest.json missing (ok for now)"); sys.exit(0)
try:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # In your real impl: compare commit SHA, config hashes, etc.
    print("[repro] manifest present")
    sys.exit(0)
except Exception as e:
    print("[repro] manifest unreadable:", e)
    sys.exit(1)
