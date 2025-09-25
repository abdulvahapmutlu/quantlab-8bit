from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple


def read_bench_cfg(path: str | Path) -> Dict[str, Any]:
    from src.utils.config import load_yaml
    return load_yaml(path)


def disclose_env() -> Dict[str, Any]:
    import json, platform
    try:
        import psutil
    except Exception:
        class _P: 
            def __getattr__(self, k): 
                raise AttributeError
        psutil = _P()

    sys_path = Path("repro/system.json")
    info = {
        "cpu_model": platform.processor() or "unknown",
        "ram_gb": round(getattr(psutil, "virtual_memory", lambda: type("X",(object,),{"total":0})())().total / (1024**3), 2) if hasattr(psutil, "virtual_memory") else None,
        "os": f"{platform.system()} {platform.release()}",
        "ort_version": None
    }
    try:
        import onnxruntime as ort
        info["ort_version"] = ort.__version__
    except Exception:
        info["ort_version"] = "unknown"
    if sys_path.exists():
        try:
            d = json.loads(sys_path.read_text(encoding="utf-8"))
            info["cpu_model"] = d.get("hardware", {}).get("cpu", {}).get("name", info["cpu_model"])
            info["os"] = d.get("os", {}).get("pretty", info["os"])
        except Exception:
            pass
    return info


def model_file_size_mb(p: str | Path) -> float:
    p = Path(p)
    return round(p.stat().st_size / (1024**2), 4)


def build_synthetic_input(chw: Tuple[int,int,int], batch: int) -> Dict[str, Any]:
    import numpy as np
    c, h, w = chw
    x = np.random.randn(batch, c, h, w).astype(np.float32)
    return {"input": x}
