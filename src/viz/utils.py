from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import load_yaml
from src.utils.reporting import write_json

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_viz_cfg(path: str | Path) -> Dict[str, Any]:
    return load_yaml(path)

def write_manifest(path: str | Path, dataset: str, model: str, method: str, items: List[Dict[str, str]]) -> None:
    write_json(path, {"dataset": dataset, "model": model, "method": method, "items": items})
