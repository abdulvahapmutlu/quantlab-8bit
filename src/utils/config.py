from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def expand_placeholders(template: str, **kwargs: Any) -> str:
    return template.format_map(_Default(kwargs))


class _Default(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
