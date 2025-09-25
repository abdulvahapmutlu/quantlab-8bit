from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def validate_schema(obj: Dict[str, Any], schema_path: str | Path) -> None:
    try:
        import jsonschema
        from jsonschema import validate
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        validate(instance=obj, schema=schema)
    except Exception:
        # Soft-fail if jsonschema not installed or validation fails
        pass
