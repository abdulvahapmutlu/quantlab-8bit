from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

def render_markdown(columns: List[str], rows: List[Dict[str, Any]], header_md_path: str | None = None) -> str:
    parts: List[str] = []
    if header_md_path and Path(header_md_path).exists():
        parts.append(Path(header_md_path).read_text(encoding="utf-8").strip())
        parts.append("")
    # table header
    parts.append("| " + " | ".join(columns) + " |")
    parts.append("| " + " | ".join(["---"]*len(columns)) + " |")
    for r in rows:
        parts.append("| " + " | ".join(str(r.get(c,"")) for c in columns) + " |")
    return "\n".join(parts)
