from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json, shutil

def write_html_bundle(out_dir: str | Path, columns: List[str], rows: List[Dict[str, Any]], hardware: Dict[str, Any]) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # JSON payload
    (out / "leaderboard.json").write_text(json.dumps({"columns": columns, "rows": rows, "hardware": hardware}, indent=2), encoding="utf-8")
    # Static HTML
    src_tpl = Path("src/report/templates/leaderboard_html.html")
    if src_tpl.exists():
        shutil.copy(src_tpl, out / "index.html")
