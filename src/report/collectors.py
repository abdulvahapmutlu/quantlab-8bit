from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json, glob

def _load_json(p: str | Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}

def collect_fp32_metrics(root: str | Path = "artifacts/reports/fp32_metrics") -> Dict[Tuple[str,str], Dict[str, Any]]:
    out: Dict[Tuple[str,str], Dict[str, Any]] = {}
    for f in glob.glob(str(Path(root) / "*.json")):
        j = _load_json(f)
        key = (j.get("model",""), j.get("dataset",""))
        if key[0] and key[1]:
            out[key] = j
    return out

def collect_method_metrics() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # PTQ static
    for f in glob.glob("artifacts/reports/ptq_static/*_metrics.json"):
        j = _load_json(f); j["method"] = j.get("method","ptq_static")
        rows.append(j)
    # PTQ dynamic
    for f in glob.glob("artifacts/reports/ptq_dynamic/*_metrics.json"):
        j = _load_json(f); j["method"] = j.get("method","ptq_dynamic")
        rows.append(j)
    # QAT
    for f in glob.glob("artifacts/reports/qat/*_metrics.json"):
        j = _load_json(f); j["method"] = j.get("method","qat")
        rows.append(j)
    return rows

def read_bench_csv(path: str | Path = "artifacts/reports/bench/ort_cpu_results.csv") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists(): return rows
    import csv
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["batch"] = int(row.get("batch", "0") or 0)
            except Exception:
                row["batch"] = 0
            rows.append(row)
    return rows
