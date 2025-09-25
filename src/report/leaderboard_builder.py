import argparse, math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.config import load_yaml
from src.utils.reporting import write_json
from src.report.collectors import collect_fp32_metrics, collect_method_metrics, read_bench_csv
from src.report.badges import badge_for_accuracy, badge_for_speedup
from src.report.md_renderer import render_markdown
from src.report.html_renderer import write_html_bundle

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build QuantLab-8bit leaderboard (CSV/MD/HTML)")
    ap.add_argument("--matrix-config", default="configs/experiment_matrix.yaml")
    ap.add_argument("--leaderboard-config", default="configs/reporting/leaderboard.yaml")
    ap.add_argument("--bench-csv", default="artifacts/reports/bench/ort_cpu_results.csv")
    ap.add_argument("--csv-out", default=None)  # override
    ap.add_argument("--md-out",  default=None)
    ap.add_argument("--html-out-dir", default="artifacts/reports/leaderboard_html")
    ap.add_argument("--md-header", default="src/report/templates/leaderboard_md_header.md")
    return ap.parse_args()

def _float(x): 
    try: return None if x in (None,"",) else float(x)
    except Exception: return None

def main() -> None:
    args = parse_args()
    lcfg = load_yaml(args.leaderboard_config)["leaderboard"]
    columns: List[str] = lcfg["columns"]
    # resolve outs
    csv_out = args.csv_out or lcfg["csv_out"]
    md_out  = args.md_out  or lcfg["md_out"]

    # collect inputs
    fp32 = collect_fp32_metrics()              # key: (model,dataset) → metrics
    methods = collect_method_metrics()         # list of {model,dataset,method,top1,top5,size_mb,...}
    bench = read_bench_csv(args.bench_csv)     # rows with batch info

    # quick index of bench by key
    def bench_lookup(model, dataset, method, batch):
        for r in bench:
            if r.get("model")==model and r.get("dataset")==dataset and r.get("method")==method and int(r.get("batch",0))==batch:
                return r
        return None

    rows: List[Dict[str, Any]] = []
    for m in methods:
        model = m.get("model",""); dataset = m.get("dataset",""); method = m.get("method","")
        if not model or not dataset or not method: 
            continue
        # fp32 baseline (for ΔTop-1 and speedup calc)
        base = fp32.get((model, dataset), {})
        base_top1 = base.get("top1")
        # bench fetch
        b1 = bench_lookup(model, dataset, method, 1)
        b8 = bench_lookup(model, dataset, method, 8)
        b1_fp32 = bench_lookup(model, dataset, "fp32", 1)

        def get(col, default=None, row=None):
            return _float(row.get(col)) if row and col in row else default

        row = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "top1": m.get("top1"),
            "top5": m.get("top5"),
            "d_top1_pp": (None if (base_top1 is None or m.get("top1") is None) else float(m["top1"]) - float(base_top1)),
            "size_mb": m.get("size_mb"),
            "b1_p50_ms": get("p50_ms", None, b1),
            "b1_p95_ms": get("p95_ms", None, b1),
            "b8_p50_ms": get("p50_ms", None, b8),
            "b8_p95_ms": get("p95_ms", None, b8),
            "b1_throughput_rps": get("rps", None, b1),
            "notes": m.get("notes","")
        }
        rows.append(row)

    # sort
    sort = lcfg.get("sort", {})
    method_priority = {m:i for i,m in enumerate(sort.get("method_priority", []))}
    rows.sort(key=lambda r: (r["dataset"], r["model"], method_priority.get(r["method"].split(":")[0], 999), (r["b1_p50_ms"] or 1e9)))

    # add emoji badges inline to notes (optional)
    thr = lcfg.get("thresholds", {})
    acc_rules = lcfg.get("emoji_rules", {}).get("accuracy", {})
    spd_rules = lcfg.get("emoji_rules", {}).get("latency_speedup", {})
    for r in rows:
        # accuracy badge
        badge_a = badge_for_accuracy(r.get("d_top1_pp"), acc_rules)
        # speedup vs fp32 @ B1
        speedup = None
        # find fp32 b1 p50
        # you might have many rows; preindex earlier in real impl
        # quick pass:
        b1_fp32 = next((x for x in rows if x["dataset"]==r["dataset"] and x["model"]==r["model"] and x["method"]=="fp32"), None)
        if b1_fp32 and b1_fp32.get("b1_p50_ms") and r.get("b1_p50_ms"):
            try:
                speedup = float(b1_fp32["b1_p50_ms"]) / float(r["b1_p50_ms"])
            except Exception:
                speedup = None
        badge_s = badge_for_speedup(speedup, spd_rules) if speedup is not None else ""
        if badge_a or badge_s:
            r["notes"] = f"{badge_a}{badge_s} {r.get('notes','')}".strip()

    # CSV
    from csv import DictWriter
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            # ensure only configured columns are written
            w.writerow({k: r.get(k, "") for k in columns})

    # MD
    md = render_markdown(columns, rows, header_md_path=args.md_header)
    Path(md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(md_out).write_text(md, encoding="utf-8")

    # HTML bundle (optional)
    # Pull minimal hardware disclosure from bench CSV first row if present
    hw = {}
    if bench:
        env_cols = ["cpu_model","ram_gb","os","ort_version","inter_op_num_threads","intra_op_num_threads","execution_mode"]
        for c in bench[0].keys():
            pass
        env = bench[0]
        hw = {
            "cpu_model": env.get("cpu_model"),
            "cpu_threads": env.get("intra_op_num_threads"),
            "ram_gb": env.get("ram_gb"),
            "os": env.get("os"),
            "ort_version": env.get("ort_version")
        }
    write_html_bundle(args.html_out_dir, columns, rows, hw)

if __name__ == "__main__":
    main()
