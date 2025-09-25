import argparse, csv, time
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import load_yaml
from src.utils.reporting import write_json, validate_schema
from src.bench.utils import read_bench_cfg, disclose_env, model_file_size_mb, build_synthetic_input


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ONNX Runtime CPU benchmark harness")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--bench-config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--onnx-path", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--schema", default="configs/eval/bench_schema.json")
    return ap.parse_args()


def _percentiles(xs: List[float], ps=(50, 95, 99)) -> Dict[str, float]:
    import numpy as np
    arr = np.array(xs, dtype=np.float64)
    out = {}
    for p in ps:
        out[f"p{p}_ms"] = float(np.percentile(arr, p))
    return out


def main() -> None:
    args = parse_args()
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)
    bcfg = read_bench_cfg(args.bench_config)

    dataset = dcfg["dataset"]["name"]
    model = mcfg["model"]["name"]
    env = disclose_env()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(bcfg["session_options"]["intra_op_num_threads"])
    so.inter_op_num_threads = int(bcfg["session_options"]["inter_op_num_threads"])
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(args.onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_size = tuple(dcfg["dataset"]["input_size"])  # C,H,W
    batches = list(bcfg["measurement"]["batches"])
    warmup_iters = int(bcfg["measurement"]["warmup_iters"])
    measure_iters = int(bcfg["measurement"]["measure_iters"])
    repeats = int(bcfg["measurement"].get("repeats", 1))
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rows = []
    for b in batches:
        # Warmup
        feed = build_synthetic_input(input_size, b)
        for _ in range(warmup_iters):
            sess.run([out_name], {in_name: feed["input"]})

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(measure_iters):
                sess.run([out_name], {in_name: feed["input"]})
            end = time.perf_counter()
            timings.append((end - start) * 1000.0 / measure_iters)  # ms per iter

        pct = _percentiles(timings)
        p50 = pct["p50_ms"]; p95 = pct["p95_ms"]; p99 = pct["p99_ms"]
        total_secs = sum(timings) / 1000.0 * measure_iters  # approx
        rps = (b * measure_iters * repeats) / (sum(timings) / 1000.0)

        row = {
            "dataset": dataset, "model": model, "method": args.method, "onnx_path": str(Path(args.onnx_path).as_posix()),
            "file_mb": model_file_size_mb(args.onnx_path),
            "batch": b,
            "warmup_iters": warmup_iters,
            "measure_iters": measure_iters,
            "repeats": repeats,
            "p50_ms": round(p50, 4), "p95_ms": round(p95, 4), "p99_ms": round(p99, 4), "rps": round(rps, 2),
            "env": {
                "provider": "CPUExecutionProvider",
                "ort_version": env.get("ort_version"),
                "intra_op_num_threads": bcfg["session_options"]["intra_op_num_threads"],
                "inter_op_num_threads": bcfg["session_options"]["inter_op_num_threads"],
                "execution_mode": "sequential",
                "cpu_model": env.get("cpu_model"),
                "ram_gb": env.get("ram_gb"),
                "os": env.get("os"),
            },
            "notes": ""
        }
        rows.append(row)
        # Write per-batch JSON
        write_json(args.out_json.replace(".json", f"_b{b}.json"), row)
        validate_schema(row, args.schema)

    # Append CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    header = ["dataset","model","method","onnx_path","file_mb","batch","warmup_iters","measure_iters","repeats","p50_ms","p95_ms","p99_ms","rps","provider","ort_version","intra_op_num_threads","inter_op_num_threads","execution_mode","cpu_model","ram_gb","os","notes"]
    write_header = not Path(args.out_csv).exists()
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            envcol = r["env"]
            w.writerow([
                r["dataset"], r["model"], r["method"], r["onnx_path"], r["file_mb"], r["batch"],
                r["warmup_iters"], r["measure_iters"], r["repeats"],
                r["p50_ms"], r["p95_ms"], r["p99_ms"], r["rps"],
                envcol["provider"], envcol["ort_version"], envcol["intra_op_num_threads"], envcol["inter_op_num_threads"], envcol["execution_mode"], envcol["cpu_model"], envcol["ram_gb"], envcol["os"],
                r["notes"]
            ])


if __name__ == "__main__":
    main()
