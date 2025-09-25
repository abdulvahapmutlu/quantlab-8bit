import argparse
from pathlib import Path
from typing import Any, Dict
from src.utils.config import load_yaml
from src.utils.reporting import write_json, validate_schema
from src.data.loaders import build_eval_loader
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parity: ONNX FP32 vs ONNX INT8 (logits-only)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--parity-config", required=True)
    ap.add_argument("--fp32-onnx", required=True)
    ap.add_argument("--int8-onnx", required=True)
    ap.add_argument("--method-int8", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=False)
    ap.add_argument("--schema", default="configs/eval/parity_schema.json")
    return ap.parse_args()


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float64)
    return float(np.mean(d * d))


def main() -> None:
    args = parse_args()
    dcfg = load_yaml(args.dataset_config)
    pcfg = load_yaml(args.parity_config)["parity"]

    import onnxruntime as ort
    s_fp = ort.InferenceSession(args.fp32_onnx, providers=["CPUExecutionProvider"])
    s_i8 = ort.InferenceSession(args.int8_onnx, providers=["CPUExecutionProvider"])
    in_name = s_fp.get_inputs()[0].name
    out_name = s_fp.get_outputs()[0].name

    n_samples = int(pcfg["n_samples"])
    bs = int(pcfg["batch_size"])
    loader = build_eval_loader(dcfg, split="val", batch_size=bs, num_workers=0)

    cos_vals, mse_vals, seen = [], [], 0
    for x, _ in loader:
        if seen >= n_samples:
            break
        x_np = x.numpy().astype(np.float32)
        y_fp = s_fp.run([out_name], {in_name: x_np})[0]
        y_i8 = s_i8.run([out_name], {in_name: x_np})[0]
        cos_vals.append(cosine_sim(y_fp, y_i8))
        mse_vals.append(mse(y_fp, y_i8))
        seen += x.shape[0]

    cmin = min(cos_vals) if cos_vals else None
    cmean = float(np.mean(cos_vals)) if cos_vals else None
    mmax = max(mse_vals) if mse_vals else None
    mmean = float(np.mean(mse_vals)) if mse_vals else None

    thr = pcfg["thresholds"]
    passed = (cmin is not None and mmax is not None and cmin >= thr["cos_min_fp32_vs_int8"] and mmax <= thr["mse_max_fp32_vs_int8"])

    report = {
        "dataset": dcfg["dataset"]["name"],
        "model": load_yaml(args.model_config)["model"]["name"],
        "ref": {"method": "onnx-fp32", "onnx_path": Path(args.fp32_onnx).as_posix()},
        "cmp": {"method": args.method_int8, "onnx_path": Path(args.int8_onnx).as_posix()},
        "n_samples": seen,
        "summary": {"cos_min": cmin, "cos_mean": cmean, "mse_max": mmax, "mse_mean": mmean, "pass": bool(passed), "notes": "logits-only"},
        "per_tap": []
    }
    write_json(args.out_json, report)
    validate_schema(report, args.schema)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_csv).write_text("name,op_type,cos,mse\n", encoding="utf-8")


if __name__ == "__main__":
    main()
