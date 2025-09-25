import argparse
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_yaml
from src.utils.reporting import write_json, validate_schema
from src.data.loaders import build_eval_loader
from src.utils.torch_utils import set_seeds, build_model, load_checkpoint
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parity: Torch FP32 vs ONNX FP32 (logits-only)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--parity-config", required=True)
    ap.add_argument("--checkpoint", required=False, default="")
    ap.add_argument("--fp32-onnx", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=False)
    ap.add_argument("--schema", default="configs/eval/parity_schema.json")
    ap.add_argument("--seeds-json", default="repro/seeds.json")
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
    mcfg = load_yaml(args.model_config)
    pcfg = load_yaml(args.parity_config)["parity"]

    set_seeds(args.seeds_json)

    # Build loaders
    n_samples = int(pcfg["n_samples"])
    bs = int(pcfg["batch_size"])
    loader = build_eval_loader(dcfg, split="val", batch_size=bs, num_workers=0)

    # Build torch model
    model = build_model(mcfg, dcfg["dataset"]["num_classes"])
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
    model.eval()

    # Build ORT session
    import onnxruntime as ort
    sess = ort.InferenceSession(args.fp32_onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    cos_vals, mse_vals, seen = [], [], 0
    with torch.no_grad():
        for x, _ in loader:
            if seen >= n_samples:
                break
            x_t = x.float()
            y_t = model(x_t).cpu().numpy()
            y_o = sess.run([out_name], {inp_name: x.numpy().astype(np.float32)})[0]
            cos_vals.append(cosine_sim(y_t, y_o))
            mse_vals.append(mse(y_t, y_o))
            seen += x.size(0)

    cmin = min(cos_vals) if cos_vals else None
    cmean = float(np.mean(cos_vals)) if cos_vals else None
    mmax = max(mse_vals) if mse_vals else None
    mmean = float(np.mean(mse_vals)) if mse_vals else None

    thr = pcfg["thresholds"]
    passed = (cmin is not None and mmax is not None and cmin >= thr["cos_min_fp32_vs_fp32"] and mmax <= thr["mse_max_fp32_vs_fp32"])

    report = {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "ref": {"method": "torch-fp32", "onnx_path": None},
        "cmp": {"method": "onnx-fp32", "onnx_path": Path(args.fp32_onnx).as_posix()},
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
