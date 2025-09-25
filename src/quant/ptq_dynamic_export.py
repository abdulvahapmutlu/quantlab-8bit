import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import load_yaml
from src.utils.reporting import write_json
from src.utils.onnx_utils import ensure_parent_dir
from src.data.loaders import build_eval_loader
from src.eval.evaluator import evaluate_onnx_session


def _parse_ops(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PTQ (dynamic) export to INT8 ONNX (QDQ)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--checkpoint")
    grp.add_argument("--fp32-onnx")
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--quantize-ops", default="MatMul,Gemm,Linear")
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval-metrics", required=True)
    ap.add_argument("--seeds-json", default="repro/seeds.json")
    return ap.parse_args()


def _export_fp32_from_ckpt(args, dcfg, mcfg) -> str:
    # Call exporter as a subprocess to avoid import cycles
    import subprocess, sys, json, tempfile

    out_dir = Path("artifacts/onnx/fp32") / f"{mcfg['model']['name']}_{dcfg['dataset']['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / "model.onnx"
    rep = out_dir / "export.json"
    cmd = [
        sys.executable, "-m", "src.quant.export_fp32_onnx",
        "--dataset-config", args.dataset_config,
        "--model-config", args.model_config,
        "--checkpoint", args.checkpoint,
        "--dynamic-batch-axis", "0",
        "--opset", str(args.opset),
        "--out-onnx", str(out_onnx),
        "--export-report", str(rep),
    ]
    subprocess.check_call(cmd)
    return str(out_onnx)


def main() -> None:
    args = parse_args()
    dcfg: Dict[str, Any] = load_yaml(args.dataset_config)
    mcfg: Dict[str, Any] = load_yaml(args.model_config)
    ops = _parse_ops(args.quantize_ops)

    fp32_path = args.fp32_onnx or _export_fp32_from_ckpt(args, dcfg, mcfg)

    from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
    ensure_parent_dir(args.out_onnx)
    quantize_dynamic(
        model_input=fp32_path,
        model_output=args.out_onnx,
        op_types_to_quantize=ops,
        weight_type=QuantType.QInt8,
        per_channel=True,
        optimize_model=True,
        quant_format=QuantFormat.QDQ,
    )

    # Graph audit
    report = {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "method": "ptq_dynamic",
        "opset": args.opset,
        "op_types_to_quantize": ops,
        "quant_format": "QDQ",
        "weight_type": "QInt8",
        "per_channel": True,
        "nodes": {"total": None, "quantized": [], "skipped": []},
        "notes": "Conv layers left FP32 by design",
        "out_onnx": Path(args.out_onnx).as_posix(),
    }
    write_json(args.report, report)

    # Quick val eval
    loader = build_eval_loader(dcfg, split="val", batch_size=64, num_workers=0)
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out_onnx, providers=["CPUExecutionProvider"])
    metrics = evaluate_onnx_session(sess, loader)
    import os
    size_mb = round(os.path.getsize(args.out_onnx) / (1024 ** 2), 4)
    write_json(args.eval_metrics, {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "method": "ptq_dynamic",
        **metrics,
        "size_mb": size_mb,
        "notes": "quick val pass"
    })


if __name__ == "__main__":
    main()
