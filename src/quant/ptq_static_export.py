import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import load_yaml
from src.utils.reporting import write_json
from src.utils.onnx_utils import ensure_parent_dir
from src.data.loaders import build_calib_loader, build_eval_loader
from src.eval.evaluator import evaluate_onnx_session


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PTQ (static) export to INT8 ONNX with Q/DQ")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--quant-config", required=True)
    ap.add_argument("--recipe-id", required=True)
    ap.add_argument("--checkpoint", required=False, default="")
    ap.add_argument("--calib-indices", required=True)
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--calib-report", required=True)
    ap.add_argument("--eval-metrics", required=True)
    ap.add_argument("--seeds-json", default="repro/seeds.json")
    return ap.parse_args()


def _find_recipe(qcfg: Dict[str, Any], recipe_id: str) -> Optional[Dict[str, Any]]:
    for r in qcfg.get("ptq_static", {}).get("recipes", []):
        if r.get("id") == recipe_id:
            return r
    return None


class _CalibDataReader:
    """ONNX Runtime calibration data reader compatible with quantize_static()."""
    def __init__(self, loader, input_name: str = "input"):
        self.loader = iter(loader)
        self.input_name = input_name
        self._leftover = None

    def get_next(self):
        try:
            import numpy as np
            if self._leftover is not None:
                data = self._leftover
                self._leftover = None
                return data
            x, _ = next(self.loader)
            x = x.numpy().astype(np.float32)
            return {self.input_name: x}
        except StopIteration:
            return None


def _export_fp32_from_ckpt(args, dcfg, mcfg) -> str:
    """Always ensure a fresh FP32 ONNX exists; export from checkpoint if provided, else from pretrained config."""
    import subprocess, sys
    out_dir = Path("artifacts/onnx/fp32") / f"{mcfg['model']['name']}_{dcfg['dataset']['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / "model.onnx"
    rep = out_dir / "export.json"

    cmd = [
        sys.executable, "-m", "src.quant.export_fp32_onnx",
        "--dataset-config", args.dataset_config,
        "--model-config", args.model_config,
        "--dynamic-batch-axis", "0",
        "--opset", str(args.opset),
        "--out-onnx", str(out_onnx),
        "--export-report", str(rep),
    ]
    # Only pass --checkpoint if non-empty (avoids PowerShell empty-arg quirks)
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]

    subprocess.check_call(cmd)
    return str(out_onnx)


def main() -> None:
    args = parse_args()
    dcfg: Dict[str, Any] = load_yaml(args.dataset_config)
    mcfg: Dict[str, Any] = load_yaml(args.model_config)
    qcfg: Dict[str, Any] = load_yaml(args.quant_config)

    recipe = _find_recipe(qcfg, args.recipe_id)
    if recipe is None:
        raise SystemExit(f"Recipe '{args.recipe_id}' not found in {args.quant_config}")

    # Export/refresh FP32 ONNX (from ckpt if provided; otherwise pretrained)
    fp32_onnx = _export_fp32_from_ckpt(args, dcfg, mcfg)

    # Build calibration loader
    default_set = qcfg["ptq_static"]["default_settings"]
    calib_bs = int(default_set.get("calibrate_batch_size", 64))
    loader = build_calib_loader(dcfg, args.calib_indices, batch_size=calib_bs, num_workers=0)

    # Map our recipe options
    per_channel_weights = bool(recipe.get("per_channel_weights", True))
    weight_symmetric = bool(recipe.get("weight_symmetric", True))
    act_symmetric = bool(recipe.get("act_symmetric", False))
    range_estimator = str(recipe.get("range_estimator", "minmax")).lower()
    percentile_clip = recipe.get("percentile_clip", None)

    # ORT quantization imports (version-agnostic) + optional calibration method
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    try:
        from onnxruntime.quantization.calibrate import CalibrationMethod as CM
        _have_cm = True
    except Exception:
        CM = None  # older ORT
        _have_cm = False

    # Choose calibration method if supported (fallback to MinMax or default)
    cal_method = None
    if _have_cm:
        cal_method = CM.MinMax
        if range_estimator in ("mse", "kl"):
            cal_method = CM.Entropy

    # Prepare data reader
    reader = _CalibDataReader(loader, input_name="input")

    # Types (direct, no QDQQuantizerConfig)
    activation_type = QuantType.QInt8 if act_symmetric else QuantType.QUInt8
    weight_type     = QuantType.QInt8 if weight_symmetric else QuantType.QUInt8
    reduce_range    = False

    # Ensure destination dir exists
    ensure_parent_dir(args.out_onnx)

    # Build kwargs guarded by the current ORT function signature
    import inspect
    sig_params = set(inspect.signature(quantize_static).parameters.keys())
    qkwargs = {
        "model_input": fp32_onnx,
        "model_output": args.out_onnx,
        "calibration_data_reader": reader,
        "quant_format": QuantFormat.QDQ,
        "per_channel": per_channel_weights,
        "activation_type": activation_type,
        "weight_type": weight_type,
        "reduce_range": reduce_range,
    }
    # Only pass if supported by this ORT build
    if "calibration_method" in sig_params and cal_method is not None:
        qkwargs["calibration_method"] = cal_method
    if "opset" in sig_params:
        qkwargs["opset"] = args.opset

    # Quantize
    quantize_static(**qkwargs)

    # Calibration report (basic)
    calib_report = {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "recipe_id": args.recipe_id,
        "opset": args.opset,
        "quant": {
            "weight_bits": qcfg["ptq_static"].get("weight_bits", 8),
            "activation_bits": qcfg["ptq_static"].get("activation_bits", 8),
            "per_channel_weights": per_channel_weights,
            "weight_symmetric": weight_symmetric,
            "act_symmetric": act_symmetric,
            "range_estimator": range_estimator,
            "percentile_clip": percentile_clip
        },
        "calibration": {
            "indices_file": args.calib_indices,
            "batches": default_set.get("calibrate_batches", None),
            "batch_size": calib_bs,
            "num_samples_seen": None,
            "layer_stats_file": None
        },
        "graph": {"num_nodes": None, "num_qdq_pairs": None, "unsupported_nodes": [], "notes": ""},
        "out_onnx": Path(args.out_onnx).as_posix()
    }
    write_json(args.calib_report, calib_report)

    # Quick val evaluation
    import onnxruntime as ort, os
    loader_val = build_eval_loader(dcfg, split="val", batch_size=64, num_workers=0)
    sess = ort.InferenceSession(args.out_onnx, providers=["CPUExecutionProvider"])
    metrics = evaluate_onnx_session(sess, loader_val)
    size_mb = round(os.path.getsize(args.out_onnx) / (1024 ** 2), 4)
    write_json(args.eval_metrics, {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "method": f"ptq_static:{args.recipe_id}",
        **metrics,
        "size_mb": size_mb,
        "notes": "quick val pass"
    })


if __name__ == "__main__":
    main()
