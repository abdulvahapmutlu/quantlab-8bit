import argparse
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_yaml
from src.utils.onnx_utils import ensure_parent_dir, export_pytorch_to_onnx, onnx_node_counts
from src.utils.torch_utils import set_seeds, build_model, fuse_bn_if_requested, load_checkpoint
from src.utils.reporting import write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export FP32 PyTorch model to ONNX")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--checkpoint", required=False, default="")
    ap.add_argument("--dynamic-batch-axis", type=int, default=0)
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--export-report", required=True)
    ap.add_argument("--seeds-json", default="repro/seeds.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dcfg: Dict[str, Any] = load_yaml(args.dataset_config)
    mcfg: Dict[str, Any] = load_yaml(args.model_config)

    set_seeds(args.seeds_json)
    num_classes = dcfg["dataset"]["num_classes"]
    model = build_model(mcfg, num_classes)
    model = fuse_bn_if_requested(model, mcfg)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)

    ensure_parent_dir(args.out_onnx)
    rep = export_pytorch_to_onnx(
        model=model,
        out_onnx=args.out_onnx,
        input_size=dcfg["dataset"]["input_size"],
        dynamic_batch_axis=args.dynamic_batch_axis,
        opset=args.opset,
    )
    counts = onnx_node_counts(args.out_onnx)
    write_json(args.export_report, {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "opset": args.opset,
        "input_size": dcfg["dataset"]["input_size"],
        "node_counts": counts,
        "out_onnx": str(Path(args.out_onnx).as_posix())
    })


if __name__ == "__main__":
    main()
