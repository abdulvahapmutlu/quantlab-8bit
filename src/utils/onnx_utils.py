from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import torch


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def export_pytorch_to_onnx(*, model, out_onnx: str | Path, input_size, dynamic_batch_axis: int, opset: int) -> Dict[str, Any]:
    model.eval()
    b = 1
    c, h, w = input_size
    dummy = torch.randn(b, c, h, w, dtype=torch.float32)
    dynamic_axes = {"input": {dynamic_batch_axis: "batch"}, "output": {dynamic_batch_axis: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        str(out_onnx),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return {"dynamic_axes": True, "input_names": ["input"], "output_names": ["output"], "opset": opset}


def onnx_node_counts(onnx_path: str | Path) -> Dict[str, int]:
    import onnx
    m = onnx.load(str(onnx_path))
    counts = {}
    for n in m.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    counts["total"] = sum(counts.values())
    return counts
