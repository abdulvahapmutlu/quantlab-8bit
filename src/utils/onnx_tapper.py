"""
Minimal ONNX "tap points" helper:
- select_tap_nodes(): pick up to K nodes per allowed op_type and return their first output names.
- add_outputs_for_taps(): promote those intermediate outputs to graph outputs so ORT can fetch them.
"""
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import onnx
from onnx import helper, numpy_helper

def select_tap_nodes(onnx_path: str | Path, tap_ops: List[str], max_per_op: int) -> List[Tuple[str, str]]:
    m = onnx.load(str(onnx_path))
    counts = {op: 0 for op in tap_ops}
    taps: List[Tuple[str, str]] = []
    for n in m.graph.node:
        if n.op_type in tap_ops and counts.get(n.op_type, 0) < max_per_op:
            if len(n.output) > 0:
                taps.append((n.output[0], n.op_type))
                counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return taps

def add_outputs_for_taps(onnx_path_in: str | Path, onnx_path_out: str | Path, tap_output_names: List[str]) -> None:
    m = onnx.load(str(onnx_path_in))

    # Build a set of existing graph outputs to avoid duplicates
    existing = {o.name for o in m.graph.output}

    # We don't know exact shapes/dtypes for intermediates here; leave type empty (ORT tolerates)
    for name in tap_output_names:
        if name in existing:
            continue
        vi = helper.ValueInfoProto()
        vi.name = name
        m.graph.output.append(vi)

    onnx.save(m, str(onnx_path_out))
