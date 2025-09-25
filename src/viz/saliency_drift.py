import argparse
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import load_yaml
from src.viz.utils import ensure_dir, load_viz_cfg, write_manifest
from src.data.loaders import build_eval_loader

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Saliency drift (occlusion): FP32 ONNX vs INT8 ONNX")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--viz-config", required=True)
    ap.add_argument("--fp32-onnx", required=True)
    ap.add_argument("--int8-onnx", required=True)
    ap.add_argument("--method", required=True)  # fp32_vs_ptq | fp32_vs_qat
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--patch", type=int, default=8, help="square occlusion size (pixels)")
    ap.add_argument("--stride", type=int, default=8)
    return ap.parse_args()

def _occlusion_map(sess, x: np.ndarray, patch: int, stride: int) -> np.ndarray:
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    _, c, h, w = x.shape
    base = sess.run([out_name], {in_name: x})[0]  # [1, num_classes]
    base_max = base.max(axis=1)[0]
    heat = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h - patch + 1, stride):
        for j in range(0, w - patch + 1, stride):
            x_occ = x.copy()
            x_occ[:, :, i:i+patch, j:j+patch] = 0.0
            y = sess.run([out_name], {in_name: x_occ})[0]
            drop = max(0.0, float(base_max - y.max(axis=1)[0]))
            heat[i:i+patch, j:j+patch] += drop
    # normalize
    if heat.max() > 0:
        heat /= heat.max()
    return heat

def main() -> None:
    args = parse_args()
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)
    vcfg = load_viz_cfg(args.viz_config)["visuals"]
    dataset, model = dcfg["dataset"]["name"], mcfg["model"]["name"]

    import onnxruntime as ort
    s_fp = ort.InferenceSession(args.fp32_onnx, providers=["CPUExecutionProvider"])
    s_i8 = ort.InferenceSession(args.int8_onnx, providers=["CPUExecutionProvider"])

    loader = build_eval_loader(dcfg, split="val", batch_size=1, num_workers=0)
    it = iter(loader)

    out_dir = Path(vcfg["saliency"]["out_dir"]) / f"{model}_{dataset}_{args.method}"
    ensure_dir(out_dir)
    out_png = out_dir / "saliency_grid.png"

    n = int(args.n_images)
    cols = 3
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_2d(axes)

    count = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            try:
                x, _ = next(it)
            except StopIteration:
                ax.axis("off"); continue
            x_np = x.numpy().astype(np.float32)
            h_fp = _occlusion_map(s_fp, x_np, args.patch, args.stride)
            h_i8 = _occlusion_map(s_i8, x_np, args.patch, args.stride)
            # drift = absolute difference
            drift = np.abs(h_fp - h_i8)
            # show drift heatmap (gray)
            ax.imshow(drift, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            count += 1
            if count >= n:
                break
        if count >= n:
            break

    fig.suptitle(f"Saliency drift (occlusion) — {model} on {dataset} [{args.method}]")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    write_manifest(args.manifest_out, dataset, model, args.method, [{"title":"saliency_grid", "path": out_png.as_posix()}])

if __name__ == "__main__":
    main()
