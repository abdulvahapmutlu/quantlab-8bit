import argparse, json, os
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt  # no seaborn

from src.utils.config import load_yaml
from src.viz.utils import ensure_dir, load_viz_cfg, write_manifest

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Weights & Activations histograms")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--viz-config",   required=True)
    ap.add_argument("--fp32-onnx", required=False, help="Optional: ONNX to derive weights/activations")
    ap.add_argument("--method", required=True)
    ap.add_argument("--manifest-out", required=True)
    return ap.parse_args()

def _load_onnx_weights(path: str) -> List[np.ndarray]:
    try:
        import onnx
        from onnx import numpy_helper
        m = onnx.load(path)
        arrs = []
        for init in m.graph.initializer:
            arrs.append(numpy_helper.to_array(init).astype(np.float32).ravel())
        return arrs
    except Exception:
        return []

def _plot_hist(data: np.ndarray, bins: int, title: str, out_path: Path):
    ensure_dir(out_path.parent)
    plt.figure()
    plt.title(title)
    plt.hist(data, bins=bins)
    plt.xlabel("value"); plt.ylabel("count")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main() -> None:
    args = parse_args()
    dcfg: Dict[str, Any] = load_yaml(args.dataset_config)
    mcfg: Dict[str, Any] = load_yaml(args.model_config)
    vcfg: Dict[str, Any] = load_viz_cfg(args.viz_config)["visuals"]
    dataset, model = dcfg["dataset"]["name"], mcfg["model"]["name"]

    items: List[Dict[str, str]] = []

    # Weights hist
    weights = []
    if args.fp32_onnx and Path(args.fp32_onnx).exists():
        ws = _load_onnx_weights(args.fp32_onnx)
        if ws:
            weights = np.concatenate(ws) if len(ws) else np.array([], dtype=np.float32)
    wdir = Path(vcfg["weights_hist"]["out_dir"]) / f"{model}_{dataset}_{args.method}"
    wpng = wdir / "weights_hist.png"
    if len(weights) > 0:
        _plot_hist(weights, int(vcfg["weights_hist"]["bins"]), f"Weights — {model} on {dataset} [{args.method}]", wpng)
        items.append({"title": "weights_hist", "path": wpng.as_posix(), "notes": f"{len(weights)} params"})
    else:
        # Emit an empty-but-valid plot
        _plot_hist(np.array([0.0]), int(vcfg["weights_hist"]["bins"]), f"Weights — {model} on {dataset} [{args.method}]", wpng)
        items.append({"title": "weights_hist", "path": wpng.as_posix(), "notes": "no weights extracted"})

    # Activations hist (logits distribution from ONNX; as a proxy)
    adir = Path(vcfg["activations_hist"]["out_dir"]) / f"{model}_{dataset}_{args.method}"
    apng = adir / "activations_hist.png"
    act_bins = int(vcfg["activations_hist"]["bins"])
    if args.fp32_onnx and Path(args.fp32_onnx).exists():
        import onnxruntime as ort
        from src.data.loaders import build_eval_loader
        loader = build_eval_loader(dcfg, split="val", batch_size=int(vcfg["activations_hist"]["batch_size"]), num_workers=0)
        sess = ort.InferenceSession(args.fp32_onnx, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        vals = []
        seen_batches = 0
        for x, _ in loader:
            y = sess.run([out_name], {in_name: x.numpy().astype(np.float32)})[0]
            vals.append(y.ravel())
            seen_batches += 1
            if seen_batches >= int(vcfg["activations_hist"]["sample_batches"]):
                break
        acts = np.concatenate(vals) if vals else np.array([0.0], dtype=np.float32)
        _plot_hist(acts, act_bins, f"Logits — {model} on {dataset} [{args.method}]", apng)
        items.append({"title": "activations_hist", "path": apng.as_posix(), "notes": f"samples={acts.size}"})
    else:
        _plot_hist(np.array([0.0], dtype=np.float32), act_bins, f"Logits — {model} on {dataset} [{args.method}]", apng)
        items.append({"title": "activations_hist", "path": apng.as_posix(), "notes": "no ONNX provided"})

    write_manifest(args.manifest_out, dataset, model, args.method, items)

if __name__ == "__main__":
    main()
