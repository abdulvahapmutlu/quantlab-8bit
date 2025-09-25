# src/viz/error_heatmap.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import load_yaml  # our helper
# Matplotlib defaults: single-plot figures, no custom styles/colors (keeps CI stable)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_parity(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _extract_logits_err(parity: Dict[str, Any], metric: str) -> Tuple[float, float]:
    """
    Returns (mse, 1 - cosine) or (value, NaN) based on what's available.
    """
    logits = parity.get("logits") or parity.get("summary", {}).get("logits")
    if not logits:
        # Try a few common shapes
        mse = parity.get("mse") or parity.get("metrics", {}).get("mse")
        cos = parity.get("cosine") or parity.get("metrics", {}).get("cosine")
        return float(mse) if mse is not None else float("nan"), \
               float(1.0 - cos) if cos is not None else float("nan")

    if metric.lower() == "mse":
        val = logits.get("mse")
        return float(val) if val is not None else float("nan"), float("nan")
    elif metric.lower() in ("cos", "cosine"):
        cos = logits.get("cosine")
        return float(1.0 - cos) if cos is not None else float("nan"), float("nan")
    else:
        # default to MSE
        val = logits.get("mse")
        return float(val) if val is not None else float("nan"), float("nan")

def _extract_layer_channel_errors(parity: Dict[str, Any], metric: str) -> Dict[str, np.ndarray]:
    """
    Tries several common parity JSON shapes to recover per-layer, per-channel errors.

    Expected possibilities:
      - parity["taps"] = [ { "name": "layer1.0.conv1", "mse_per_channel": [...], "cosine_per_channel": [...] }, ... ]
      - parity["layers"] = [ { "layer": "conv1", "per_channel": {"mse": [...], "cosine": [...] } }, ... ]
      - parity["per_layer"] = { "<name>": {"mse_per_channel": [...]} }
    """
    out: Dict[str, np.ndarray] = {}
    mkey = "mse" if metric.lower() == "mse" else "cosine"
    # 1) taps list
    if isinstance(parity.get("taps"), list):
        for t in parity["taps"]:
            lname = t.get("name") or t.get("layer") or t.get("node") or "unk"
            # prefer per-channel
            arr = t.get(f"{mkey}_per_channel")
            if arr is None:
                # maybe nested
                per_ch = (t.get("per_channel") or {})
                arr = per_ch.get(mkey)
            if arr is None:
                # fallback: scalar repeated once (not great but keeps matrix shape)
                scalar = t.get(mkey)
                if scalar is not None:
                    arr = [float(scalar)]
            if arr is not None:
                out[lname] = np.asarray(arr, dtype=float)
    # 2) layers list
    if not out and isinstance(parity.get("layers"), list):
        for t in parity["layers"]:
            lname = t.get("layer") or t.get("name") or "unk"
            per_ch = t.get("per_channel") or {}
            arr = per_ch.get(mkey) or t.get(f"{mkey}_per_channel")
            if arr is None:
                scalar = t.get(mkey)
                if scalar is not None:
                    arr = [float(scalar)]
            if arr is not None:
                out[lname] = np.asarray(arr, dtype=float)
    # 3) dict of layers
    if not out and isinstance(parity.get("per_layer"), dict):
        for lname, blob in parity["per_layer"].items():
            arr = blob.get(f"{mkey}_per_channel")
            if arr is None:
                per_ch = blob.get("per_channel") or {}
                arr = per_ch.get(mkey)
            if arr is None:
                scalar = blob.get(mkey)
                if scalar is not None:
                    arr = [float(scalar)]
            if arr is not None:
                out[lname] = np.asarray(arr, dtype=float)

    # Clean: drop empty or NaN-only rows
    clean: Dict[str, np.ndarray] = {}
    for k, v in out.items():
        v = np.asarray(v, dtype=float)
        if v.size == 0 or np.all(np.isnan(v)):
            continue
        clean[k] = v
    return clean

def _plot_logits_bar(value: float, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 2.2))
    plt.bar(["logits"], [value])
    plt.title(title)
    plt.ylabel("error")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path)
    plt.close()

def _plot_heatmap(layer_errs: Dict[str, np.ndarray], metric: str, title: str, out_path: Path) -> None:
    if not layer_errs:
        return
    # Sort rows by descending max error (makes hot rows bubble to the top)
    items = sorted(layer_errs.items(), key=lambda kv: (np.nanmax(kv[1]) if kv[1].size else -1), reverse=True)
    layers = [k for k, _ in items]
    max_ch = max((v.size for _, v in items), default=1)
    mat = np.zeros((len(items), max_ch), dtype=float)
    mat[:] = np.nan
    for i, (_, arr) in enumerate(items):
        n = min(arr.size, max_ch)
        mat[i, :n] = arr[:n]

    plt.figure(figsize=(12, 6))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, label=f"{metric.upper()} drift")
    plt.yticks(np.arange(len(layers)), layers, fontsize=8)
    plt.xlabel("Channel index")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser("QuantLab-8bit | error heatmap / logits plot")
    ap.add_argument("--dataset-config", type=str, required=False, help="Unused, kept for CLI consistency")
    ap.add_argument("--model-config", type=str, required=False, help="Unused, kept for CLI consistency")
    ap.add_argument("--viz-config", type=str, required=True, help="YAML with visuals.* directories")
    ap.add_argument("--parity-report", type=str, required=True, help="Parity JSON path")
    ap.add_argument("--method", type=str, required=True, help="Label for method (e.g., ptq_static:pcw_symW_asymA_minmax)")
    ap.add_argument("--metric", type=str, default="mse", choices=["mse", "cosine"], help="Error metric to visualize")
    ap.add_argument("--manifest-out", type=str, required=True, help="Where to write a small manifest.json")
    args = ap.parse_args()

    viz_cfg = load_yaml(args.viz_config)
    out_root = Path(viz_cfg.get("visuals", {}).get("error_heatmap", {}).get("out_dir", "artifacts/reports/viz/error_heatmaps"))
    parity_path = Path(args.parity_report)
    parity = _load_parity(parity_path)

    # 1) Logits bar
    logits_val, _ = _extract_logits_err(parity, args.metric)
    # Title & file
    method_sanitized = args.method.replace(":", "_")
    title_logits = f"Error (logits-only) — {Path(parity_path).stem} [{method_sanitized}]"
    out_dir = out_root / method_sanitized
    logits_png = out_dir / "logits_error.png"
    _plot_logits_bar(logits_val, title_logits, logits_png)

    # 2) Layer×channel heatmap (if available)
    layer_errs = _extract_layer_channel_errors(parity, args.metric)
    heatmap_png = out_dir / "error_heatmap.png"
    if layer_errs:
        title_heatmap = f"Per-layer {args.metric.upper()} drift — {Path(parity_path).stem} [{method_sanitized}]"
        _plot_heatmap(layer_errs, args.metric, title_heatmap, heatmap_png)
        heatmap_status = "generated"
    else:
        heatmap_status = "skipped (no layer taps in parity JSON)"

    # 3) Manifest
    manifest = {
        "parity_report": str(parity_path),
        "method": args.method,
        "metric": args.metric,
        "outputs": {
            "logits_bar": str(logits_png),
            "heatmap": str(heatmap_png) if layer_errs else None
        },
        "notes": heatmap_status
    }
    man_path = Path(args.manifest_out)
    _ensure_dir(man_path.parent)
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
