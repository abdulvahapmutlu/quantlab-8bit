import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.config import load_yaml
from src.viz.utils import ensure_dir, load_viz_cfg, write_manifest
from src.utils.reporting import write_json

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Outlier detector based on parity per-tap drift")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--viz-config", required=True)
    ap.add_argument("--parity-report", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--manifest-out", required=True)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)
    vcfg = load_viz_cfg(args.viz_config)["visuals"]
    dataset, model = dcfg["dataset"]["name"], mcfg["model"]["name"]

    out_dir = Path(vcfg["outliers"]["out_dir"]) / f"{model}_{dataset}_{args.method}"
    ensure_dir(out_dir)
    out_json = out_dir / "outliers.json"

    rep = json.loads(Path(args.parity_report).read_text(encoding="utf-8"))
    per = rep.get("per_tap", [])
    if not per:
        # fallback to a single logits-only item
        items = [{"name":"logits","metric":"mse","value": float(rep.get("summary",{}).get("mse_mean", 0.0) or 0.0)}]
    else:
        # rank by mse (or 1-cos if you prefer)
        scored = []
        for p in per:
            scored.append({"name": p.get("name","tap"), "op_type": p.get("op_type",""), "mse": float(p.get("mse", 0.0) or 0.0), "cos": float(p.get("cos", 0.0) or 0.0)})
        scored.sort(key=lambda d: d["mse"], reverse=True)
        k = int(vcfg["outliers"].get("top_k", 10))
        items = scored[:k]

    write_json(out_json, {"dataset": dataset, "model": model, "method": args.method, "items": items})
    write_manifest(args.manifest_out, dataset, model, args.method, [{"title":"outliers", "path": out_json.as_posix()}])

if __name__ == "__main__":
    main()
