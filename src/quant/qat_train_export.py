import argparse
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_yaml
from src.utils.reporting import write_json
from src.utils.onnx_utils import ensure_parent_dir
from src.utils.torch_utils import set_seeds, build_model, load_checkpoint
from src.data.loaders import build_train_loader, build_eval_loader, build_calib_loader

import torch
import torch.nn as nn
import torch.optim as optim

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QAT: fake-quant finetune → INT8 ONNX (QDQ, ORT)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--train-config", required=True)      # configs/train/qat_finetune.yaml
    ap.add_argument("--quant-config", required=True)      # configs/quant/qat.yaml
    ap.add_argument("--fp32-checkpoint", required=False, default="")
    ap.add_argument("--calib-indices", required=True)     # reuse your committed calibration subset
    ap.add_argument("--opset", type=int, default=19)

    ap.add_argument("--qat-checkpoint-out", required=True)
    ap.add_argument("--onnx-out", required=True)
    ap.add_argument("--export-report", required=True)
    ap.add_argument("--eval-metrics", required=True)

    ap.add_argument("--seeds-json", default="repro/seeds.json")
    return ap.parse_args()

def _make_qat_model(model, qcfg: Dict[str, Any]):
    """
    Prepare QAT using torch.ao.quantization with a conservative config.
    We keep per-tensor activations and per-channel weights (default fbgemm behavior).
    """
    import torch.ao.quantization as tq
    backend = qcfg["qat"].get("backend", "fbgemm")
    torch.backends.quantized.engine = backend

    # base qconfig (you can branch per model later)
    qconfig = tq.get_default_qat_qconfig(backend)

    # attach & prepare
    model.train()
    model_fused = model  # place for fuse if you add it later (fx graph fuse etc.)
    # assign qconfig to the whole model (can still be overridden per-module if you do so elsewhere)
    model_fused.qconfig = qconfig
    prepared = tq.prepare_qat(model_fused, inplace=False)  # fake-quant observers inserted
    return prepared

def _export_fp32_onnx_from_model(model, dcfg: Dict[str, Any], out_path: str, opset: int):
    # Export dequantized/eval FP32 graph
    from src.utils.onnx_utils import export_pytorch_to_onnx, ensure_parent_dir
    ensure_parent_dir(out_path)
    model_cpu = model.cpu().eval()
    export_pytorch_to_onnx(
        model=model_cpu,
        out_onnx=out_path,
        input_size=dcfg["dataset"]["input_size"],
        dynamic_batch_axis=0,
        opset=opset,
    )

def _ort_quantize_static(fp32_onnx: str, int8_onnx: str, calib_loader, act_symmetric: bool, weight_symmetric: bool, per_channel: bool):
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
    class _Reader(CalibrationDataReader):
        def __init__(self, loader):
            self.it = iter(loader)
            self.name = "input"
        def get_next(self):
            import numpy as np
            try:
                x, _ = next(self.it)
            except StopIteration:
                return None
            return {self.name: x.numpy().astype(np.float32)}
    reader = _Reader(calib_loader)
    quantize_static(
        model_input=fp32_onnx,
        model_output=int8_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8 if act_symmetric else QuantType.QUInt8,
        weight_type=QuantType.QInt8 if weight_symmetric else QuantType.QUInt8,
        per_channel=per_channel,
        reduce_range=False,
    )

def main() -> None:
    args = parse_args()
    dcfg: Dict[str, Any] = load_yaml(args.dataset_config)
    mcfg: Dict[str, Any] = load_yaml(args.model_config)
    tcfg: Dict[str, Any] = load_yaml(args.train_config)
    qcfg: Dict[str, Any] = load_yaml(args.quant_config)

    set_seeds(args.seeds_json)

    # Build datasets
    bs = int(tcfg["trainer"]["batch_size"])
    nw = int(tcfg["trainer"]["num_workers"])
    train_loader = build_train_loader(dcfg, batch_size=bs, num_workers=nw)
    val_loader = build_eval_loader(dcfg, split="val", batch_size=bs, num_workers=0)
    calib_loader = build_calib_loader(dcfg, args.calib_indices, batch_size=bs, num_workers=0)

    # Build model and initialize from FP32 checkpoint if provided
    model = build_model(mcfg, dcfg["dataset"]["num_classes"])
    if args.fp32_checkpoint:
        model = load_checkpoint(model, args.fp32_checkpoint)

    # Prepare QAT model
    qat_model = _make_qat_model(model, qcfg)
    qat_model.train()

    # Optimizer & scheduler (minimal, stable)
    opt_cfg = tcfg["trainer"]["optimizer"]
    sch_cfg = tcfg["trainer"]["scheduler"]
    lr = float(opt_cfg.get("lr", 0.02))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    momentum = float(opt_cfg.get("momentum", 0.9))
    optimizer = optim.SGD(qat_model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    total_epochs = int(tcfg["trainer"]["epochs"])
    warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
    min_lr = float(sch_cfg.get("min_lr", lr * 0.01))

    def cosine_lr(epoch):
        import math
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr / lr + 0.5 * (1 - min_lr / lr) * (1 + math.cos(math.pi * t))

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=float(tcfg["trainer"].get("label_smoothing", 0.1)))

    # Simple training loop
    best_top1, best_state = -1.0, None
    for epoch in range(total_epochs):
        # LR step
        for pg in optimizer.param_groups:
            pg["lr"] = lr * cosine_lr(epoch)

        qat_model.train()
        running = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = qat_model(x.float())
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)

        # Eval (dequantized)
        qat_model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in val_loader:
                logits = qat_model(x.float())
                pred = logits.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
            top1 = correct / max(1, total)

        if top1 > best_top1:
            best_top1 = top1
            best_state = {k: v.cpu() for k, v in qat_model.state_dict().items()}

    # Save QAT checkpoint (state dict)
    Path(args.qat_checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else qat_model.state_dict(), args.qat_checkpoint_out)

    # Export FP32 ONNX from QAT-trained (eval mode)
    fp32_tmp = Path(args.onnx_out).with_suffix(".fp32.onnx")
    qat_model.load_state_dict(best_state if best_state is not None else qat_model.state_dict())
    qat_model.eval()
    _export_fp32_onnx_from_model(qat_model, dcfg, str(fp32_tmp), args.opset)

    # INT8 ONNX via ORT static quantization using the QAT-trained model as source
    act_sym = bool(qcfg["qat"].get("act_symmetric", False))
    w_sym = bool(qcfg["qat"].get("weight_symmetric", True))
    per_chan = bool(qcfg["qat"].get("per_channel_weights", True))
    _ort_quantize_static(str(fp32_tmp), args.onnx_out, calib_loader, act_sym, w_sym, per_chan)

    # Quick eval metrics on INT8 ONNX
    import onnxruntime as ort, os, numpy as np
    sess = ort.InferenceSession(args.onnx_out, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    n, top1, top5 = 0, 0, 0
    for x, y in val_loader:
        x_np = x.numpy().astype(np.float32)
        logits = sess.run([out_name], {in_name: x_np})[0]
        pred_top1 = logits.argmax(axis=1)
        n += y.size(0)
        top1 += (pred_top1 == y.numpy()).sum()
        # top5
        idx = np.argsort(-logits, axis=1)[:, :5]
        top5 += (idx == y.numpy().reshape(-1, 1)).any(axis=1).sum()
    size_mb = round(os.path.getsize(args.onnx_out) / (1024 ** 2), 4)

    write_json(args.eval_metrics, {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "method": "qat",
        "top1": float(top1) / max(1, n),
        "top5": float(top5) / max(1, n),
        "loss": None,
        "n_eval": int(n),
        "size_mb": size_mb,
        "notes": "QAT→FP32 ONNX→ORT static quant (QDQ)"
    })

    # Export report
    write_json(args.export_report, {
        "dataset": dcfg["dataset"]["name"],
        "model": mcfg["model"]["name"],
        "opset": args.opset,
        "schedule": {
            "epochs": int(tcfg["trainer"]["epochs"]),
            "warmup_epochs": int(tcfg["trainer"]["scheduler"].get("warmup_epochs", 0))
        },
        "export": {"format": "QDQ", "source": "QAT-trained FP32 ONNX + ORT static quant"},
        "out_onnx": Path(args.onnx_out).as_posix(),
        "notes": "Ranges benefit from QAT training; ORT handles final QDQ placement."
    })

if __name__ == "__main__":
    main()
