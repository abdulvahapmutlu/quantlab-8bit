from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn.functional as F
import numpy as np


def _topk(output: torch.Tensor, target: torch.Tensor, ks=(1, 5)):
    maxk = max(ks)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res


def evaluate_torch_model(model: Any, loader: Any) -> Dict[str, float]:
    model.eval()
    n, top1, top5, loss_sum = 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.float()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            k1, k5 = _topk(logits, y, ks=(1, 5))
            bs = y.size(0)
            n += bs
            top1 += k1
            top5 += k5
            loss_sum += loss.item() * bs
    return {"top1": top1 / n, "top5": top5 / n, "loss": loss_sum / n, "n_eval": n}


def evaluate_onnx_session(session: Any, loader: Any) -> Dict[str, float]:
    import onnxruntime as ort

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    n, top1, top5, loss_sum = 0, 0.0, 0.0, 0.0
    for x, y in loader:
        x_np = x.numpy().astype(np.float32)
        logits = session.run([output_name], {input_name: x_np})[0]
        # soft CE proxy (not exact Torch CE because we don't keep probs), but close enough for sanity
        # we'll just compute topk and skip loss (set to None)
        bs = y.size(0)
        n += bs
        pred = np.argsort(-logits, axis=1)
        top1 += (pred[:, :1] == y.numpy().reshape(-1, 1)).sum()
        top5 += (pred[:, :5] == y.numpy().reshape(-1, 1)).any(axis=1).sum()
    return {"top1": float(top1) / n, "top5": float(top5) / n, "loss": None, "n_eval": n}
